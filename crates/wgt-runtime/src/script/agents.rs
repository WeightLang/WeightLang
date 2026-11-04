//! Dedicated agent/environment instances. Methods dispatch only for these handles.
use super::{number, records::RecordBatch, Runtime, Value};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentPlan { pub learner: Option<String>, pub critic: bool }
#[derive(Clone, Debug)]
pub struct Agent { pub policy: Value, pub critic: Option<Value>, pub learner: Option<String> }
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnvironmentPlan { pub observation: usize, pub action: usize, pub feedback: usize, pub reset: String, pub step: String }
#[derive(Clone, Debug)]
pub struct Environment { pub plan: EnvironmentPlan, pub initial: Value, pub state: Value, pub observation: Value, pub seed: u64, pub rng: u64, pub schema: super::record_schema::BatchSchema, pub ready: bool, pub done: bool }

fn fields(value: Value) -> Result<std::collections::BTreeMap<String,Value>> {
    let Value::Object(fields)=value else {bail!("environment handler must return a record")}; Ok(fields)
}
fn vector(value: &Value, width: usize, role: &str) -> Result<()> {
    let values=value.vector()?;ensure!(values.len()==width,"environment {role} width must be {width}");Ok(())
}
fn state(value:&Value, remaining:&mut u64)->Result<()> {
    // Reuse the finite data, shape and storage bounds of record rows.
    let mut batch=RecordBatch::default();batch.push(vec![("state".into(),value.clone())],remaining)
}
impl Runtime {
    pub fn agent_policy(&self, value:&Value)->Result<Value> {
        match value {Value::Agent(i)=>Ok(self.agents[*i].policy.clone()),Value::Model(_)|Value::Adaptive(_)=>Ok(value.clone()),_=>bail!("expected agent or model")}
    }
    pub fn resolve_learner(&self, target:&Value, explicit:Option<&str>)->Result<String> {
        let name=if let Some(name)=explicit {name.to_owned()} else {
            let Value::Agent(i)=target else {bail!("model learn requires using a learner")};
            self.agents[*i].learner.clone().context("agent has no default learner; specify using")?
        };
        self.validate_learner(&name)?;Ok(name)
    }
    pub(super) fn agent_builtin(&mut self,name:&str,args:&[Value])->Result<Value> {
        match name {
            "__agent"=>{
                let plan:AgentPlan=serde_json::from_str(args[0].text()?)?;
                let policy=args[1].clone(); self.agent_policy(&policy)?;
                ensure!(!matches!(policy,Value::Agent(_)),"agent policy must be a model");
                let critic=if plan.critic {let value=args[2].clone();self.agent_policy(&value)?;ensure!(!matches!(value,Value::Agent(_)),"critic must be a model");Some(value)}else{None};
                if let Some(learner)=&plan.learner {self.validate_learner(learner)?;}
                let index=self.agents.len();self.agents.push(Agent{policy,critic,learner:plan.learner});Ok(Value::Agent(index))
            }
            "__environment"=>{
                let plan:EnvironmentPlan=serde_json::from_str(args[0].text()?)?;
                state(&args[1],&mut self.remaining)?;
                let seed=args[2].integer()? as u64;
                self.validate_learner(&plan.reset)?;self.validate_learner(&plan.step)?;
                let index=self.environments.len();self.environments.push(Environment{plan,initial:args[1].clone(),state:args[1].clone(),observation:Value::Null,seed,rng:seed,schema:Default::default(),ready:false,done:false});
                if let Err(error)=self.instance_method(Value::Environment(index),"reset",vec![]) {self.environments.pop();return Err(error);}
                Ok(Value::Environment(index))
            }
            "__record"=>{
                ensure!(args.len()%2==0,"invalid record fields");
                let mut values=std::collections::BTreeMap::new();
                for pair in args.chunks_exact(2) {ensure!(values.insert(pair[0].text()?.to_owned(),pair[1].clone()).is_none(),"duplicate record field");}
                let value=Value::Object(values);state(&value,&mut self.remaining)?;Ok(value)
            }
            _=>bail!("unknown agent runtime operation"),
        }
    }
    pub fn instance_method(&mut self, instance:Value, method:&str,args:Vec<Value>)->Result<Value> {
        match instance {
            Value::Agent(i)=>{
                ensure!(method=="act" && args.len()==1,"agent supports act(observation)");
                self.sample_model(self.agents[i].policy.clone(),args[0].clone())
            }
            Value::Environment(i)=>{
                let env=self.environments[i].clone();
                let outer_rng=self.rng;self.rng=if method=="reset" {env.seed}else{env.rng};
                let output=(|| -> Result<Value> {Ok(match method {
                    "reset"=>{
                        ensure!(args.is_empty(),"reset expects no arguments");
                        self.call(&env.plan.reset,vec![env.initial.clone(),number(env.seed as f64)?])?
                    }
                    "step"=>{
                        ensure!(args.len()==1,"step expects one action");
                        ensure!(env.ready && !env.done,"environment requires reset before step or after termination/truncation");
                        vector(&args[0],env.plan.action,"action")?;
                        self.call(&env.plan.step,vec![env.state.clone(),args[0].clone()])?
                    }
                    _=>bail!("environment supports reset() and step(action)"),
                })})();
                let next_rng=self.rng;self.rng=outer_rng;
                let values=fields(output?)?;
                let expected=if method=="reset" {2}else{5};
                ensure!(values.len()==expected,"environment handler returned unexpected fields");
                let next=values.get("state").context("environment result requires state")?;
                let mut validation=RecordBatch::default();validation.push(vec![("state".into(),next.clone())],&mut self.remaining)?;
                if let Some(schema)=env.schema.fields() {ensure!(Some(schema)==validation.schema.fields(),"environment state schema must remain fixed");}
                let observation=values.get("observation").context("environment result requires observation")?;
                vector(observation,env.plan.observation,"observation")?;
                let mut done=false;
                if method=="step" {
                    vector(values.get("feedback").context("environment result requires feedback")?,env.plan.feedback,"feedback")?;
                    let terminated=values.get("terminated").context("environment result requires terminated")?.boolean()?;
                    let truncated=values.get("truncated").context("environment result requires truncated")?.boolean()?;
                    done=terminated || truncated;
                }
                // Commit only after every output field validates. External effects
                // performed by a handler are outside this data transaction.
                self.environments[i].state=next.clone();self.environments[i].observation=observation.clone();self.environments[i].rng=next_rng;self.environments[i].schema=validation.schema;self.environments[i].ready=true;self.environments[i].done=done;
                if method=="reset" {Ok(observation.clone())} else {Ok(Value::Object(values))}
            }
            _=>bail!("method calls require an agent or environment instance"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    #[test]
    fn failed_collection_unlocks_and_does_not_call_learner() {
        let source=r#"wgt 0.5
        model Policy
        input x:f32[1]
        output y:f32[1]
        parameter w:f32[1]=[1]
        fn forward(x){return w*x;}
        fn loss(p,t){return square(p-t).mean;}
        fn update(model,batch){metric("called",true);return 0;}
        fn fail(model){learn model using update {yield {x:[1]};assert false,"stop";}}
        fn main(){return Policy("sgd");}"#;
        let (mut runtime,model)=super::super::execute(source,Path::new("."),100000,Default::default()).unwrap();
        assert!(runtime.call("fail",vec![model.clone()]).is_err());
        assert!(!runtime.metrics.contains_key("called"));
        let token=runtime.begin_collection(&model).unwrap();drop(token);
        assert!(Value::List(vec![Value::Agent(0)]).json().is_err());
        assert!(Value::Object(std::collections::BTreeMap::from([("env".into(),Value::Environment(0))])).json().is_err());
        assert!(super::super::async_runtime::data(&Value::List(vec![Value::Agent(0)])).is_err());
        assert!(super::super::async_runtime::data(&Value::Object(std::collections::BTreeMap::from([("env".into(),Value::Environment(0))]))).is_err());
    }
    #[test]
    fn rejected_step_preserves_state_observation_done_and_rng() {
        let source=r#"wgt 0.5
        environment Counter {observation:f32[1];action:f32[1];feedback:f32[1];reset:reset_env;step:step_env;}
        fn reset_env(state,seed){return {state:state,observation:[state]};}
        fn step_env(state,action){let draw=random();if action[0]==0 {return {state:[state],observation:[draw],feedback:[0],terminated:true,truncated:false};}return {state:state+1,observation:[draw],feedback:[0],terminated:false,truncated:true};}
        fn main(){return Counter(0,7);}"#;
        let (mut runtime,value)=super::super::execute(source,Path::new("."),100000,Default::default()).unwrap();
        let Value::Environment(index)=value else {panic!()};
        let before=runtime.environments[index].clone();
        assert!(runtime.instance_method(Value::Environment(index),"step",vec![Value::List(vec![Value::Number(0.)])]).is_err());
        let after=&runtime.environments[index];
        assert_eq!(after.state,before.state);assert_eq!(after.observation,before.observation);
        assert_eq!(after.rng,before.rng);assert_eq!(after.done,before.done);
        let result=runtime.instance_method(Value::Environment(index),"step",vec![Value::List(vec![Value::Number(1.)])]).unwrap();
        let Value::Object(fields)=result else {panic!()};
        let mut rng=7;let expected=crate::training::random(&mut rng);
        assert_eq!(fields["observation"],Value::List(vec![Value::Number(expected)]));
        assert_eq!(fields["terminated"],Value::Bool(false));assert_eq!(fields["truncated"],Value::Bool(true));
        assert!(runtime.instance_method(Value::Environment(index),"step",vec![Value::List(vec![Value::Number(1.)])]).is_err());
    }
}
