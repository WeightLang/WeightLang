use serde::{Serialize,Deserialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentPlan { pub learner: Option<String>, pub critic: bool }
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnvironmentPlan { pub observation: usize, pub action: usize, pub feedback: usize, pub reset: String, pub step: String }
