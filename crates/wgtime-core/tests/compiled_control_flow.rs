use wgtime_core::compiled_forward::CompiledForward;
use wgt_language::{adaptive::{Instruction as I,Program,Register},categorical::VectorType};
fn program(instructions:Vec<I>)->Program {
    Program{observations:Vec::new(),version:1,output_type:VectorType::F32,input:"x".into(),registers:vec![Register{name:"x".into(),len:1,shape:vec![1]}],parameters:vec![],instructions,loss:None}
}
#[test]
fn native_branches_reach_actual_return_and_loops_exhaust_budget() {
    let plan=program(vec![I::Branch{condition:"x".into(),positive:"yes".into(),otherwise:"no".into()},I::Label{name:"yes".into()},I::Return{value:"x".into()},I::Label{name:"no".into()},I::Return{value:"x".into()}]);
    let mut code=CompiledForward::compile(&plan,4096).unwrap();
    for (input,expected_return) in [(1.,2),(-1.,4)] {
        let mut returned=None;
        let value=code.run_with_hook(&[input],&[],10,false,&mut |index,_|{if matches!(plan.instructions[index],I::Return{..}){returned=Some(index)}Ok(())}).unwrap();
        assert_eq!(value,vec![input]);assert_eq!(returned,Some(expected_return));
    }
    let endless=program(vec![I::Label{name:"again".into()},I::Jump{target:"again".into()},I::Return{value:"x".into()}]);
    let mut code=CompiledForward::compile(&endless,4096).unwrap();
    assert!(code.run_budget(&[1.],&[],12).unwrap_err().to_string().contains("budget"));assert_eq!(code.last_steps(),12);
}
