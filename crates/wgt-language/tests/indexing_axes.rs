//! Indexing changes the selected semantic axis, not the names of retained axes.
use wgt_language::adaptive::{self, Instruction};

fn source(input: usize, output: usize, body: &str) -> String {
    format!("wgt 0.5\nmodel IndexAxes\ninput x:f32[{input}]\noutput y:f32[{output}]\nfn forward(x) {{ {body} }}")
}

#[test]
fn gather_and_embedding_replace_vocabulary_axis_with_named_index_axis() {
    for operation in ["gather", "embedding"] {
        let text=source(6,4,&format!("
            let table:f32[vocabulary=3,features=2]=x;
            let ids:f32[tokens=2]=[2,0];
            let rows={operation}(table,ids);
            let normalized=softmax(rows,\"features\");
            return permute(normalized,[features,tokens]) as [4];"));
        let program=adaptive::parse(&text).unwrap();
        assert!(program.instructions.iter().any(|i| matches!(i,Instruction::Permute{axes,..} if axes==&[1,0])));
        assert!(adaptive::parse(&text.replace("[features,tokens]","[features,vocabulary]")).is_err(),"selected source axis must be replaced");
        assert!(adaptive::parse(&text.replace("ids:f32[tokens=2]","ids:f32[features=2]")).is_err(),"result axis names must stay unique");
        // Anonymous index vectors clear the selected vocabulary axis, retaining features.
        let anonymous=text.replace("ids:f32[tokens=2]","ids:f32[2]")
            .replace("return permute(normalized,[features,tokens]) as [4];","return normalized as [4];");
        adaptive::parse(&anonymous).unwrap();
    }
}

#[test]
fn gather_nonleading_axis_preserves_other_names() {
    let text=source(6,4,"
        let values:f32[batch=2,vocabulary=3]=x;
        let ids:f32[tokens=2]=[2,0];
        let rows=gather(values,ids,1);
        let normalized=softmax(rows,\"batch\");
        return permute(normalized,[tokens,batch]) as [4];");
    adaptive::parse(&text).unwrap();
    assert!(adaptive::parse(&text.replace("ids:f32[tokens=2]","ids:f32[batch=2]")).is_err());
}

#[test]
fn concat_merges_names_from_both_inputs_including_concat_axis() {
    for (left,right) in [("rows=2,3","1,features=3"),("2,features=3","rows=1,3"),("rows=2,features=3","rows=1,features=3")] {
        let text=source(9,9,&format!("
            let left:f32[{left}]=x[0:6];
            let right:f32[{right}]=x[6:9];
            let combined=concat(left,right);
            return permute(combined,[features,rows]) as [9];"));
        adaptive::parse(&text).unwrap();
    }
    let text=source(9,9,"
        let left:f32[rows=2,features=3]=x[0:6];
        let right:f32[rows=1,features=3]=x[6:9];
        return concat(left,right) as [9];");
    assert!(adaptive::parse(&text.replace("right:f32[rows=1","right:f32[other=1")).is_err());
    assert!(adaptive::parse(&text.replace("right:f32[rows=1,features=3]","right:f32[rows=1,other=3]")).is_err());
    let duplicate=text.replace("left:f32[rows=2,features=3]","left:f32[rows=2,3]")
        .replace("right:f32[rows=1,features=3]","right:f32[1,rows=3]");
    assert!(adaptive::parse(&duplicate).is_err());
}

#[test]
fn dynamic_slice_preserves_the_input_axis_name() {
    let text=source(4,2,"
        let values:f32[tokens=4]=x;
        let window=dynamic_slice(values,[1],2);
        return softmax(window,\"tokens\");");
    let program=adaptive::parse(&text).unwrap();
    assert!(program.instructions.iter().any(|i| matches!(i,Instruction::SoftmaxAxis{axis:0,..})));
}
