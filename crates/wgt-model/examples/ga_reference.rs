//! Test oracle for cross-language checkpoints; no model execution or service.
use serde::Deserialize;
use wgt::evolution::{GeneticAlgorithm, Score};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    checkpoint: GeneticAlgorithm,
    scores: Option<Vec<Score>>,
}

fn main() -> wgt::Result<()> {
    let request: Request = serde_json::from_reader(std::io::stdin().lock())?;
    let mut search = request.checkpoint;
    if let Some(scores) = request.scores {
        search.tell(&scores)?;
    }
    serde_json::to_writer(std::io::stdout().lock(), &search)?;
    Ok(())
}
