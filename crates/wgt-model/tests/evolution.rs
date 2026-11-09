mod support;
use serde_json::{Value, json};
use wgt::{
    InstanceOptions, ModelDefinition,
    evolution::{
        Candidate, EvaluationOptions, GAConfig, GeneDomain, GeneticAlgorithm, GenomeSchema,
        Objective, Score, evaluate_isolated, fit_forward_forward,
    },
};
use wgt_language::model::ModelPlan;

const DOMAINS: &str = r#"wgt 0.5
model Domains
parameter matrix:f32[2,2]=[1,2,3,4]
parameter free:f32[1]=[0]
learn bounded:float in -1..=1=0;
learn positive:float > 0=1;
learn integer:int in 0..=5=2;
learn gate:bool=true;
frozen constant:f32[1]=[2]
parameter probability:prob=0.25
fn observe(x:f32[1])->f32[1]{return x+free;}
"#;
const SCALAR: &str = "wgt 0.5\nmodel Search\nlearn weight:float in -4..=4=0;\nfn forward(x:f32[1])->f32[1]{return x*weight;}\n";
const PARITY: &str = "wgt 0.5\nmodel Parity\nparameter weight:f32[2]=[0.25,-0.5]\nfn forward(x:f32[2])->f32[2]{return x*weight;}\n";
const CELL: &str = r#"wgt 0.5
model Cell
parameter gain:f32[1]=[1]
state activity:f32[1]=[0];
fn step(x:f32[1])->f32[1]{ activity <- activity+x; return activity*gain; }
fn adapt(x:f32[1]){ gain <- x; }
fn reset(){ activity <- [0]; }
fn reset_bad(){ gain <- [1]; }
fn fail(x:f32[1])->f32[1]{ activity <- activity+x; return log([-1]); }
"#;

fn schema(source: &str) -> GenomeSchema {
    let definition = ModelDefinition::parse(source).unwrap();
    let parameters = match definition.instantiate_plan().unwrap() {
        ModelPlan::Graph(graph) => graph.initial_parameters().unwrap(),
        ModelPlan::Forward(program) => program.parameters,
        ModelPlan::Operations(plan) => plan.parameters,
    }
    .into_iter()
    .map(|p| (p.name, p.shape, p.data))
    .collect::<Vec<_>>();
    GenomeSchema::from_definition(&definition, &parameters).unwrap()
}

fn config() -> GAConfig {
    GAConfig {
        population_size: 24,
        elite_count: 2,
        tournament_size: 3,
        mutation_rate: 1.,
        mutation_scale: 0.5,
        crossover_rate: 0.75,
        integer_step: 2,
        ..Default::default()
    }
}

fn quadratic_scores(search: &GeneticAlgorithm) -> Vec<Score> {
    search
        .ask()
        .iter()
        .map(|candidate| Score {
            id: candidate.id,
            fitness: (candidate.values[0] as f64 - 2.5).powi(2),
        })
        .collect()
}

#[test]
fn python_and_rust_share_the_seeded_candidate_sequence() {
    // Independently computed by the Python GA from the same native definition.
    let mut search = GeneticAlgorithm::new(
        schema(PARITY),
        GAConfig {
            population_size: 4,
            elite_count: 1,
            tournament_size: 2,
            mutation_rate: 1.,
            mutation_scale: 0.25,
            crossover_rate: 1.,
            integer_step: 1,
            objective: Objective::Minimize,
        },
        7,
    )
    .unwrap();
    let bits = |search: &GeneticAlgorithm| {
        search
            .ask()
            .iter()
            .map(|c| c.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        bits(&search),
        [
            vec![1048576000, 3204448256],
            vec![1007257499, 3203056917],
            vec![1040149242, 3205890453],
            vec![1045661509, 3196732849],
        ]
    );
    let checkpoint: Value = serde_json::from_str(&search.checkpoint_json().unwrap()).unwrap();
    assert_eq!(checkpoint["rng_state"], json!(7681369315911520515_u64));
    let scores = search
        .ask()
        .iter()
        .map(|c| Score {
            id: c.id,
            fitness: c.id as f64,
        })
        .collect::<Vec<_>>();
    search.tell(&scores).unwrap();
    assert_eq!(
        bits(&search),
        [
            vec![1048576000, 3204448256],
            vec![1051504744, 3205753164],
            vec![1021192971, 3207714226],
            vec![3176086531, 3207610257],
        ]
    );
    let checkpoint: Value = serde_json::from_str(&search.checkpoint_json().unwrap()).unwrap();
    assert_eq!(checkpoint["rng_state"], json!(14970076879386038200_u64));
    assert_eq!(checkpoint["next_id"], 8);
}

#[test]
fn schema_orders_full_storage_and_excludes_frozen_probability_and_shared_duplicates() {
    let schema = schema(DOMAINS);
    assert_eq!(
        schema
            .parameters()
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>(),
        [
            "bounded",
            "constant",
            "free",
            "gate",
            "integer",
            "matrix",
            "positive",
            "probability"
        ]
    );
    assert_eq!(schema.width(), 9);
    assert!(!schema.parameters()[1].updatable);
    assert!(!schema.parameters()[7].updatable);
    assert_eq!(schema.parameters()[5].shape, [2, 2]);
    let candidate = Candidate {
        id: 0,
        values: schema.initial_values(),
    };
    let updates = schema.candidate_parameters(&candidate).unwrap();
    assert_eq!(
        updates.iter().map(|p| p.0.as_str()).collect::<Vec<_>>(),
        ["bounded", "free", "gate", "integer", "matrix", "positive"]
    );

    let shared = "wgt 0.5\nmodel Shared\ninput x:f32[1]\noutput y:f32[1]\nmodel Block {\ninput x:f32[1]\noutput y:f32[1]\nparameter weight:f32[1]=[2]\nfn forward(x){return x*weight;}\n}\nblocks:Block[2];\nfn forward(x){return blocks[0](blocks[0](x))+blocks[1](x);}\n";
    let shared = schema_from_source(shared);
    assert_eq!(shared.width(), 2);
    assert_eq!(shared.parameters().len(), 2);
}

fn schema_from_source(source: &str) -> GenomeSchema {
    schema(source)
}

#[test]
fn initial_values_and_discrete_candidates_require_exact_valid_storage() {
    let definition = ModelDefinition::parse(DOMAINS).unwrap();
    let original = schema(DOMAINS)
        .parameters()
        .iter()
        .map(|p| (p.name.clone(), p.shape.clone(), p.initial.clone()))
        .collect::<Vec<_>>();
    for kind in 0..5 {
        let mut values = original.clone();
        match kind {
            0 => {
                values.pop();
            }
            1 => values.push(values[0].clone()),
            2 => values[0].1 = vec![2],
            3 => values.iter_mut().find(|p| p.0 == "integer").unwrap().2[0] = 1.5,
            _ => values[0].2[0] = f32::NAN,
        }
        assert!(GenomeSchema::from_definition(&definition, &values).is_err());
    }
    let schema = schema(DOMAINS);
    let mut candidate = Candidate {
        id: 0,
        values: schema.initial_values(),
    };
    candidate.values[2] = 0.5;
    assert!(schema.validate(&candidate).is_err());
    candidate.values = schema.initial_values();
    candidate.values[8] = 0.;
    assert!(schema.validate(&candidate).is_err());
}

#[test]
fn domain_mutations_remain_valid_and_ids_never_repeat() {
    let mut search = GeneticAlgorithm::new(schema(DOMAINS), config(), u64::MAX).unwrap();
    let mut ids = std::collections::BTreeSet::new();
    let mut saw_bool_change = false;
    let mut saw_int_change = false;
    for generation in 0..12 {
        assert_eq!(search.generation(), generation);
        for candidate in search.ask() {
            assert!(ids.insert(candidate.id));
            search.schema().validate(&candidate).unwrap();
            saw_bool_change |= candidate.values[2] == 0.;
            saw_int_change |= candidate.values[3] != 2.;
        }
        let scores = search
            .ask()
            .iter()
            .map(|c| Score {
                id: c.id,
                fitness: c.values.iter().map(|v| *v as f64).sum(),
            })
            .collect::<Vec<_>>();
        search.tell(&scores).unwrap();
    }
    assert!(saw_bool_change && saw_int_change);
    assert_eq!(
        search
            .schema()
            .parameters()
            .iter()
            .find(|p| p.name == "integer")
            .unwrap()
            .domain,
        GeneDomain::Int { low: 0, high: 5 }
    );
}

#[test]
fn selection_improves_fitness_and_objective_direction_is_explicit() {
    let mut minimizing = GeneticAlgorithm::new(schema(SCALAR), config(), 7).unwrap();
    let mut maximizing = GeneticAlgorithm::new(
        schema(SCALAR),
        GAConfig {
            objective: Objective::Maximize,
            ..config()
        },
        7,
    )
    .unwrap();
    for _ in 0..30 {
        let scores = quadratic_scores(&minimizing);
        let negative = scores
            .iter()
            .map(|s| Score {
                id: s.id,
                fitness: -s.fitness,
            })
            .collect::<Vec<_>>();
        minimizing.tell(&scores).unwrap();
        maximizing.tell(&negative).unwrap();
        assert_eq!(minimizing.ask(), maximizing.ask());
    }
    assert!(minimizing.best().unwrap().fitness < 0.001);
    assert_eq!(
        minimizing.best().unwrap().fitness,
        -maximizing.best().unwrap().fitness
    );
    assert_eq!(
        minimizing.best().unwrap().candidate,
        maximizing.best().unwrap().candidate
    );
}

#[test]
fn asking_is_repeatable_and_bad_or_stale_score_batches_are_atomic() {
    let mut search = GeneticAlgorithm::new(schema(SCALAR), config(), 7).unwrap();
    assert_eq!(search.ask(), search.ask());
    let good = quadratic_scores(&search);
    for kind in 0..5 {
        let mut bad = good.clone();
        match kind {
            0 => {
                bad.pop();
            }
            1 => bad[1].id = bad[0].id,
            2 => bad[0].id = u64::MAX,
            3 => bad[0].fitness = f64::INFINITY,
            _ => bad[0].fitness = f64::NAN,
        }
        let before = search.checkpoint_json().unwrap();
        assert!(search.tell(&bad).is_err());
        assert_eq!(search.checkpoint_json().unwrap(), before);
    }
    search
        .tell(&good.iter().rev().copied().collect::<Vec<_>>())
        .unwrap();
    let before = search.checkpoint_json().unwrap();
    assert!(search.tell(&good).is_err());
    assert_eq!(search.checkpoint_json().unwrap(), before);
}

#[test]
fn tie_breaking_uses_lowest_id_and_elites_get_fresh_ids() {
    let mut search = GeneticAlgorithm::new(
        schema(PARITY),
        GAConfig {
            population_size: 4,
            elite_count: 1,
            tournament_size: 2,
            ..Default::default()
        },
        7,
    )
    .unwrap();
    let original = search.ask()[0].clone();
    let scores = search
        .ask()
        .iter()
        .map(|c| Score {
            id: c.id,
            fitness: if c.id % 2 == 0 { 0. } else { -0. },
        })
        .rev()
        .collect::<Vec<_>>();
    search.tell(&scores).unwrap();
    assert_eq!(search.best().unwrap().candidate.id, 0);
    assert_eq!(search.ask()[0].id, 4);
    assert_eq!(search.ask()[0].values, original.values);
}

#[test]
fn checkpoints_resume_rng_population_and_fitness_and_refuse_overwrite() {
    let mut search = GeneticAlgorithm::new(schema(SCALAR), config(), 22).unwrap();
    search.tell(&quadratic_scores(&search)).unwrap();
    let text = search.checkpoint_json().unwrap();
    let mut restored = GeneticAlgorithm::from_checkpoint_json(&text).unwrap();
    assert_eq!(restored.checkpoint_json().unwrap(), text);
    for _ in 0..3 {
        let scores = quadratic_scores(&search);
        search.tell(&scores).unwrap();
        restored.tell(&scores).unwrap();
        assert_eq!(
            search.checkpoint_json().unwrap(),
            restored.checkpoint_json().unwrap()
        );
    }
    let directory = support::scratch().unwrap();
    let path = directory.path().join("search.json");
    search.save_checkpoint(&path).unwrap();
    let stored = std::fs::read(&path).unwrap();
    assert!(search.save_checkpoint(&path).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), stored);
    assert_eq!(
        GeneticAlgorithm::load_checkpoint(&path)
            .unwrap()
            .checkpoint_json()
            .unwrap(),
        search.checkpoint_json().unwrap()
    );
}

#[test]
fn malformed_checkpoints_and_overflowing_mutations_are_rejected() {
    let search = GeneticAlgorithm::new(
        schema(PARITY),
        GAConfig {
            population_size: 4,
            elite_count: 1,
            tournament_size: 2,
            ..Default::default()
        },
        7,
    )
    .unwrap();
    let value: Value = serde_json::from_str(&search.checkpoint_json().unwrap()).unwrap();
    for kind in 0..7 {
        let mut bad = value.clone();
        match kind {
            0 => bad["format"] = json!("future"),
            1 => bad["population"][0]["id"] = json!(9),
            2 => bad["population"][0]["values"] = json!([1]),
            3 => bad["schema"]["parameters"][0]["shape"] = json!([9]),
            4 => bad["schema"]["parameters"][0]["updatable"] = json!(false),
            5 => bad["generation"] = json!(u64::MAX),
            _ => bad["config"]["elite_count"] = json!(0),
        }
        assert!(GeneticAlgorithm::from_checkpoint_json(&bad.to_string()).is_err());
    }
    let mut overflow = value;
    overflow["config"]["mutation_rate"] = json!(1.);
    overflow["config"]["mutation_scale"] = json!(1e100);
    let mut search = GeneticAlgorithm::from_checkpoint_json(&overflow.to_string()).unwrap();
    let before = search.checkpoint_json().unwrap();
    let scores = search
        .ask()
        .iter()
        .map(|c| Score {
            id: c.id,
            fitness: c.id as f64,
        })
        .collect::<Vec<_>>();
    assert!(search.tell(&scores).is_err());
    assert_eq!(search.checkpoint_json().unwrap(), before);
}

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn isolated_native_evaluation_resets_every_episode_and_commits_only_parameters() {
    let service = support::Service::start().unwrap();
    let client = service.client().unwrap();
    let definition = ModelDefinition::parse(CELL).unwrap();
    let target = client
        .initialize(&definition, InstanceOptions::default())
        .unwrap();
    target.call("step", &[vec![5.]]).unwrap();
    let state = target.state().unwrap();
    let initial = target.parameters().unwrap();
    let schema = GenomeSchema::from_model(&target).unwrap();
    let candidates = [
        Candidate {
            id: 11,
            values: vec![2.],
        },
        Candidate {
            id: 12,
            values: vec![3.],
        },
    ];
    let options = EvaluationOptions {
        operation: Some("step".into()),
        reset_operation: Some("reset".into()),
        episode_seeds: vec![11, 23],
        ..Default::default()
    };
    let mut seeds = Vec::new();
    let scores = evaluate_isolated(
        &client,
        &definition,
        &schema,
        &candidates,
        &options,
        |model, seed| {
            seeds.push(seed);
            assert_eq!(model.state()?[0].2, [0.]);
            model.forward(&[1.])?;
            Ok(model.forward(&[2.])?[0] as f64)
        },
    )
    .unwrap();
    assert_eq!(
        scores,
        [
            Score {
                id: 11,
                fitness: 6.
            },
            Score {
                id: 12,
                fitness: 9.
            }
        ]
    );
    assert_eq!(seeds, [11, 23, 11, 23]);
    assert_eq!(target.state().unwrap(), state);
    assert_eq!(target.parameters().unwrap(), initial);
    assert_eq!(client.stats().unwrap()["sessions"], 1);

    let mut search = GeneticAlgorithm::new(schema, config(), 7).unwrap();
    assert!(search.commit_best(&target).is_err());
    let scores = search
        .ask()
        .iter()
        .map(|c| Score {
            id: c.id,
            fitness: (c.values[0] - 2.).abs() as f64,
        })
        .collect::<Vec<_>>();
    search.tell(&scores).unwrap();
    search.commit_best(&target).unwrap();
    assert_eq!(
        target.parameters().unwrap()[0].2,
        search.best().unwrap().candidate.values
    );
    assert_eq!(target.state().unwrap(), state);
    target.close().unwrap();
    assert_eq!(client.stats().unwrap()["sessions"], 0);
    assert_eq!(client.stats().unwrap()["reserved"], 0);
}

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn isolated_failures_drop_workers_and_reject_parameter_writing_effects() {
    let service = support::Service::start().unwrap();
    let client = service.client().unwrap();
    let definition = ModelDefinition::parse(CELL).unwrap();
    let schema = schema(CELL);
    let candidates = [Candidate {
        id: 0,
        values: vec![2.],
    }];
    for (operation, reset) in [
        ("adapt", Some("reset")),
        ("step", Some("reset_bad")),
        ("fail", Some("reset")),
    ] {
        let options = EvaluationOptions {
            operation: Some(operation.into()),
            reset_operation: reset.map(Into::into),
            ..Default::default()
        };
        assert!(
            evaluate_isolated(
                &client,
                &definition,
                &schema,
                &candidates,
                &options,
                |model, _| Ok(model.forward(&[1.])?[0] as f64)
            )
            .is_err()
        );
        assert_eq!(client.stats().unwrap()["sessions"], 0);
        assert_eq!(client.stats().unwrap()["reserved"], 0);
    }
    let options = EvaluationOptions {
        operation: Some("step".into()),
        ..Default::default()
    };
    assert!(
        evaluate_isolated(
            &client,
            &definition,
            &schema,
            &candidates,
            &options,
            |model, _| {
                model.forward(&[1.])?;
                anyhow::bail!("failed episode")
            }
        )
        .is_err()
    );
    assert!(
        evaluate_isolated(
            &client,
            &definition,
            &schema,
            &candidates,
            &options,
            |_, _| Ok(f64::NAN)
        )
        .is_err()
    );
    assert_eq!(client.stats().unwrap()["sessions"], 0);

    let writing = ModelDefinition::parse("wgt 0.5\nmodel Write\ninput x:f32[1]\noutput y:f32[1]\nparameter gain:f32[1]=[1]\nfn forward(x){write(gain,x);return gain;}\n").unwrap();
    let writing_schema = schema_from_source(writing.source());
    assert!(
        evaluate_isolated(
            &client,
            &writing,
            &writing_schema,
            &candidates,
            &EvaluationOptions::default(),
            |m, _| Ok(m.forward(&[2.])?[0] as f64)
        )
        .is_err()
    );
    assert_eq!(client.stats().unwrap()["sessions"], 0);
}

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn fixed_pretrained_probability_baselines_require_matching_weights() {
    let service = support::Service::start().unwrap();
    let client = service.client().unwrap();
    let definition = ModelDefinition::parse("wgt 0.5\nmodel Probability\ninput x:f32[1]\noutput y:f32[1]\nparameter gain:f32[1]=[1]\nparameter probability:prob=0.25\nfn forward(x){return x*gain*probability;}\n").unwrap();
    let model = client
        .initialize(
            &definition,
            InstanceOptions {
                optimizer: Some(wgt::Optimizer::Sgd),
                ..Default::default()
            },
        )
        .unwrap();
    model
        .train(
            &[vec![1.]],
            wgt::Targets::Values(&[vec![1.]]),
            wgt::TrainOptions {
                rate: 0.1,
                loss: wgt::Loss::Mse,
                clip: None,
            },
        )
        .unwrap();
    let schema = GenomeSchema::from_model(&model).unwrap();
    let candidates = [Candidate {
        id: 0,
        values: schema.initial_values(),
    }];
    let expected = model.forward(&[1.]).unwrap()[0];
    assert_eq!(
        model.call("forward", &[vec![1.]]).unwrap(),
        Some(vec![expected])
    );
    assert!(model.call("forward", &[]).is_err());
    assert!(
        evaluate_isolated(
            &client,
            &definition,
            &schema,
            &candidates,
            &EvaluationOptions::default(),
            |m, _| Ok(m.forward(&[1.])?[0] as f64)
        )
        .is_err()
    );
    let weights = service.path().join("baseline.safetensors");
    model.save_weights(&weights).unwrap();
    let options = EvaluationOptions {
        initial_weights: Some(weights),
        ..Default::default()
    };
    let result = evaluate_isolated(
        &client,
        &definition,
        &schema,
        &candidates,
        &options,
        |m, _| Ok(m.forward(&[1.])?[0] as f64),
    )
    .unwrap();
    assert_eq!(result[0].fitness, expected as f64);
    assert_eq!(client.stats().unwrap()["sessions"], 1);
    model.close().unwrap();
}

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn forward_forward_helper_runs_local_explicit_transactions_without_ad() {
    let service = support::Service::start().unwrap();
    let client = service.client().unwrap();
    let definition = ModelDefinition::parse("wgt 0.5\nmodel LocalFF\nparameter weight:f32[1]=[0.5]\nfn forward(x:f32[1])->f32[1]{return relu(weight*x);}\nfn adapt(positive:f32[1],negative:f32[1],rate:f32[1]){let hp=relu(weight*positive);let hn=relu(weight*negative);let dp=([1]-sigmoid(square(hp)-[1]))*hp*positive;let dn=-sigmoid(square(hn)-[1])*hn*negative;weight <- weight+rate*(dp+dn);}\n").unwrap();
    let model = client
        .initialize(&definition, InstanceOptions::default())
        .unwrap();
    assert_eq!(
        fit_forward_forward(&model, [(vec![2.], vec![0.])], 0.1, "adapt").unwrap(),
        1
    );
    assert!((model.parameters().unwrap()[0].2[0] - 0.6).abs() < 1e-6);
    let before = model.parameters().unwrap();
    assert!(fit_forward_forward(&model, [(vec![2.], vec![0.])], f32::NAN, "adapt").is_err());
    assert_eq!(model.parameters().unwrap(), before);
    assert!(fit_forward_forward(&model, [(vec![2.], vec![0., 0.])], 0.1, "adapt").is_err());
    assert_eq!(model.parameters().unwrap(), before);
    model.close().unwrap();
}
