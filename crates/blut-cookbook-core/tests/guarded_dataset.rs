//! A real data stage behind the engine's built-in checks.
//!
//! `load_dataset` emits `dataset.jsonl`; `check_jsonl` and `assert` take only
//! `checks.jsonl`. These plans put the guards on the data path with the
//! standard cookbook's `to_checks_jsonl` / `from_checks_jsonl` and run them
//! through the engine's executor: a clean dataset reaches `split_train_eval`,
//! and a planted row fails the guard before the split ever starts.

use std::path::{Path, PathBuf};

use blut::artifacts::DatasetSplit;
use blut::checks::{Assert, AssertArgs, CheckJsonl, CheckJsonlArgs, CmpOp, OnBreach};
use blut::framework::error::PlanError;
use blut::framework::executor::{ExecCtx, PlanResult, SequentialExecutor};
use blut::framework::plan::{CompiledPlan, Plan};
use blut::framework::plan_spec::PlanSpec;
use blut::framework::status::StageEvent;
use blut_backends::LamuTrainerBackend;
use blut_backends::stages::split_train_eval::{Args as SplitArgs, SplitTrainEval};
use blut_backends::stages::{FromChecksJsonl, ToChecksJsonl};
use blut_cookbook_core::stages::load_dataset::{Args as LoadArgs, LoadDataset};

/// The declared recipe this repository ships for a guarded dataset.
const SHIPPED_RECIPE: &str = include_str!("../examples/guarded_dataset.json");

const CLEAN: &str = "{\"text\":\"one\"}\n{\"text\":\"two\"}\n{\"text\":\"three\"}\n";

/// Every registered stage a declared recipe can name: both cookbooks, and the
/// engine's `checks` cookbook, which the CLI adds to every registry.
fn registry() -> blut::framework::Registry {
    let mut reg = blut_cookbook_core::registry();
    blut::checks::register(&mut reg);
    reg
}

fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, body).unwrap();
    p
}

/// `load_dataset → to_checks_jsonl → check_jsonl → assert → from_checks_jsonl
/// → split_train_eval`, guarding for exactly three rows.
fn guarded_spec(dataset: &Path) -> PlanSpec {
    serde_json::from_value(serde_json::json!({
        "name": "guarded",
        "nodes": [
            {"stage": "load_dataset", "args": {"path": dataset}},
            {"stage": "to_checks_jsonl"},
            {"stage": "check_jsonl", "args": {"min_rows": 3, "max_rows": 3}},
            {"stage": "assert", "args": {"actual": 3.0, "op": "eq", "expected": 3.0}},
            {"stage": "from_checks_jsonl"},
            {"stage": "split_train_eval", "args": {"eval_ratio": 0.34, "seed": 7}}
        ],
        "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
    }))
    .expect("a well-formed plan spec")
}

struct Run {
    result: Result<PlanResult, PlanError>,
    events: Vec<StageEvent>,
    _job_dir: tempfile::TempDir,
}

async fn execute(plan: CompiledPlan) -> Run {
    let job_dir = tempfile::tempdir().unwrap();
    let ctx = ExecCtx::new(job_dir.path().to_path_buf());
    let mut rx = ctx.status.subscribe();
    let result = SequentialExecutor::execute(plan, ctx).await;
    let mut events = Vec::new();
    loop {
        match rx.try_recv() {
            Ok(ev) => events.push(ev),
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(_) => break,
        }
    }
    Run {
        result,
        events,
        _job_dir: job_dir,
    }
}

fn began(events: &[StageEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|ev| match ev {
            StageEvent::StageBegin { stage_name, .. } => Some(stage_name.clone()),
            _ => None,
        })
        .collect()
}

fn split_of(run: Run) -> DatasetSplit {
    let out = run
        .result
        .unwrap_or_else(|e| panic!("the guarded plan must pass: {e}"))
        .final_output
        .expect("the split is the plan's output");
    out.into_typed::<DatasetSplit>().expect("a dataset split")
}

/// A clean dataset passes both guards and reaches the split whole.
#[tokio::test]
async fn a_clean_dataset_passes_the_guards_and_reaches_its_consumer() {
    let td = tempfile::tempdir().unwrap();
    let data = write(td.path(), "clean.jsonl", CLEAN);
    let plan = guarded_spec(&data)
        .compile(&registry())
        .expect("kinds line up");
    let run = execute(plan).await;
    assert_eq!(
        began(&run.events),
        [
            "load_dataset",
            "to_checks_jsonl",
            "check_jsonl",
            "assert",
            "from_checks_jsonl",
            "split_train_eval"
        ]
    );
    let split = split_of(run);
    assert_eq!(split.train.n_examples + split.eval.n_examples, 3);
}

/// A planted fourth row breaches `check_jsonl`'s bound: the node fails with
/// the engine's data-quality code, the plan fails, and nothing downstream of
/// the guard starts, the consumer included.
#[tokio::test]
async fn a_planted_row_fails_the_guard_and_the_consumer_never_runs() {
    let td = tempfile::tempdir().unwrap();
    let data = write(
        td.path(),
        "planted.jsonl",
        &format!("{CLEAN}{{\"text\":\"planted\"}}\n"),
    );
    let plan = guarded_spec(&data)
        .compile(&registry())
        .expect("kinds line up");
    let run = execute(plan).await;
    assert!(run.result.is_err(), "a blocking breach must fail the plan");
    assert_eq!(
        began(&run.events),
        ["load_dataset", "to_checks_jsonl", "check_jsonl"],
        "nothing after the guard may start"
    );
    let breaches = blut::checks::scan_breaches(run.events);
    assert_eq!(breaches.len(), 1, "{breaches:?}");
    assert_eq!(breaches[0].stage, "check_jsonl");
    assert_eq!(breaches[0].code, "DATA_QUALITY_MAX_ROWS");
    assert_eq!(breaches[0].disposition, "block");
}

/// The same guard in a compiled recipe: the typed plan only builds because
/// the adapters and the engine's checks are `Compatible` with the backend.
#[tokio::test]
async fn a_compiled_recipe_can_guard_its_data_too() {
    let td = tempfile::tempdir().unwrap();
    let data = write(td.path(), "clean.jsonl", CLEAN);
    let plan = Plan::<(), LamuTrainerBackend>::new("guarded", serde_json::json!({}))
        .start(
            LoadDataset,
            LoadArgs {
                path: Some(data),
                split: "train".into(),
                ..LoadArgs::default()
            },
        )
        .then(ToChecksJsonl, ())
        .then(
            CheckJsonl,
            CheckJsonlArgs {
                min_rows: Some(3),
                max_rows: Some(3),
                on_breach: OnBreach::Block,
            },
        )
        .then(
            Assert,
            AssertArgs {
                actual: 3.0,
                op: CmpOp::Eq,
                expected: 3.0,
                tolerance: 0.0,
                on_breach: OnBreach::Block,
            },
        )
        .then(FromChecksJsonl, ())
        .then(
            SplitTrainEval,
            SplitArgs {
                eval_ratio: 0.34,
                seed: 7,
            },
        )
        .finish()
        .into_compiled();
    let split = split_of(execute(plan).await);
    assert_eq!(split.train.n_examples + split.eval.n_examples, 3);
}

/// The recipe in `examples/` names only registered stages, kind-checks, and
/// runs once its dataset path and row bounds are set.
#[tokio::test]
async fn the_shipped_recipe_compiles_and_runs() {
    let mut spec: serde_json::Value =
        serde_json::from_str(SHIPPED_RECIPE).expect("the shipped recipe is JSON");
    let reg = registry();
    let shipped: PlanSpec = serde_json::from_value(spec.clone()).expect("a plan spec");
    shipped
        .compile(&reg)
        .expect("the shipped recipe kind-checks");

    let td = tempfile::tempdir().unwrap();
    let data = write(td.path(), "clean.jsonl", CLEAN);
    spec["nodes"][0]["args"]["path"] = serde_json::json!(data);
    spec["nodes"][2]["args"] = serde_json::json!({"min_rows": 3, "max_rows": 3});
    let spec: PlanSpec = serde_json::from_value(spec).unwrap();
    let split = split_of(execute(spec.compile(&reg).unwrap()).await);
    assert_eq!(split.train.n_examples + split.eval.n_examples, 3);
}
