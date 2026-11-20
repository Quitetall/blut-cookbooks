//! Trainer progress, relayed to the engine.
//!
//! `blut log` and `blut jobs` read a plan's training progress from the updates
//! a stage forwards as `StageEvent::StageStep`, each `update` one
//! `blut::protocol::StatusUpdate` (the engine's `jobs::read_status`). A stage
//! that forwards anything else, or nothing, is invisible to both.
//!
//! [`Forwarder`] is that one send, for a stage node. A trainer whose stdout is
//! also its terminal output (`blut_core.trainer`, which `train_model` runs)
//! marks its updates with [`STATUS_PREFIX`]; [`Forwarder::relay_line`] forwards
//! those and prints every other line, so the terminal reads as it did.

use std::io::Write;

use blut::framework::stage::StageContext;
use blut::framework::status::StageEvent;
use blut::protocol::StatusUpdate;

/// Marks a trainer stdout line that carries one `StatusUpdate` as JSON:
/// `BLUT_STATUS {"kind":"step",...}`. The Python side is
/// `blut_core.status.RELAY_PREFIX`.
pub const STATUS_PREFIX: &str = "BLUT_STATUS ";

/// Forwards trainer updates as one stage node's progress.
#[derive(Clone)]
pub struct Forwarder {
    tx: tokio::sync::broadcast::Sender<StageEvent>,
    node_idx: u32,
    stage_name: &'static str,
}

impl Forwarder {
    /// For the stage running in `ctx`. `stage_name` is the stage the updates
    /// are attributed to, with `ctx.node_idx` (what `blut log` prefixes and
    /// the HPO scheduler maps to a trial).
    pub fn new(ctx: &StageContext, stage_name: &'static str) -> Self {
        Self {
            tx: ctx.status_tx.clone(),
            node_idx: ctx.node_idx,
            stage_name,
        }
    }

    /// Send `update` to the executor.
    pub fn send(&self, update: &StatusUpdate) {
        match serde_json::to_value(update) {
            Ok(update) => self.send_value(update),
            Err(e) => self.warn_unserializable(&e),
        }
    }

    /// [`send`](Self::send), with a non-finite measurement beside the update
    /// as `metric: "nan"` (or `"inf"`, `"-inf"`). JSON has no such number and
    /// `StatusUpdate` no such field, and this string is how the engine takes
    /// one: `StepMetrics::has_non_finite` and the HPO objective both read it,
    /// so a diverged trial is killed, while `read_status` ignores the extra
    /// key and still reads the update.
    pub fn send_non_finite(&self, update: &StatusUpdate, metric: &str, value: &str) {
        match serde_json::to_value(update) {
            Ok(mut update) => {
                update[metric] = value.into();
                self.send_value(update);
            }
            Err(e) => self.warn_unserializable(&e),
        }
    }

    fn send_value(&self, update: serde_json::Value) {
        // No subscriber (a stage run outside an executor) is not an error.
        let _ = self.tx.send(StageEvent::StageStep {
            node_idx: self.node_idx,
            stage_name: self.stage_name.to_string(),
            update,
        });
    }

    fn warn_unserializable(&self, error: &serde_json::Error) {
        tracing::warn!(
            "{}: cannot serialize a trainer update: {error}",
            self.stage_name
        );
    }

    /// One line of the trainer's stdout. A [`STATUS_PREFIX`] line is forwarded
    /// as written, keys `StatusUpdate` does not have included (a divergence
    /// token, see [`send_non_finite`](Self::send_non_finite)), and returned;
    /// any other line, or a marked line that is not a valid update, is printed
    /// to this process's stdout unchanged.
    pub fn relay_line(&self, line: &str) -> Option<StatusUpdate> {
        if let Some(json) = line.strip_prefix(STATUS_PREFIX) {
            match serde_json::from_str::<serde_json::Value>(json).and_then(|raw| {
                serde_json::from_value::<StatusUpdate>(raw.clone()).map(|typed| (raw, typed))
            }) {
                Ok((raw, update)) => {
                    self.send_value(raw);
                    return Some(update);
                }
                Err(e) => tracing::warn!(
                    "{}: unparseable trainer status ({e}): {}",
                    self.stage_name,
                    json.chars().take(200).collect::<String>()
                ),
            }
        }
        // A closed stdout must not fail the stage, so no `println!`.
        let _ = writeln!(std::io::stdout().lock(), "{line}");
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn forwarder() -> (Forwarder, tokio::sync::broadcast::Receiver<StageEvent>) {
        let mut ctx = StageContext::for_test("/tmp/j".into(), "/tmp/j/s".into());
        ctx.node_idx = 5;
        let rx = ctx.status_tx.subscribe();
        (Forwarder::new(&ctx, "train_model"), rx)
    }

    #[test]
    fn a_marked_line_is_forwarded_as_the_stages_update() {
        let (fwd, mut rx) = forwarder();
        let got = fwd.relay_line(
            r#"BLUT_STATUS {"kind":"step","step":3,"total":8,"loss":0.5,"lr":0.001,"vram_mb":0}"#,
        );
        let step = StatusUpdate::Step {
            step: 3,
            total: 8,
            loss: 0.5,
            lr: 0.001,
            vram_mb: 0,
        };
        assert_eq!(got, Some(step.clone()));
        match rx.try_recv().unwrap() {
            StageEvent::StageStep {
                node_idx,
                stage_name,
                update,
            } => {
                assert_eq!((node_idx, stage_name.as_str()), (5, "train_model"));
                assert_eq!(
                    serde_json::from_value::<StatusUpdate>(update).unwrap(),
                    step
                );
            }
            other => panic!("expected a StageStep, got {other:?}"),
        }
    }

    /// Terminal output and the metric lines other tools read pass through;
    /// they are not progress.
    #[test]
    fn other_lines_are_not_forwarded() {
        let (fwd, mut rx) = forwarder();
        for line in [
            "[trainer] step 3/8 loss=0.5000",
            r#"BLUT_METRIC {"step": 3, "loss": 0.5, "kind": "step"}"#,
            r#"{"kind":"step","step":3,"total":8,"loss":0.5,"lr":0.001,"vram_mb":0}"#,
            r#"BLUT_STATUS {"kind":"step","step":3}"#,
            "BLUT_STATUS not json",
        ] {
            assert_eq!(fwd.relay_line(line), None, "{line}");
        }
        assert!(rx.try_recv().is_err(), "nothing forwarded");
    }

    /// A diverged trainer's update keeps its token through the relay, so the
    /// engine's kill path sees it, and stays a heartbeat for `read_status`.
    #[test]
    fn a_divergence_token_survives_the_relay_and_trips_the_engines_checks() {
        use blut::framework::control::{Control, ControlPolicy, KillOnNaN, StepMetrics};
        let (fwd, mut rx) = forwarder();
        let got = fwd.relay_line(
            r#"BLUT_STATUS {"kind":"heartbeat","phase":"step 2/8 loss=nan","vram_mb":0,"loss":"nan"}"#,
        );
        assert!(matches!(got, Some(StatusUpdate::Heartbeat { .. })));
        let StageEvent::StageStep { update, .. } = rx.try_recv().unwrap() else {
            panic!("expected a StageStep");
        };
        let metrics = StepMetrics {
            node_idx: 5,
            stage_name: "train_model",
            update: &update,
        };
        assert!(metrics.has_non_finite());
        assert_eq!(KillOnNaN.on_step(&metrics), Control::KillBranch);
        let objective = blut::hpo::scheduler::dotted_f64(&update, "loss").unwrap();
        assert!(!objective.is_finite(), "the HPO scheduler's kill condition");
        assert!(matches!(
            serde_json::from_value::<StatusUpdate>(update).unwrap(),
            StatusUpdate::Heartbeat { .. }
        ));
    }

    /// The prefix is a contract with the Python runtime the trainer uses.
    #[test]
    fn the_prefix_matches_the_python_runtime() {
        let status_py = include_str!("../python/blut_core/status.py");
        assert!(
            status_py.contains(&format!("RELAY_PREFIX = \"{STATUS_PREFIX}\"")),
            "blut_core.status.RELAY_PREFIX must be {STATUS_PREFIX:?}"
        );
    }
}
