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
            Ok(update) => {
                // No subscriber (a stage run outside an executor) is not an error.
                let _ = self.tx.send(StageEvent::StageStep {
                    node_idx: self.node_idx,
                    stage_name: self.stage_name.to_string(),
                    update,
                });
            }
            Err(e) => tracing::warn!(
                "{}: cannot serialize a trainer update: {e}",
                self.stage_name
            ),
        }
    }

    /// One line of the trainer's stdout. A [`STATUS_PREFIX`] line is forwarded
    /// and returned; any other line, or a marked line that is not a valid
    /// update, is printed to this process's stdout unchanged.
    pub fn relay_line(&self, line: &str) -> Option<StatusUpdate> {
        if let Some(json) = line.strip_prefix(STATUS_PREFIX) {
            match serde_json::from_str::<StatusUpdate>(json) {
                Ok(update) => {
                    self.send(&update);
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

/// The `final_loss` an `HfCheckpoint` records, from the loss the trainer
/// reported (`None` when it reported none: a run that trained no step).
///
/// `blut::artifacts::HfCheckpoint::final_loss` is an `f32`, so a missing loss
/// cannot be recorded as missing. It is recorded as 0.0, as before, with a
/// warning saying so. NaN is no way out: the engine projects every stage
/// output through JSON to rebase its paths, and NaN does not survive that
/// (JSON has none), so the projection would fail. Recording the absence needs
/// the engine field to become `Option<f32>`.
pub fn checkpoint_loss(stage_name: &str, reported: Option<f32>) -> f32 {
    reported.unwrap_or_else(|| {
        tracing::warn!(
            "{stage_name}: the trainer reported no final loss; the checkpoint records \
             final_loss 0.0 only because HfCheckpoint.final_loss must be a number"
        );
        0.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reported_loss_is_the_checkpoints() {
        assert_eq!(checkpoint_loss("s", Some(1.25)), 1.25);
        assert_eq!(checkpoint_loss("s", None), 0.0);
    }

    /// Why a missing loss is not recorded as NaN: the engine rebases a stage
    /// output's paths through a JSON round trip, which NaN fails.
    #[test]
    fn a_nan_final_loss_would_not_survive_the_engines_output_projection() {
        let ckpt = |final_loss| blut::artifacts::HfCheckpoint {
            path: "/c".into(),
            base_model: "m".into(),
            method_tag: "full".into(),
            content_hash: blut::framework::artifact::ContentHash::of_bytes(b""),
            final_loss,
        };
        let round_trip = |c: blut::artifacts::HfCheckpoint| {
            serde_json::from_value::<blut::artifacts::HfCheckpoint>(
                serde_json::to_value(c).unwrap(),
            )
        };
        assert!(round_trip(ckpt(0.0)).is_ok());
        assert!(round_trip(ckpt(f32::NAN)).is_err());
    }

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
