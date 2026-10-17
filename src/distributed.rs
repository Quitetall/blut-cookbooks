//! Distributed launch — the single decision about whether a training run
//! goes under `torchrun`, and with what rendezvous.
//!
//! Every training backend in this repository shells out to Python, and each
//! one used to assemble its own `torchrun` prefix. The copies diverged: the
//! launcher-wrapped path passed `--standalone` together with `--nnodes N`,
//! which cannot mean anything (`--standalone` *is* a one-node rendezvous and
//! overrides `--nnodes`), and it never passed `--node_rank` at all. One
//! module now owns the decision so a fix lands everywhere at once.

use blut::config::launcher::{LaunchTarget, Launcher as _, SlurmLauncher};

/// Serde default for a process or node count: one.
pub fn one() -> u32 {
    1
}

/// Serde `skip_serializing_if` for a count left at its default of one.
///
/// Stage and recipe arguments are serialized into cache keys and plan
/// identities. Omitting a count at its default keeps every run that never set
/// one at the identity it had before the field existed, instead of missing the
/// cache on an argument that changes nothing.
pub fn is_one(n: &u32) -> bool {
    *n == 1
}

/// Rendezvous coordinates for a multi-node `torchrun` launch.
///
/// Held as data rather than read at the call site, so the launch decision is
/// testable without mutating process environment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rendezvous {
    master_addr: String,
    master_port: String,
    node_rank: String,
}

impl Rendezvous {
    /// Build a rendezvous directly. Test and caller-supplied coordinates.
    pub fn new(
        master_addr: impl Into<String>,
        master_port: impl Into<String>,
        node_rank: impl Into<String>,
    ) -> Self {
        Self {
            master_addr: master_addr.into(),
            master_port: master_port.into(),
            node_rank: node_rank.into(),
        }
    }

    /// Read the rendezvous the launcher exported.
    ///
    /// `MASTER_ADDR` and `NODE_RANK` are required and deliberately have no
    /// default. Defaulting them to `127.0.0.1` and `0` — which this did —
    /// makes every node its own coordinator and gives them all rank zero, so
    /// the world never forms: `torchrun` sits until its rendezvous timeout
    /// instead of naming the variable nobody set. `MASTER_PORT` keeps a
    /// default because torch's own is the same 29500.
    pub fn from_env() -> Result<Self, String> {
        let required = |key: &str| -> Result<String, String> {
            std::env::var(key).map_err(|_| {
                format!(
                    "multi-node training needs {key}; the Slurm launcher exports it, \
                     and a manual multi-node launch must set it on every node"
                )
            })
        };
        Ok(Self {
            master_addr: required("MASTER_ADDR")?,
            node_rank: required("NODE_RANK")?,
            master_port: std::env::var("MASTER_PORT").unwrap_or_else(|_| "29500".into()),
        })
    }
}

/// The `python` arguments that put a run under `torchrun`, or `None` when it
/// is a single process.
///
/// A run is distributed when `nproc_per_node * nnodes > 1`, not when
/// `nproc_per_node > 1`. Two hosts holding one GPU each — the cheapest real
/// multi-node shape there is — have `nproc_per_node == 1`, and gating on that
/// alone launched bare `python` on both: no rendezvous, no process group,
/// `WORLD_SIZE` unset, so the trainer took its single-process path. Each node
/// then trained its own full copy of the model and neither ever spoke to the
/// other, with nothing in the output to say so.
pub fn torchrun_args(nproc: u32, nnodes: u32, rdzv: Option<&Rendezvous>) -> Option<Vec<String>> {
    let (nproc, nnodes) = (nproc.max(1), nnodes.max(1));
    if nproc * nnodes <= 1 {
        return None;
    }
    let mut args = vec!["-m".to_string(), "torch.distributed.run".to_string()];
    match rdzv {
        // One node: `--standalone` picks a free loopback port on its own, and
        // is mutually exclusive with the explicit rendezvous flags below.
        None => args.extend([
            "--standalone".to_string(),
            "--nproc_per_node".to_string(),
            nproc.to_string(),
        ]),
        Some(r) => args.extend([
            "--nnodes".to_string(),
            nnodes.to_string(),
            "--nproc_per_node".to_string(),
            nproc.to_string(),
            "--rdzv_backend".to_string(),
            "c10d".to_string(),
            "--rdzv_endpoint".to_string(),
            format!("{}:{}", r.master_addr, r.master_port),
            "--node_rank".to_string(),
            r.node_rank.clone(),
        ]),
    }
    Some(args)
}

/// The launch prefix for `nproc` processes across `nnodes`, reading the
/// rendezvous from the environment when the job spans more than one node.
pub fn launch_prefix(nproc: u32, nnodes: u32) -> Result<Vec<String>, String> {
    let rdzv = if nnodes.max(1) > 1 {
        Some(Rendezvous::from_env()?)
    } else {
        None
    };
    Ok(torchrun_args(nproc, nnodes, rdzv.as_ref()).unwrap_or_default())
}

/// Shell run by every Slurm task: it resolves this node's rendezvous from the
/// allocation and `exec`s `torchrun`. Rank and coordinator are per-node facts,
/// so they cannot be computed by the process that calls `srun`: building
/// `--node_rank` there handed every node the same rank. Positional arguments:
/// `nnodes nproc python trainer-args...`.
const SLURM_TORCHRUN: &str = r#"set -eu
: "${SLURM_NODEID:?not running inside a Slurm step}"
: "${SLURM_JOB_NODELIST:?not running inside a Slurm allocation}"
nnodes=$1; nproc=$2; py=$3; shift 3
addr=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)}
exec "$py" -m torch.distributed.run --nnodes "$nnodes" --nproc_per_node "$nproc" \
  --rdzv_backend c10d --rdzv_endpoint "$addr:${MASTER_PORT:-29500}" \
  --node_rank "$SLURM_NODEID" "$@""#;

/// The program and arguments that run a training process where `target`
/// says, for `nproc` processes on each of `nnodes` nodes. `inner` is
/// everything after the interpreter (`-m blut_core.trainer --config …`).
///
/// A target this function cannot honor is an error, never a local run: a
/// `--launcher slurm` request that quietly trained on the submitting host —
/// which every training stage did — looks like success and is not one.
pub fn training_argv(
    target: LaunchTarget,
    python: &str,
    nproc: u32,
    nnodes: u32,
    gpus: bool,
    inner: &[String],
) -> Result<(String, Vec<String>), String> {
    let (nproc, nnodes) = (nproc.max(1), nnodes.max(1));
    match target {
        LaunchTarget::Local => {
            let mut argv = launch_prefix(nproc, nnodes)?;
            argv.extend(inner.iter().cloned());
            Ok((python.to_string(), argv))
        }
        LaunchTarget::Slurm => {
            let mut task: Vec<String> = Vec::new();
            if nproc * nnodes > 1 {
                task.extend(["bash", "-c", SLURM_TORCHRUN, "blut-train"].map(String::from));
                task.extend([nnodes.to_string(), nproc.to_string()]);
            }
            task.push(python.to_string());
            task.extend(inner.iter().cloned());
            let w = slurm_launcher(nproc, nnodes, gpus)?
                .wrap("blut-train", &task)
                .map_err(|e| format!("slurm launcher: {e}"))?;
            Ok((w.program, w.args))
        }
        other => Err(format!(
            "training cannot run under --launcher {other:?}: this stage supports \
             `local` and `slurm`"
        )),
    }
}

/// `srun` placement for a training step: one task per node, because
/// `torchrun` starts the `nproc` ranks itself. `BLUT_SLURM_*` supplies the
/// site settings; a node or task count there that contradicts the stage's
/// own is refused rather than letting one silently win.
fn slurm_launcher(nproc: u32, nnodes: u32, gpus: bool) -> Result<SlurmLauncher, String> {
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    let env_u32 = |k: &str| -> Result<Option<u32>, String> {
        env(k)
            .map(|v| {
                v.parse::<u32>()
                    .map_err(|_| format!("{k}={v:?} is not a count"))
            })
            .transpose()
    };
    if let Some(n) = env_u32("BLUT_SLURM_NODES")?.filter(|&n| n != nnodes) {
        return Err(format!(
            "BLUT_SLURM_NODES={n} contradicts the stage's nnodes={nnodes}"
        ));
    }
    if let Some(n) = env_u32("BLUT_SLURM_NTASKS_PER_NODE")?.filter(|&n| n != 1) {
        return Err(format!(
            "BLUT_SLURM_NTASKS_PER_NODE={n}: a training step runs one task per node \
             (torchrun starts the {nproc} ranks); set nproc_per_node instead"
        ));
    }
    // `--gpus` is a whole-job total in Slurm; an explicit site value is kept
    // verbatim, otherwise each node asks for the GPUs its ranks will use.
    let site_gpus = env_u32("BLUT_SLURM_GPUS")?;
    let mut extra = Vec::new();
    if gpus && site_gpus.is_none() {
        extra.push(format!("--gpus-per-node={nproc}"));
    }
    Ok(SlurmLauncher {
        partition: env("BLUT_SLURM_PARTITION"),
        mem: env("BLUT_SLURM_MEM"),
        cpus: env_u32("BLUT_SLURM_CPUS")?,
        gpus: site_gpus,
        time: env("BLUT_SLURM_TIME"),
        nodes: Some(nnodes),
        ntasks_per_node: Some(1),
        extra,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The defect this module exists for. Two hosts, one GPU each, is a real
    /// distributed run; gating on `nproc > 1` launched bare Python on both.
    #[test]
    fn one_process_on_each_of_two_nodes_still_goes_under_torchrun() {
        let rdzv = Rendezvous::new("10.0.0.1", "29500", "1");
        let args = torchrun_args(1, 2, Some(&rdzv)).expect("2 nodes is a distributed run");
        assert_eq!(&args[..2], &["-m", "torch.distributed.run"]);
        assert!(args.windows(2).any(|w| w == ["--nnodes", "2"]));
        assert!(args.windows(2).any(|w| w == ["--node_rank", "1"]));
        assert!(
            args.windows(2)
                .any(|w| w == ["--rdzv_endpoint", "10.0.0.1:29500"])
        );
    }

    /// `--standalone` is a one-node rendezvous and overrides `--nnodes`, so a
    /// multi-node launch carrying both ran as N disconnected single-node jobs.
    #[test]
    fn multi_node_never_passes_standalone() {
        let rdzv = Rendezvous::new("10.0.0.1", "29500", "0");
        for (nproc, nnodes) in [(1, 2), (2, 2), (4, 8)] {
            let args = torchrun_args(nproc, nnodes, Some(&rdzv)).unwrap();
            assert!(
                !args.iter().any(|a| a == "--standalone"),
                "--standalone leaked into a {nnodes}-node launch: {args:?}"
            );
        }
    }

    #[test]
    fn single_node_multi_gpu_uses_standalone_and_no_endpoint() {
        let args = torchrun_args(4, 1, None).unwrap();
        assert!(args.iter().any(|a| a == "--standalone"));
        assert!(args.windows(2).any(|w| w == ["--nproc_per_node", "4"]));
        assert!(!args.iter().any(|a| a == "--rdzv_endpoint"));
    }

    #[test]
    fn a_single_process_is_not_launched_under_torchrun() {
        assert_eq!(torchrun_args(1, 1, None), None);
        // Zero is not a smaller-than-one world; it clamps to a single process.
        assert_eq!(torchrun_args(0, 0, None), None);
    }

    /// Clamping must not turn a distributed request into a silent local run.
    #[test]
    fn zero_nproc_across_several_nodes_is_still_distributed() {
        let rdzv = Rendezvous::new("10.0.0.1", "29500", "0");
        let args = torchrun_args(0, 2, Some(&rdzv)).expect("still two nodes");
        assert!(args.windows(2).any(|w| w == ["--nproc_per_node", "1"]));
    }

    fn inner() -> Vec<String> {
        ["-m", "blut_core.trainer", "--config", "/shared/c.json"]
            .map(String::from)
            .to_vec()
    }

    fn slurm(nproc: u32, nnodes: u32) -> (String, Vec<String>) {
        let _g = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        for k in [
            "BLUT_SLURM_NODES",
            "BLUT_SLURM_NTASKS_PER_NODE",
            "BLUT_SLURM_GPUS",
        ] {
            // SAFETY: serialized by TEST_ENV_LOCK; no other thread reads env here.
            unsafe { std::env::remove_var(k) };
        }
        training_argv(
            LaunchTarget::Slurm,
            "python3",
            nproc,
            nnodes,
            true,
            &inner(),
        )
        .unwrap()
    }

    /// The defect: `--launcher slurm` never reached a training stage, which
    /// ran on the submitting host. A Slurm target must produce an `srun`.
    #[test]
    fn a_slurm_target_runs_the_trainer_through_srun() {
        let (prog, args) = slurm(1, 1);
        assert_eq!(prog, "srun");
        let sep = args.iter().position(|a| a == "--").unwrap();
        assert_eq!(
            &args[sep + 1..],
            [
                "python3",
                "-m",
                "blut_core.trainer",
                "--config",
                "/shared/c.json"
            ]
        );
        assert!(args.iter().any(|a| a == "--nodes=1"));
        assert!(args.iter().any(|a| a == "--ntasks-per-node=1"));
        assert!(args.iter().any(|a| a == "--gpus-per-node=1"));
    }

    /// Rank is decided on each node from `SLURM_NODEID`, never baked into the
    /// command line by the submitter (which gave every node the same rank).
    #[test]
    fn multi_node_slurm_resolves_rank_on_each_node() {
        let (prog, args) = slurm(2, 2);
        assert_eq!(prog, "srun");
        assert!(args.iter().any(|a| a == "--nodes=2"));
        assert!(args.iter().any(|a| a == "--gpus-per-node=2"));
        assert!(
            !args.iter().any(|a| a == "--node_rank"),
            "rank baked in: {args:?}"
        );
        let sep = args.iter().position(|a| a == "--").unwrap();
        let task = &args[sep + 1..];
        assert_eq!(&task[..2], ["bash", "-c"]);
        assert!(task[2].contains(r#"--node_rank "$SLURM_NODEID""#));
        assert_eq!(&task[3..7], ["blut-train", "2", "2", "python3"]);
        assert_eq!(&task[7..], inner().as_slice());
    }

    /// The task script must parse as shell and fail loudly outside a step.
    #[test]
    fn the_slurm_task_script_refuses_to_run_outside_an_allocation() {
        let out = std::process::Command::new("bash")
            .args(["-c", SLURM_TORCHRUN, "blut-train", "2", "1", "python3"])
            .env_remove("SLURM_NODEID")
            .output()
            .expect("bash");
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stderr).contains("not running inside a Slurm step"));
    }

    #[test]
    fn a_contradicting_site_node_count_is_refused() {
        let _g = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        // SAFETY: serialized by TEST_ENV_LOCK.
        unsafe { std::env::set_var("BLUT_SLURM_NODES", "4") };
        let r = training_argv(LaunchTarget::Slurm, "python3", 1, 2, true, &inner());
        unsafe { std::env::remove_var("BLUT_SLURM_NODES") };
        assert!(r.unwrap_err().contains("contradicts"));
    }

    #[test]
    fn a_target_the_stage_cannot_honor_is_an_error_not_a_local_run() {
        for t in [LaunchTarget::Ray, LaunchTarget::P2P] {
            assert!(
                training_argv(t, "python3", 1, 1, true, &inner()).is_err(),
                "{t:?}"
            );
        }
    }

    #[test]
    fn local_single_process_is_bare_python() {
        let (prog, args) =
            training_argv(LaunchTarget::Local, "python3", 1, 1, true, &inner()).unwrap();
        assert_eq!(prog, "python3");
        assert_eq!(args, inner());
    }

    #[test]
    fn rendezvous_formats_the_endpoint_as_host_colon_port() {
        let args =
            torchrun_args(1, 2, Some(&Rendezvous::new("node0.local", "12345", "0"))).unwrap();
        let i = args.iter().position(|a| a == "--rdzv_endpoint").unwrap();
        assert_eq!(args[i + 1], "node0.local:12345");
    }
}
