//! Distributed launch — the single decision about whether a training run
//! goes under `torchrun`, and with what rendezvous.
//!
//! Every training backend in this repository shells out to Python, and each
//! one used to assemble its own `torchrun` prefix. The copies diverged: the
//! launcher-wrapped path passed `--standalone` together with `--nnodes N`,
//! which cannot mean anything (`--standalone` *is* a one-node rendezvous and
//! overrides `--nnodes`), and it never passed `--node_rank` at all. One
//! module now owns the decision so a fix lands everywhere at once.

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

    #[test]
    fn rendezvous_formats_the_endpoint_as_host_colon_port() {
        let args =
            torchrun_args(1, 2, Some(&Rendezvous::new("node0.local", "12345", "0"))).unwrap();
        let i = args.iter().position(|a| a == "--rdzv_endpoint").unwrap();
        assert_eq!(args[i + 1], "node0.local:12345");
    }
}
