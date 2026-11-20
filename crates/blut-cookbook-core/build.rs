// SPDX-License-Identifier: Apache-2.0
//! Stamps this crate's build identity for `blut::cookbook_identity!()`.
//!
//! The repository root's `build.rs` carries the same function: each crate is
//! packaged on its own, so neither can include the other's file.

fn main() {
    let dir = std::path::PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into()),
    );
    // This crate, the `blut-cookbook-standard` crate it depends on by path,
    // and the Python trainer both run: the whole repository's code.
    emit_cookbook_identity(
        &dir,
        &[
            "../../src",
            "../../crates",
            "../../python",
            "../../Cargo.toml",
            "../../Cargo.lock",
            "../../build.rs",
        ],
    );
}

/// Emit `BLUT_COOKBOOK_COMMIT` and `BLUT_COOKBOOK_DIRTY` for
/// `blut::cookbook_identity!()`: the last commit that touched `scope` (paths
/// relative to `base`, the crate directory), and a git object id over the uncommitted changes in it
/// (the binary diff of tracked files plus the content of untracked,
/// non-ignored files), or nothing when it is clean. Every stage cache key
/// folds both, so a rebuild with other code never reuses a cached result. A
/// build without git (crates.io) emits empty values and the identity falls
/// back to the crate version.
fn emit_cookbook_identity(base: &std::path::Path, scope: &[&str]) {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let git = |args: &[&str], input: Option<&[u8]>| -> Option<Vec<u8>> {
        let mut cmd = Command::new("git");
        cmd.arg("-C")
            .arg(base)
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        cmd.stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        });
        let mut child = cmd.spawn().ok()?;
        if let Some(bytes) = input {
            child.stdin.take()?.write_all(bytes).ok()?;
        }
        let out = child.wait_with_output().ok()?;
        out.status.success().then_some(out.stdout)
    };
    let scoped = |head: &[&str]| -> Option<Vec<u8>> {
        let mut args = head.to_vec();
        args.push("--");
        args.extend_from_slice(scope);
        git(&args, None)
    };
    let text = |b: Vec<u8>| String::from_utf8(b).ok().map(|s| s.trim().to_string());
    println!("cargo:rerun-if-changed=build.rs");
    // Only a crate whose own manifest is tracked in the repository around it:
    // a crates.io or vendored copy reports no git identity rather than the
    // identity of whatever repository happens to contain it.
    if git(&["ls-files", "--error-unmatch", "Cargo.toml"], None).is_none() {
        println!("cargo:rustc-env=BLUT_COOKBOOK_COMMIT=");
        println!("cargo:rustc-env=BLUT_COOKBOOK_DIRTY=");
        return;
    }

    let commit = scoped(&["log", "-1", "--format=%H"])
        .and_then(text)
        .unwrap_or_default();
    let dirty = (|| {
        let diff = scoped(&["diff", "HEAD", "--binary"])?;
        let untracked = scoped(&["ls-files", "--others", "--exclude-standard", "-z"])?;
        if diff.is_empty() && untracked.is_empty() {
            return None;
        }
        let mut manifest = diff;
        manifest.extend_from_slice(b"\0untracked\0");
        for path in untracked.split(|b| *b == 0).filter(|p| !p.is_empty()) {
            let path = String::from_utf8_lossy(path).into_owned();
            manifest.extend_from_slice(path.as_bytes());
            manifest.push(0);
            manifest.extend_from_slice(&git(&["hash-object", "--", path.as_str()], None)?);
        }
        git(&["hash-object", "--stdin"], Some(&manifest)).and_then(text)
    })()
    .unwrap_or_default();
    println!("cargo:rustc-env=BLUT_COOKBOOK_COMMIT={commit}");
    println!("cargo:rustc-env=BLUT_COOKBOOK_DIRTY={dirty}");

    // Re-run when the covered source changes (for the dirty digest) and when
    // HEAD moves (for the commit). Only existing paths: a missing one would
    // re-run this script on every build.
    let watch = |p: std::path::PathBuf| {
        if p.exists() {
            println!("cargo:rerun-if-changed={}", p.display());
        }
    };
    for rel in scope {
        watch(base.join(rel));
    }
    for rel in ["HEAD", "logs/HEAD", "packed-refs"] {
        if let Some(p) = git(&["rev-parse", "--git-path", rel], None).and_then(text) {
            let p = std::path::PathBuf::from(p);
            watch(if p.is_absolute() { p } else { base.join(p) });
        }
    }
    if let Some(r) = git(&["symbolic-ref", "--quiet", "HEAD"], None).and_then(text)
        && let Some(p) = git(&["rev-parse", "--git-path", r.as_str()], None).and_then(text)
    {
        let p = std::path::PathBuf::from(p);
        watch(if p.is_absolute() { p } else { base.join(p) });
    }
}
