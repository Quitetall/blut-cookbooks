use std::process::Command;

#[test]
fn recipe_error_produces_nonzero_exit_status() {
    let output = Command::new(env!("CARGO_BIN_EXE_blut-core"))
        .args(["recipe", "run", "definitely_missing", "--dry-run"])
        .output()
        .expect("run blut-core");

    assert!(
        !output.status.success(),
        "recipe errors must propagate through the process exit status"
    );
}
