use std::process::Command;

#[test]
fn invalid_command_exits_nonzero() {
    let output = Command::new(env!("CARGO_BIN_EXE_blut-core"))
        .arg("definitely-not-a-command")
        .output()
        .expect("run blut-core");

    assert!(
        !output.status.success(),
        "invalid CLI input must fail; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
