use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

const POLICY_PATH: &str = "src/backends/hf_trainer/requirements-hf.in";

fn main() {
    println!("cargo:rerun-if-changed={POLICY_PATH}");

    let policy = fs::read_to_string(POLICY_PATH).expect("read HF requirements policy");
    let requirements: Vec<_> = policy
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    assert!(!requirements.is_empty(), "HF requirements policy is empty");
    for requirement in &requirements {
        assert!(
            requirement.contains("==")
                && requirement
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || ".=_-+[]".contains(ch)),
            "HF requirement must be one exact, option-free package pin: {requirement}"
        );
    }

    let mut generated = String::from(
        "/// Exact direct requirements represented by the transitive lock.\n\
         /// Generated from requirements-hf.in; provisioning consumes the full lock.\n\
         pub const REQUIRED_PKGS: &[&str] = &[\n",
    );
    for requirement in requirements {
        writeln!(generated, "    {requirement:?},").expect("write generated requirement");
    }
    generated.push_str("];\n");

    let out =
        PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set")).join("hf_requirements.rs");
    fs::write(out, generated).expect("write generated HF requirements policy");
}
