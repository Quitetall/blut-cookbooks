# Security policy

## Supported versions

Only the latest published `0.2` preview receives security fixes. Unpublished
stages and older internal tags are unsupported.

## Reporting

Use GitHub private vulnerability reporting for this repository. Do not open a
public issue containing an exploit, credential, private model, or private data.
Include affected version, reproduction steps, impact, and any proposed fix.

Never attach production datasets, checkpoints, access tokens, or job manifests
containing secrets. Replace them with minimal synthetic fixtures.

## Artifact boundary

Model files, checkpoints, datasets, Python packages, and external executables
are untrusted input. Consumers must verify provenance and hashes before use.
BLUT cookbook code must not deserialize pickle-bearing artifacts by default or
persist secrets in plans, logs, status events, or provenance records.
