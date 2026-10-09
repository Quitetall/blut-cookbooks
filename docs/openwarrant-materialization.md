# OpenWarrant materialization repair

Shared contract: [OW-WAR-0047 adapter repair](https://github.com/Quitetall/OpenWarrant/blob/31aaff742ef1886dbd38a37da50b3a8908247bf8/docs/warrants/OW-WAR-0047/implementation/shared-materialization-contract.md).
That document owns cross-project scope and evidence requirements; this pointer
avoids a second drifting contract.

Behavior schema 2 of `materialize_dataset_path` copies the selected source into
stage-owned `dataset.jsonl`, hashes and counts the copied bytes, then publishes
atomically. Source-file changes cannot mutate the retained artifact. Failed reads
or validation preserve previous output, and an existing output symlink is
replaced rather than followed. Argument and DatasetJsonl wire schemas are unchanged.

The behavior version invalidates old locator-only cache entries. The source must
still be readable and callers must still supply a trusted, owned stage directory.
This is an implementation repair, not OpenWarrant qualification or a release.
