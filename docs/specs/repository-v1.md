# WeightLang repository references v1 and revision records v2

Status: implemented experimental interoperability contract. Existing immutable
revision v1 records and raw objects retain their identities and encoding.

## Revision v2

Revision v2 is [revision v1](revision-v1.md) with `format_version: 2` and exactly
one additional required field, `message`, a string. Everything else, including
object/file tables, parent meaning, hash verification and reconstruction, is
unchanged. See [schema](revision-v2.schema.json).

Canonical JSON uses recursively sorted keys, no extra whitespace or newline,
ASCII escaping via Python JSON `ensure_ascii=True`, and no non-finite numbers.
For message characters above U+FFFF, escape UTF-16 surrogate pairs; hex digits
in `\uXXXX` escapes are lowercase. The revision ID hashes this entire record,
including message and parent. No timestamp or mutable branch name is hashed.
Old readers that accept only v1 MUST reject v2; new readers MUST preserve and read
v1 unchanged. The low-level `RevisionStore.save(message=None)` still writes v1;
repository commits write v2 even when their message is empty.

## refs.json v1

```
STORE/
  objects/<sha256>
  revisions/<sha256>
  refs.json
  refs.lock             present only while a writer owns the lock
```

Example:

```json
{
  "format": "weightlang.refs",
  "format_version": 1,
  "head": {"branch": "main"},
  "branches": {"main": "<revision ID>", "experiment": "<revision ID>"}
}
```

See [schema](refs-v1.schema.json). Fields are exact: extra fields MUST be rejected.
`branches` maps branch names to full lowercase SHA-256 revision IDs or null for an
unborn branch. HEAD MUST be exactly `{"branch": NAME}` for an existing entry, or
`{"revision": ID}` for detached HEAD. A detached ID cannot be null. A branch name
matches `[A-Za-z][A-Za-z0-9_-]*(/[A-Za-z][A-Za-z0-9_-]*)*` and MUST NOT equal `HEAD`.
Branch names are JSON keys, never filesystem paths.

refs.json is mutable and is not content addressed. Writers emit canonical JSON as
above. Readers MUST validate structure and resolve the selected revision through
the hash-checking revision reader. They MUST NOT follow a symlink for refs.json.
Readers may read concurrently: atomic replacement ensures a complete old or new
state. Readers of exported artifacts do not need refs.json.

All mutations of refs MUST first acquire refs.lock using exclusive creation and
MUST fail if it already exists. Store the owner's decimal process ID for diagnostics;
it does not authorize automatic lock stealing. Write a new complete refs file to
a sibling temporary file, flush and fsync it, then atomically replace refs.json.
Release the lock after completion/failure. Publish immutable objects and revision
before updating refs. A failed ref update MUST leave the previous valid refs in
place; valid unreachable objects may remain. Directory fsync/power-loss durability
is not guaranteed in this implementation.

## Commit behavior

The parent MUST be the selected HEAD at the time of commit. Successful commits
advance only the selected branch, or detached HEAD if no branch is selected.
A byte-identical artifact (all three file hashes unchanged) MUST be a no-op;
message changes alone do not make a new commit. Concurrent builds MUST compare
both selected branch and tip with their pre-training token. On mismatch, preserve
the result as a revision with the original tip as parent, report its ID, and do
not update refs. An explicit legacy parent option MUST match HEAD; it is not a
history override.

A fresh repository defaults to unborn main. Initializing a legacy object store
without refs may adopt the sole tip (a revision not named as another stored
revision's parent). Multiple tips require an explicit starting revision; MUST NOT
choose based on filesystem times or iteration order. Initialization is idempotent
without a requested starting point and MUST NOT reset an existing repository.

Switching a branch changes refs only; checkout exports to a new directory without
changing refs. Branch creation shares objects. No automatic pruning, resetting,
merging, remote synchronization, implicit training resumption or mutation of
exported model directories is part of this contract.
