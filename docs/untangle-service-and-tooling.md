# Service, compiler and tooling structure

This continuation applies [untangle-codebase](https://github.com/caldoai/skills/blob/main/skills/untangle-codebase/SKILL.md) to WeightLang at `2157424cc547276eae3c20caf4bac81f0049d35e`. The [first pass](untangle-codebase.md) already consolidated tensor, parameter, service admission and publication contracts. This inventory covers the remaining service/compiler/tooling boundaries and the operation storage path. Discovery below was completed before implementation, with independent reviewers for service/client APIs, host compilation, and editor/viewer tooling.

## Decisions and evidence

### 1. Bind operation storage identities during preparation

An operation's compiled parameter order and write targets should refer directly to the service-owned storage slots. Parameters and state retain distinct public schemas and persistence rules.

- `wgtime-core/src/operations.rs::prepare` resolves initializer names repeatedly with linear searches.
- Every `call` reconstructs a name-to-storage map and resolves the same compiled names again.
- Write hooks allocate parameter-name strings and commits scan both storage collections to rediscover their destinations.
- The binding between a compiled program and its storage is immutable, so these lookups belong to preparation.

Regression evidence must cover reordered program declarations, repeated writes to one slot, updates across state and parameters, rollback, revision accounting, and weight files excluding state.

### 2. Discover client dependencies from the selected entrypoint

Client generation should follow ordinary function calls from its entrypoint. A static strategy call belongs to the service, while its argument expressions still belong to the client.

- `wgt/src/client_compiler.rs` uses two partial AST walkers to identify and remove service functions.
- Calls in conditions, unary expressions, assertions, tasks and other nested expressions can escape those walkers.
- A helper used by both the client and a service strategy is removed even though the client needs it.

One exhaustive dependency traversal should preserve shared helpers and named entrypoints without reproducing the service's strategy call graph.

### 3. Share synchronous and asynchronous expression semantics

The two client emitters should share literals, indexing, unary and binary operations, and explicit builtin dispatch. Runtime ownership and asynchronous operations remain explicit at their call sites.

- `client_compiler.rs` and `client_async_codegen.rs` independently emit the same expression forms.
- The asynchronous emitter treats `BuiltinCall` as environment-dispatched `Call`, despite the AST's explicit builtin semantics.
- Indexing and boolean checks have separate generated implementations, allowing their behavior to drift.

Behavior tests must exercise builtin shadowing, indexing, evaluated nonboolean operands and lazy boolean branches in both compilation paths.

The asynchronous AOT fixture also uses an implicit service endpoint and obsolete timer calls. Move it onto the existing isolated service fixture and supported task operations, retaining nested-await and return-value assertions. Scheduler cancellation/timing tests and exact runtime budget checks remain separate behavioral coverage.

### 4. Give service artifact descriptors one persistence boundary

One private service module should own the maximum artifact size, bounded reads from offset zero, and writes to empty staging descriptors.

- Five persistence request arms in `wgtime-service/src/main.rs` repeat size limits, rewind/read or staging/write/sync rules.
- Session restoration and admission are mixed with descriptor transport details inside the socket loop.

Keep backend dispatch, optimizer eligibility and reservation accounting with their existing owners. Check oversize rejection, rewind, nonempty-target rejection, exported bytes and failed-load reservation cleanup.

### 5. Resolve resident strategy call identities once

Strategy instructions should carry a validated builtin or helper target. Builtin arity and execution should use that same identity.

- `wgtime-service/src/strategy.rs` maintains an independent name/arity table and runtime string switch.
- Registration gives builtins precedence, while execution first checks source functions.
- Short-circuit lowering validates the left operand but leaves an evaluated right operand unchecked; the core evaluator validates both evaluated operands.

Resolve calls during registration and validate evaluated boolean results without making skipped branches eager. Preserve recursion rejection, capability restrictions, budgets and transactional strategy state.

### 6. Own translation pairs in one catalog

The viewer should derive both translation directions from one bilingual catalog while retaining standalone HTML and browser loading behavior.

- `web/viewer/locales/en.json` and `zh-CN.json` contain 177 exact inverse pairs.
- The viewer loads and indexes both copies; `wgt/src/graph.rs` embeds both copies again.

Check every translation in both directions, unknown identifiers, explicit registration and embedded standalone data. Rendering and layout remain governed by the existing viewer behavior tests.

### 7. Store conformance expectations with their source cases

Highlight capture expectations belong beside source and syntax-node expectations in `language/conformance.json`.

- `scripts/test_source_conformance.cjs` selects expected keywords with branches on two fixture IDs.
- A fixture change therefore requires a separately coordinated runner change.

Use a generic expectation loop and check that an incorrect expected capture produces a failure.

### 8. Ship the complete editor regeneration inputs

The editor source archive should contain the dependency closure of its advertised generator and validation scripts.

- `.github/workflows/publish-editors.yml` ships the generator without `language/prelude.json`.
- It ships the editor runner without the conformance runner and its input fixtures.
- Reproducing the current archive makes `generate_editor_support.cjs --check` fail with a missing prelude catalog.

Validate the extracted archive using the same regeneration and checking commands offered to its consumers.

### 9. Share browser-test lifecycle ownership

One fixture should own Playwright loading, Chromium selection, page errors and cleanup. Each test keeps its viewport, model fixture and behavioral assertions.

- Eight viewer browser scripts repeat setup and teardown.
- Seven hardcode Chromium while one honors `CHROMIUM`.

Run the existing browser scenarios through the common lifecycle; keep model-artifact preparation explicit.

### 10. Retain behavioral macro tests

Remove the macro unit test that asserts generated token spelling. Keep the real renamed-dependency compilation and transitive WGT dependency rebuild tests, which verify the behavior the spelling test approximates. Preserve raw-source escaping coverage.

## Required distinctions

The review retains transport poisoning versus complete application errors, descriptor consumption before metadata rejection, backend capabilities, atomic create versus explicit replacement, sampling versus ordinary writes, operation state versus persisted parameters, resource budgets, bounded readiness polling and public compatibility facades. CPU, packed and GPU arithmetic remain separate where their storage and execution requirements differ.

## Verification

Use Rust/Cargo 1.99.0 with locked offline dependencies. Record workspace compilation, focused behavioral suites, generated metadata/editor checks, archive reproduction, viewer tests and independent final review against the published commits. New regression gates should run in CI. Claims about runtime behavior require executable evidence; compilation alone is recorded as compilation.
