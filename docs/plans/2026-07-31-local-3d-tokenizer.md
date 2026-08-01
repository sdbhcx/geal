# Local 3D Tokenizer Implementation Plan

> **For implementer:** Use TDD throughout. Write failing test first. Watch it fail. Then implement.

**Goal:** Add the V1 continuous local 3D tokenizer to GEAL without changing the default Branch3D behavior or public forward interface.

**Architecture:** A standalone geometry-only tokenizer samples centers with FPS, groups fixed-size KNN neighborhoods, encodes relative positions into continuous tokens, interpolates tokens back to all points, and fuses them into the existing dense PointNet++ feature with a zero-initialized gated residual. The feature is disabled by default and uses dedicated V1 configs.

**Tech Stack:** Python, PyTorch, pytest/unittest-compatible tests, YAML.

---

### Task 1: Tokenizer unit tests

**Files:**
- Create: `tests/test_local_3d_tokenizer.py`
- Create: `model/local_3d_tokenizer.py`

**Steps:**
1. Write tests for fixed output shapes, metadata indices, K clamping, translation invariance, interpolation identity, and zero-initialized fusion.
2. Run `python -m unittest tests.test_local_3d_tokenizer -v`; confirm import failure.
3. Implement the minimum tokenizer, interpolator, and fusion classes.
4. Run the unit tests; confirm pass.

### Task 2: Branch3D integration

**Files:**
- Modify: `model/branch_3d.py`
- Modify: `tests/test_local_3d_tokenizer.py`

**Steps:**
1. Add a lightweight source-level integration test that verifies tokenizer modules are guarded by `local_tokenizer.enabled` and fusion occurs before the Transformer decoder.
2. Run the integration test and confirm failure.
3. Register tokenizer modules in `Branch3D.__init__` and invoke them after multi-level fusion.
4. Keep the existing training and inference return signatures unchanged.
5. Run tests and Python compile checks.

### Task 3: Dedicated V1 configs

**Files:**
- Create: `config/train_stage2_v1_tokenizer.yaml`
- Create: `config/evaluation_v1_tokenizer.yaml`
- Create: `config/evaluation_corrupt_v1_tokenizer.yaml`

**Steps:**
1. Copy the current corresponding configs to preserve existing project-specific paths and model settings.
2. Add the same `model_3d.local_tokenizer` block to all three configs.
3. In the V1 training config, disable IAM/ADM and other optional new loss pipelines to isolate V1.
4. Parse all configs with `read_yaml` and assert the tokenizer structures match.

### Task 4: Final verification

**Files:**
- Verify all changed files.

**Steps:**
1. Run the tokenizer unit tests.
2. Run `python -m py_compile` for changed Python files.
3. Run configuration parity assertions.
4. Review git diff and check no unrelated user modifications were overwritten.
