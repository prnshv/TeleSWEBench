# Test cases

## Data source

Bundled rows live in `tele_swebench/dataset/bundled/*.json` (734 tasks). Each row may include `ctest_r_exact` shell commands used by the test runner.

## Execution

Tests run with working directory:

`outputs/runs/<run_id>/workspaces/<question_id>/ws/<repo>/`

(inner extracted tree for that task).

If the workspace is not configured/built, the runner reports `no_cmake_build_tree` instead of a misleading pass. For a full benchmark, plan either manual CMake configure/build in that tree or a future automated prepare step—see the README section on executable tests.
