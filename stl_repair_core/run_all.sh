#!/usr/bin/env bash
# One reproducible command: build stl_repair_core from a clean virtual
# environment and run the complete test suite.
#   ./run_all.sh            (needs python>=3.10, a Rust toolchain >=1.80, internet for pip/cargo)
# PyMeshLab (test oracle) needs the system library libOpenGL.so.0
# (Debian/Ubuntu: apt install libopengl0 libglu1-mesa).
set -euo pipefail
cd "$(dirname "$0")"
VENV=${VENV:-.venv-test}
rm -rf "$VENV"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q ".[test]" networkx
"$VENV/bin/python" -c "import stl_repair_core, stl_repair_core._native; print('built stl_repair_core', stl_repair_core.__version__, '(native kernel loaded)')"
cd tests && "../$VENV/bin/python" -m pytest -q -p no:cacheprovider --junitxml=../benchmarks/pytest_junit.xml "$@"
