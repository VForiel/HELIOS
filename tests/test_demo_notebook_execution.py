import sys
import subprocess
from pathlib import Path
import pytest


def test_execute_demo_notebook():
    """Run the headless notebook executor and assert it completes successfully.

    This test is skipped if `nbformat`/`nbconvert` are not installed in the
    current environment, since executing notebooks requires those packages.
    """
    try:
        import nbformat  # noqa: F401
    except Exception:
        pytest.skip("nbformat not installed; skipping notebook execution test")

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "execute_demo_notebook.py"
    assert script.exists(), f"Execution script not found: {script}"

    proc = subprocess.run([sys.executable, str(script)], cwd=str(repo_root))
    assert proc.returncode == 0, f"Notebook execution failed (rc={proc.returncode})"
