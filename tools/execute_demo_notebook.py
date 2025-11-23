"""Execute the examples/demo.ipynb notebook headlessly and save the executed notebook.

This script uses nbformat + nbconvert ExecutePreprocessor to run the notebook
from top to bottom. It will exit with a non-zero status if execution fails.
"""
import sys
from pathlib import Path

try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except Exception as exc:
    print("Required packages nbformat/nbconvert not available:", exc, file=sys.stderr)
    raise


def main(notebook_path: Path = Path("examples/demo.ipynb"), timeout: int = 600):
    notebook_path = Path(notebook_path)
    if not notebook_path.exists():
        print(f"Notebook not found: {notebook_path}", file=sys.stderr)
        return 2

    print(f"Executing notebook: {notebook_path}")
    nb = nbformat.read(str(notebook_path), as_version=4)

    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": str(notebook_path.parent)}})
    except Exception as exc:
        print("Error during notebook execution:", exc, file=sys.stderr)
        # save partial notebook for debugging
        out_path = Path("tools") / "_executed_demo_error.ipynb"
        nbformat.write(nb, str(out_path))
        print(f"Wrote partial executed notebook to {out_path}", file=sys.stderr)
        return 1

    out_path = Path("tools") / "_executed_demo.ipynb"
    nbformat.write(nb, str(out_path))
    print(f"Executed notebook saved to: {out_path}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
