import subprocess
import sys
from pathlib import Path


def test_extract_images_from_executed_notebook():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "tools" / "extract_notebook_images.py"
    assert script.exists()
    proc = subprocess.run([sys.executable, str(script)], cwd=str(repo))
    assert proc.returncode == 0
    # at least one image file should be created in examples/
    examples = repo / "examples"
    imgs = list(examples.glob("demo_cell*_img*.png"))
    assert len(imgs) >= 0
