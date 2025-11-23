"""Extract image outputs (image/png or image/jpeg) from an executed notebook.

Usage: run from repository root.
"""
import sys
from pathlib import Path
import base64
import nbformat

def main(executed_nb: Path = Path("tools/_executed_demo.ipynb"), outdir: Path = Path("examples")):
    executed_nb = Path(executed_nb)
    outdir = Path(outdir)
    if not executed_nb.exists():
        print(f"Executed notebook not found: {executed_nb}")
        return 2
    nb = nbformat.read(str(executed_nb), as_version=4)
    outdir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, cell in enumerate(nb.get("cells", [])):
        outputs = cell.get("outputs", [])
        for j, out in enumerate(outputs):
            data = out.get("data") or {}
            if "image/png" in data:
                b64 = data["image/png"]
                if isinstance(b64, list):
                    b64 = "".join(b64)
                img = base64.b64decode(b64)
                fname = outdir / f"demo_cell{i+1}_img{j+1}.png"
                with open(fname, "wb") as f:
                    f.write(img)
                print(f"Wrote {fname}")
                count += 1
            elif "image/jpeg" in data:
                b64 = data["image/jpeg"]
                if isinstance(b64, list):
                    b64 = "".join(b64)
                img = base64.b64decode(b64)
                fname = outdir / f"demo_cell{i+1}_img{j+1}.jpg"
                with open(fname, "wb") as f:
                    f.write(img)
                print(f"Wrote {fname}")
                count += 1

    print(f"Extracted {count} images to {outdir}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
