"""Chuyển file nguồn có đánh dấu cell (`# %%` và `# %% [markdown]`) thành notebook .ipynb.

Chạy:  .venv313/bin/python scripts/build_notebook.py scripts/nb_coupon_src.py notebooks/03_coupon_b0_h1_h2.ipynb
Sau đó thực thi:  .venv313/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/03_coupon_b0_h1_h2.ipynb
"""
from __future__ import annotations
import sys
from pathlib import Path
import nbformat


def parse_cells(text: str) -> list[tuple[str, str]]:
    """Tách văn bản nguồn thành danh sách (loại_cell, nội_dung).

    Tham số: text — toàn bộ nội dung file nguồn.
    Trả về: danh sách tuple ("markdown" | "code", nội dung đã bỏ dòng đánh dấu).
    Markdown viết dưới dạng chuỗi bình thường (không cần dấu #) sau dòng `# %% [markdown]`.
    """
    cells: list[tuple[str, str]] = []
    kind, buf = None, []
    for line in text.splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    if kind is not None:
        cells.append((kind, "\n".join(buf).strip("\n")))
    return [(k, c) for k, c in cells if c.strip()]


def build(src: Path, dst: Path) -> None:
    """Ghi notebook mới (chưa có output) từ file nguồn."""
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for kind, content in parse_cells(src.read_text(encoding="utf-8")):
        if kind == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(content))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(content))
    dst.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, dst)
    print(f"wrote {dst} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
