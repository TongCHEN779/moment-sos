"""Execute every ```python block of the documentation so that the examples cannot rot."""
import pathlib
import re
import pytest

DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"
BLOCK = re.compile(r"```python\n(.*?)```", re.S)
PAGES = sorted(p for p in DOCS.rglob("*.md") if "_build" not in p.parts)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(DOCS)))
def test_python_blocks_run(page):
    blocks = BLOCK.findall(page.read_text())
    if not blocks:
        pytest.skip("no python blocks")
    ns = {}                                                  # blocks of one page share a namespace, like a notebook
    for i, code in enumerate(blocks):
        try:
            exec(compile(code, f"{page.name}[{i}]", "exec"), ns)
        except Exception as e:                               # noqa: BLE001
            raise AssertionError(f"{page.name}, block {i} failed: {e!r}\n{code}") from e
