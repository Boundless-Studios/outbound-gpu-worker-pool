"""The README quickstart is executable documentation.

The test runs the exact block the README publishes, so the snippet cannot drift
away from the library it advertises.
"""

import hashlib
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
QUICKSTART_HEADING = "## Quickstart"
PYTHON_FENCE = "```python\n"
FENCE = "```"


def _quickstart_source() -> str:
    body = README.read_text().split(QUICKSTART_HEADING, 1)[1]
    return body.split(PYTHON_FENCE, 1)[1].split(FENCE, 1)[0]


def test_the_readme_quickstart_completes_one_job(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _quickstart_source()

    exec(compile(source, str(README), "exec"), {"__name__": "__main__"})

    lines = capsys.readouterr().out.rstrip("\n").splitlines()
    digest = hashlib.sha256(b"hello worker pool").hexdigest()
    assert lines == [
        "completed -> completed",
        "deterministic-echo/v1",
        "label=quickstart",
        "seed=7",
        f"sha256={digest}",
    ]
