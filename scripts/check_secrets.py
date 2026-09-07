#!/usr/bin/env python3
"""Full-history and raw-object Gitleaks checks with metadata-only public output.

The caller fetches refs first. No repository code is executed, ignored-file
rules cannot hide tracked objects, and no raw scanner output is printed.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def run_scan(tool: Path, arguments: list[str], work: Path, name: str) -> dict:
    report = work / f"{name}.json"
    config = work / "default-rules.toml"
    config.write_text("[extend]\nuseDefault = true\n")
    ignore = work / "empty-ignore"
    ignore.write_text("")
    result = subprocess.run(
        [str(tool), *arguments, "--no-banner", "--redact=100",
         "--ignore-gitleaks-allow", "--max-decode-depth=2", "--max-archive-depth=2",
         "--config=" + str(config), "--gitleaks-ignore-path=" + str(ignore),
         "--report-format=json", "--report-path=" + str(report)],
        cwd=work, capture_output=True, text=True, timeout=300,
    )
    # Never print stdout/stderr or matched credential fields, even on failure.
    if result.returncode not in (0, 1) or not report.is_file():
        raise RuntimeError(f"{name}: scanner failed to complete")
    findings = json.loads(report.read_text())
    if not isinstance(findings, list) or (result.returncode == 1 and not findings):
        raise RuntimeError(f"{name}: inconsistent scanner result")
    safe = [
        {key: finding.get(key) for key in ("RuleID", "File", "StartLine", "EndLine", "Commit")}
        for finding in findings
    ]
    return {"scan": name, "finding_count": len(safe), "findings": safe}


def check(repository: Path, tool: Path) -> int:
    repository, tool = repository.resolve(), tool.resolve()

    def git(*args: str, input: bytes | None = None) -> bytes:
        return subprocess.check_output(
            ["git", "-C", str(repository), *args], input=input,
            stderr=subprocess.PIPE, timeout=120,
        )

    if git("rev-parse", "--is-shallow-repository").strip() != b"false":
        raise RuntimeError("full history is required; shallow checkout refused")
    git("fsck", "--full", "--no-dangling")
    objects = git("rev-list", "--objects", "--all", "HEAD", "--no-object-names")
    descriptions = git(
        "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)", input=objects
    ).decode().splitlines()
    with tempfile.TemporaryDirectory(prefix="ogwp-secrets-") as directory:
        work = Path(directory)
        corpus = work / "corpus"
        corpus.mkdir()
        counts = {"blob": 0, "commit": 0}
        for description in descriptions:
            sha, kind, size = description.split()
            if kind not in counts:
                continue
            content = git("cat-file", kind, sha)
            if len(content) != int(size):
                raise RuntimeError("incomplete object read")
            (corpus / f"{kind}-{sha}.txt").write_bytes(content)
            counts[kind] += 1
        results = [
            run_scan(tool, ["git", str(repository), "--log-opts=--all HEAD --full-history -m"], work, "history"),
            run_scan(tool, ["dir", str(corpus)], work, "raw-objects"),
        ]
        print("SECRETS_CHECK " + json.dumps({"objects": counts, "scans": results}, sort_keys=True))
        return 1 if any(result["finding_count"] for result in results) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--gitleaks", type=Path, required=True)
    args = parser.parse_args()
    try:
        return check(args.repository, args.gitleaks)
    except Exception as exc:
        # Exception text can include scanner output or inputs: only its type is public.
        print(f"Secrets check incomplete ({type(exc).__name__}); failing closed.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
