"""Credential-file safeguards and redaction of scanner output (#19)."""

import importlib.util
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_secrets", ROOT / "scripts/check_secrets.py")
scanner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanner)


@pytest.mark.parametrize("path,ignored", [
    ("agent.env", True), ("nested/agent.env", True), ("agent.env.backup", True),
    (".env.production", True), ("service-account-prod.json", True),
    ("worker.service-account.json", True), (".local-credentials/key.json", True),
    ("agent.env.example", False), (".env.example", False),
    (".env.production.example", False), ("service-account.example.json", False),
])
def test_local_credentials_are_ignored_but_examples_can_be_tracked(tmp_path, path, ignored):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text((ROOT / ".gitignore").read_text())
    result = subprocess.run(["git", "-C", str(tmp_path), "check-ignore", "--no-index", "-q", path])
    assert (result.returncode == 0) is ignored


def test_scanner_reports_only_metadata_even_when_its_process_prints_sensitive_values(tmp_path):
    def completed(*args, **kwargs):
        (tmp_path / "history.json").write_text(json.dumps([{
            "RuleID": "fixture-rule", "File": "fixture.txt", "StartLine": 1,
            "EndLine": 1, "Commit": "revision", "Secret": "sensitive-fixture-value",
            "Match": "sensitive-fixture-match",
        }]))
        return subprocess.CompletedProcess(args[0], 1, "sensitive-stdout", "sensitive-stderr")
    with patch.object(scanner.subprocess, "run", side_effect=completed):
        result = scanner.run_scan(Path("/fake/scanner"), ["git", "repo"], tmp_path, "history")
    serialized = json.dumps(result)
    assert result["finding_count"] == 1
    assert "sensitive" not in serialized
    assert set(result["findings"][0]) == {"RuleID", "File", "StartLine", "EndLine", "Commit"}


def test_scanner_fails_closed_on_process_error_without_repeating_its_output(tmp_path):
    with patch.object(scanner.subprocess, "run", return_value=subprocess.CompletedProcess([], 2, "raw-value", "raw-value")):
        with pytest.raises(RuntimeError) as error:
            scanner.run_scan(Path("/fake/scanner"), [], tmp_path, "history")
    assert "raw-value" not in str(error.value)
