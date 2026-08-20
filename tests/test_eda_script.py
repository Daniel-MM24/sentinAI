import subprocess
import sys
from pathlib import Path


def test_eda_script_runs_successfully():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "eda_sentinai.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, result.stdout + result.stderr
