# scripts/bootstrap.py
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REQ = REPO_ROOT / "req.txt"


def run(cmd: list[str], *, cwd: Path | None = None) -> int:
    print(f"\n$ {' '.join(cmd)}")
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert p.stdout is not None
    for line in p.stdout:
        print(line.rstrip("\n"))
    return p.wait()


def install_requirements() -> None:
    if not REQ.exists():
        raise FileNotFoundError(f"req.txt not found at {REQ}")
    code = run([sys.executable, "-m", "pip", "install", "-r", str(REQ)], cwd=REPO_ROOT)
    if code != 0:
        raise RuntimeError(f"pip install failed (exit_code={code})")


def main() -> None:
    print("[bootstrap] starting")
    print(f"[bootstrap] repo_root={REPO_ROOT}")

    install_requirements()

    print("\n[bootstrap] local LLM mode ✅ (skipping HF picker entirely)")
    print("[bootstrap] done ✅")


if __name__ == "__main__":
    main()
