"""
scripts.bootstrap
Only for installing 'req.txt' and then 
Streams 'pip' output line-by-line to terminal

Ouputs: Installed deps.
Prints: local LLM mode and [bootstrap] done
"""

# scripts/bootstrap.py
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REQ = REPO_ROOT / "req.txt"

# clone INSIDE src/, so git creates src/secguard automatically
SRC_DIR = REPO_ROOT / "src"
SECGUARD_DIR = SRC_DIR / "secguard"
SECGUARD_REPO = "https://github.com/adysinghh/secguard.git"


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


def ensure_secguard_repo() -> None:
    SRC_DIR.mkdir(parents=True, exist_ok=True)

    if not SECGUARD_DIR.exists():
        print(f"[bootstrap] cloning secguard into {SRC_DIR} (repo will become {SECGUARD_DIR})")
        code = run(["git", "clone", SECGUARD_REPO], cwd=SRC_DIR)
        if code != 0:
            raise RuntimeError(f"git clone secguard failed (exit_code={code})")
    else:
        print(f"[bootstrap] secguard already exists at {SECGUARD_DIR}, pulling latest")
        code = run(["git", "-C", str(SECGUARD_DIR), "pull"], cwd=REPO_ROOT)
        if code != 0:
            raise RuntimeError(f"git pull secguard failed (exit_code={code})")


def install_secguard() -> None:
    if not SECGUARD_DIR.exists():
        raise FileNotFoundError(f"secguard repo not found at {SECGUARD_DIR}")
    code = run([sys.executable, "-m", "pip", "install", "-e", str(SECGUARD_DIR)], cwd=REPO_ROOT)
    if code != 0:
        raise RuntimeError(f"secguard install failed (exit_code={code})")


def main() -> None:
    print("[bootstrap] starting")
    print(f"[bootstrap] repo_root={REPO_ROOT}")

    install_requirements()
    ensure_secguard_repo()
    install_secguard()

    print("\n[bootstrap] local LLM mode ✅ (skipping HF picker entirely)")
    print("[bootstrap] req.txt installed ✅")
    print("[bootstrap] secguard installed from src/secguard ✅")
    print("[bootstrap] done ✅")


if __name__ == "__main__":
    main()