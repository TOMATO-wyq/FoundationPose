#!/usr/bin/env python3
"""Fetch and verify locked SAM source/weights; never changes an existing env."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "src" / "sam2-lock.json").read_text())
    source = root / "third_party" / "sam2"
    if not source.exists():
        if args.verify_only:
            raise FileNotFoundError(source)
        source.parent.mkdir(exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", lock["repository"], str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", lock["commit"]], check=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != lock["commit"]:
        raise RuntimeError(f"SAM revision differs: {revision}; use a separate checkout, do not overwrite it")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise RuntimeError("SAM tracked source has local changes")
    target = source / "checkpoints" / lock["checkpoint_file"]
    if not target.exists():
        if args.verify_only:
            raise FileNotFoundError(target)
        target.parent.mkdir(exist_ok=True)
        temporary = target.with_suffix(".download")
        with urllib.request.urlopen(lock["checkpoint_url"], timeout=60) as response, temporary.open("wb") as file:
            while chunk := response.read(1024*1024):
                file.write(chunk)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != lock["checkpoint_sha256"]:
            raise RuntimeError("Downloaded checkpoint checksum mismatch; .download retained for diagnosis")
        temporary.rename(target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != lock["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint checksum mismatch")
    print(f"Verified SAM source {revision} and checkpoint {target}")


if __name__ == "__main__":
    main()
