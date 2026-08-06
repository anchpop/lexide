#!/usr/bin/env python3
"""Acquire the official GPL Hindi Formal Akshara-to-Sound release.

The release is retained as an auditable comparison backend, not silently used
as training truth.  It is a 32-bit PyInstaller/Python-2 program, so acquisition
and execution are intentionally separate steps.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


URL = (
    "https://downloads.sourceforge.net/project/pls-for-indic-languages/"
    "Re_6/Re_6/Hindi_Formal_A2S_Linux.zip"
)
SHA256 = "b17f2915e0c2563ffa1334d9ce793bb7c316c5af19c2144eaf6218578880f5da"
MEMBER = "Hindi_Formal_A2S_Linux/Hindi_Word_Stress_Formal"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "lexide-pronunciation" / "hindi-a2s",
    )
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    archive = args.cache_dir / "Hindi_Formal_A2S_Linux.zip"
    if not archive.exists() or sha256(archive) != SHA256:
        temporary = archive.with_suffix(".zip.part")
        with urllib.request.urlopen(URL) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        if sha256(temporary) != SHA256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Hindi A2S archive checksum mismatch")
        temporary.replace(archive)

    executable = args.cache_dir / "Hindi_Word_Stress_Formal"
    with zipfile.ZipFile(archive) as package, package.open(MEMBER) as source:
        with executable.open("wb") as out:
            shutil.copyfileobj(source, out)
    executable.chmod(0o755)

    print(f"archive: {archive}")
    print(f"sha256: {SHA256}")
    print(f"executable: {executable}")
    print("license: GPLv3 (project release metadata/source package)")
    print("note: 32-bit frozen Python 2; use only through a reviewed extraction")


if __name__ == "__main__":
    main()
