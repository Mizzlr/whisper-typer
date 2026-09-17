#!/usr/bin/env python3
"""Provision optional SymSpell frequencies; never called during dictation."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

REVISION = "2dbf3b2d766d5d530e1c961e7edc7dbae2d94f4c"
FILENAME = "frequency_dictionary_en_82_765.txt"
URL = f"https://raw.githubusercontent.com/reneklacan/symspell/{REVISION}/data/{FILENAME}"
SHA256 = "43223aa83b55519851d5c93baf96f7843e5f5410f2623a9ce263409ab50314b2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path.home() / ".config/whisper-typer")
    args = parser.parse_args()
    destination = args.directory / FILENAME
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != SHA256:
            raise SystemExit(f"Refusing to replace customized frequency data: {destination}")
        print(f"Already verified: {destination}")
        return
    data = urllib.request.urlopen(URL, timeout=30).read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise SystemExit("Downloaded frequency data failed checksum verification")
    args.directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.directory, prefix=".spelling-", delete=False) as temporary:
        temporary.write(data)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        # A hard link publishes without overwriting a concurrently created file.
        os.link(temporary_path, destination)
    finally:
        temporary_path.unlink()
    print(f"Verified and installed: {destination}\nSource: {URL}\nSHA256: {SHA256}")


if __name__ == "__main__":
    main()
