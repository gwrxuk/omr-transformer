"""Download the OpenScore Lieder corpus (CC0) and convert it to MusicXML.

    python scripts/fetch_openscore.py            # -> data/openscore/*.mscx, *.musicxml

Needs the GitHub CLI (`gh`) for the file list and MuseScore 4 for conversion.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

MSCORE = "/Applications/MuseScore 4.app/Contents/MacOS/mscore"
RAW = "https://raw.githubusercontent.com/OpenScore/Lieder/main/"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/openscore")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    paths = subprocess.run(["gh", "api", "repos/OpenScore/Lieder/git/trees/main?recursive=1", "--jq",
                            '.tree[] | select(.path|endswith(".mscx")) | .path'],
                           capture_output=True, text=True, check=True).stdout.split()
    (out / "all_paths.txt").write_text("\n".join(sorted(paths)) + "\n")

    def get(p: str) -> str:
        dst = out / (Path(p).stem + ".mscx")
        if dst.exists() and dst.stat().st_size:
            return "skip"
        try:
            urllib.request.urlretrieve(RAW + urllib.parse.quote(p), dst)
            return "ok"
        except Exception:
            return "error"

    with cf.ThreadPoolExecutor(16) as ex:
        results = list(ex.map(get, paths))
    print({r: results.count(r) for r in set(results)}, flush=True)

    jobs = [{"in": str(p.resolve()), "out": str(p.with_suffix(".musicxml").resolve())}
            for p in sorted(out.glob("*.mscx")) if not p.with_suffix(".musicxml").exists()]
    if jobs:
        job = out / "convert_job.json"
        job.write_text(json.dumps(jobs))
        subprocess.run([MSCORE, "-j", str(job)], capture_output=True, timeout=7200)
    print("musicxml files:", len(list(out.glob("*.musicxml"))))


if __name__ == "__main__":
    main()
