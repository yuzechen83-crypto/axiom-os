r"""
Download SPARC Rotmod_LTG.zip for offline use.
Run: py axiom_os/experiments/download_sparc.py
Then set: set SPARC_ZIP_PATH=c:\path\to\Rotmod_LTG.zip (Windows)
         export SPARC_ZIP_PATH=/path/to/Rotmod_LTG.zip (Linux/Mac)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "axiom_os" / "data"
OUT_FILE = OUT_DIR / "Rotmod_LTG.zip"
URLS = [
    "http://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip",
    "https://zenodo.org/records/16284118/files/Rotmod_LTG.zip?download=1",
]


def main():
    print("Downloading SPARC Rotmod_LTG.zip...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for url in URLS:
        try:
            from urllib.request import urlopen, Request
            req = Request(url, headers={"User-Agent": "AxiomOS/1.0"})
            with urlopen(req, timeout=60) as r:
                data = r.read()
                if len(data) > 1000:
                    OUT_FILE.write_bytes(data)
                    print(f"Saved to: {OUT_FILE}")
                    print(f"Size: {len(data)} bytes")
                    print("\nTo use: set SPARC_ZIP_PATH=" + str(OUT_FILE))
                    return
        except Exception as e:
            print(f"  {url}: {e}")
            continue
        try:
            import requests
            r = requests.get(url, timeout=60, headers={"User-Agent": "AxiomOS/1.0"})
            if r.status_code == 200 and len(r.content) > 1000:
                OUT_FILE.write_bytes(r.content)
                print(f"Saved to: {OUT_FILE}")
                print(f"Size: {len(r.content)} bytes")
                return
        except Exception as e:
            print(f"  {url}: {e}")

    print("Download failed. Manually download from:")
    print("  http://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip")
    print("  or https://zenodo.org/records/16284118")
    print(f"Save to: {OUT_FILE}")


if __name__ == "__main__":
    main()
