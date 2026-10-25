"""Download a PDF and print keyword-centred context windows (compact).

Usage: python scripts/pdf_grep.py <url> <outname> <kw1> <kw2> ...
Saves PDF under data/raw_pdf/<outname>.pdf; prints +/-N char windows around each hit,
deduplicated and truncated. Also writes the full extracted text next to it.
"""
import os
import re
import sys
import urllib.request

from pypdf import PdfReader

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw_pdf")
os.makedirs(OUT, exist_ok=True)


def download(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        f.write(r.read())
    return os.path.getsize(path)


def main():
    url, name = sys.argv[1], sys.argv[2]
    kws = [k.lower() for k in sys.argv[3:]]
    pdf = os.path.join(OUT, name + ".pdf")
    if not os.path.exists(pdf):
        print("downloading", url)
        try:
            print("bytes:", download(url, pdf))
        except Exception as e:  # noqa: BLE001
            print("DOWNLOAD FAILED:", e)
            return
    reader = PdfReader(pdf)
    pages = []
    for i, p in enumerate(reader.pages):
        try:
            pages.append(p.extract_text() or "")
        except Exception as e:  # noqa: BLE001
            pages.append("")
    full = "\n".join(pages)
    txt = os.path.join(OUT, name + ".txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(full)
    print(f"pages={len(pages)} chars={len(full)} text={txt}")
    flat = re.sub(r"[ \t]+", " ", full)
    seen = []
    for kw in kws:
        idxs = [m.start() for m in re.finditer(re.escape(kw), flat, re.I)]
        print(f"\n--- KW '{kw}' hits={len(idxs)}")
        for s in idxs[:6]:
            if any(abs(s - t) < 300 for t in seen):
                continue
            seen.append(s)
            lo, hi = max(0, s - 500), min(len(flat), s + 900)
            print("  >> " + re.sub(r"\s+", " ", flat[lo:hi]).strip())


if __name__ == "__main__":
    main()
