"""Fetch a Google Patents page and print compact metadata + first claims.

Usage: python scripts/patent_grep.py <patent_url_or_id> [more ids...]
"""
import re
import sys
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def strip(t):
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"&nbsp;?", " ", t)
    t = re.sub(r"&amp;", "&", t)
    return re.sub(r"\s+", " ", t).strip()


def one(pid):
    if pid.startswith("http"):
        url = pid
        pid = pid.rstrip("/").split("/")[-1].replace("/en", "")
    else:
        url = f"https://patents.google.com/patent/{pid}/en"
    try:
        req = urllib.request.Request(url, headers=UA)
        html = urllib.request.urlopen(req, timeout=90).read().decode("utf-8", "ignore")
    except Exception as e:  # noqa: BLE001
        print(f"=== {pid}: FETCH FAILED {e}\n")
        return
    def meta(name):
        m = re.search(r'<meta name="' + name + r'" content="([^"]*)"', html)
        return strip(m.group(1)) if m else ""
    title = meta("DC.title") or meta("citation_title")
    dates = meta("DC.date")
    assignee = ""
    m = re.search(r'"assigneeOriginal"\s*:\s*"([^"]*)"', html) or re.search(
        r'itemprop="assigneeOriginal"[^>]*>([^<]*)<', html)
    if m:
        assignee = strip(m.group(1))
    ab = meta("description") or meta("DC.description")
    nclaims = len(re.findall(r'<div class="claim"', html))
    print(f"=== {pid} | {title}")
    print(f"    assignee={assignee} | dates={dates} | claims_found={nclaims}")
    if ab:
        print(f"    META-DESC: {ab[:700]}")
    # abstract section
    m = re.search(r'<section[^>]*itemprop="abstract"[^>]*>(.*?)</section>', html, re.S)
    if m:
        print(f"    ABSTRACT: {strip(m.group(1))[:900]}")
    # claims
    m = re.search(r'<section[^>]*itemprop="claims"[^>]*>(.*?)</section>', html, re.S)
    if m:
        claims = strip(m.group(1))
        print(f"    CLAIMS: {claims[:2200]}")
    # first independent-ish description snippet around keywords
    for kw in ["re-identif", "occlu", "lost", "appearance", "similar"]:
        idxs = [mm.start() for mm in re.finditer(kw, html, re.I)][:2]
        for s in idxs:
            seg = strip(html[max(0, s - 300): s + 400])
            if len(seg) > 80:
                print(f"    [{kw}] ...{seg}...")
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        one(p)
