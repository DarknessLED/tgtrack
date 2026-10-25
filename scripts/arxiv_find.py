"""arXiv title search -> compact digest.

Usage: python scripts/arxiv_find.py "Thigh title one" "Title two" ...
Prints, for each query, the best-matching entry: id, pub date, comment, abstract (truncated).
"""
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}


def q(title, n=2):
    # accept a raw query when it already contains a field prefix / boolean op
    if re.search(r"\b(AND|OR|ti:|abs:|all:)\b", title):
        sq = title
    else:
        sq = f'ti:"{title}"'
    url = (
        "http://export.arxiv.org/api/query?search_query="
        + urllib.parse.quote(sq)
        + f"&start=0&max_results={n}&sortBy=relevance"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return ET.fromstring(r.read())


def main():
    for t in sys.argv[1:]:
        for attempt in range(3):
            try:
                root = q(t)
                break
            except Exception as e:  # noqa: BLE001
                print("retry", e, file=sys.stderr)
                time.sleep(3)
        else:
            print(f"=== QUERY: {t}\n  FAILED\n")
            continue
        ents = root.findall("a:entry", NS)
        print(f"=== QUERY: {t}")
        if not ents:
            print("  (no result)\n")
            continue
        for e in ents[:2]:
            def txt(tag):
                n = e.find("a:" + tag, NS)
                return (n.text or "").strip() if n is not None else ""
            aid = txt("id").split("/abs/")[-1]
            title = re.sub(r"\s+", " ", txt("title"))
            summ = re.sub(r"\s+", " ", txt("summary"))
            c = e.find("a:comment", NS)
            cmt = re.sub(r"\s+", " ", (c.text or "")).strip() if c is not None else ""
            auth = [a.find("a:name", NS).text for a in e.findall("a:author", NS)][:2]
            print(f"  - {title} [{aid}] pub {txt('published')[:10]} | {', '.join(auth)}")
            print(f"    COMMENT: {cmt}")
            print(f"    ABS: {summ[:900]}")
        print()
        time.sleep(3)


if __name__ == "__main__":
    main()
