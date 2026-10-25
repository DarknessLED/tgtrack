"""Fetch compact arXiv metadata (title, date, comment, abstract) for a list of IDs.

Usage: python scripts/arxiv_meta.py 1611.09224 1606.09549 ...
Writes a compact text digest to stdout.
"""
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}


def fetch(ids):
    url = "http://export.arxiv.org/api/query?id_list=" + ",".join(ids) + "&max_results=100"
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main():
    ids = sys.argv[1:]
    out = []
    for i in range(0, len(ids), 25):
        chunk = ids[i:i + 25]
        for attempt in range(3):
            try:
                raw = fetch(chunk)
                break
            except Exception as e:  # noqa: BLE001
                print("retry", e, file=sys.stderr)
                time.sleep(3)
        else:
            print("FAILED CHUNK", chunk, file=sys.stderr)
            continue
        root = ET.fromstring(raw)
        for e in root.findall("a:entry", NS):
            def txt(tag):
                n = e.find("a:" + tag, NS)
                return (n.text or "").strip() if n is not None else ""
            aid = txt("id").split("/abs/")[-1]
            pub = txt("published")[:10]
            upd = txt("updated")[:10]
            title = re.sub(r"\s+", " ", txt("title"))
            summ = re.sub(r"\s+", " ", txt("summary"))
            cmt = ""
            c = e.find("a:comment", NS)
            if c is not None and c.text:
                cmt = re.sub(r"\s+", " ", c.text).strip()
            auth = [a.find("a:name", NS).text for a in e.findall("a:author", NS)][:3]
            out.append(
                f"### {title}\nID {aid} | pub {pub} | upd {upd} | authors {', '.join(auth)}\n"
                f"COMMENT: {cmt}\nABSTRACT: {summ}\n"
            )
        time.sleep(3)
    print("\n".join(out))


if __name__ == "__main__":
    main()
