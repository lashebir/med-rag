import re
from typing import Optional, Dict
from bs4 import BeautifulSoup

DOI_PREFIX_RE  = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)
DOI_INLINE_RE  = re.compile(r"\b10\.\d{4,9}/\S+\b", re.IGNORECASE)
_ARXIV_VER_RE = re.compile(r"^arXiv:", re.IGNORECASE)

def normalize_arxiv_id(raw_id: str) -> str:
    rid = (raw_id or "").strip()
    rid = _ARXIV_VER_RE.sub("", rid)   # remove leading 'arXiv:'
    rid = re.sub(r"v\d+$", "", rid)    # remove trailing version
    return rid

def normalize_doi(raw: str) -> str:
    s = (raw or "").strip()
    s = DOI_PREFIX_RE.sub("", s)
    return s.lower()

def datacite_doi_for_arxiv(arxiv_id: str) -> str:
    core = normalize_arxiv_id(arxiv_id)
    return f"10.48550/arXiv.{core}"

def extract_doi_from_bioc(bioc: dict) -> Optional[str]:
    # ... keep your existing logic ...
    # If not found, scan passages’ text:
    docs = []
    if isinstance(bioc, dict):
        docs = bioc.get("documents") or bioc.get("collection", {}).get("documents") or []
    elif isinstance(bioc, list):
        docs = bioc
    for d in docs:
        for p in d.get("passages", []):
            t = (p.get("text") or "").strip()
            if t:
                m = DOI_INLINE_RE.search(t)
                if m:
                    return normalize_doi(m.group(0))
    return None

def extract_doi_from_nxml(nxml: str) -> Optional[str]:
    if not nxml:
        return None
    soup = BeautifulSoup(nxml, "lxml-xml")
    tag = soup.find("article-id", {"pub-id-type": "doi"})
    if tag:
        return normalize_doi(tag.get_text(" ", strip=True))
    # rare: meta tags
    meta = soup.find("meta", {"name": "citation_doi"})
    if meta and meta.get("content"):
        return normalize_doi(meta["content"])
    return None

def extract_doi_from_arxiv_entry(entry) -> Optional[str]:
    if entry is None:
        return None

    # 1) Journal DOI in <arxiv:doi>
    tag = entry.find("arxiv:doi")
    if tag:
        s = tag.get_text(strip=True)
        if s:
            return normalize_doi(s)

    # 2) DOI link <link title="doi" href="https://doi.org/...">
    for link in entry.find_all("link"):
        if (link.get("title") or "").lower() == "doi" and link.get("href"):
            return normalize_doi(link["href"])

    # 3) Occasionally in journal_ref: "... doi:10.xxxx/yyy"
    jref = entry.find("arxiv:journal_ref")
    if jref:
        m = re.search(DOI_INLINE_RE, jref.get_text(" ", strip=True))
        if m:
            return normalize_doi(m.group(0))

    # 4) Fallback: synthesize DataCite DOI for the arXiv e-print
    #    This exists even if not shown in the Atom entry.
    aid_node = entry.find("id")
    if aid_node:
        aid = aid_node.get_text(strip=True).split("/abs/")[-1]
        return datacite_doi_for_arxiv(aid)

    return None
