import os, asyncio, httpx, sys, re, random, urllib.parse
from typing import Optional
from psycopg import connect
from psycopg.rows import dict_row
from bs4 import BeautifulSoup

from doi_utils import extract_doi_from_bioc, extract_doi_from_nxml, extract_doi_from_arxiv_entry, normalize_arxiv_id

# ---- arXiv id ↔ DOI helpers ----
ARXIV_VER_RE = re.compile(r"^arXiv:", re.IGNORECASE)

def normalize_arxiv_id(raw_id: str) -> str:
    rid = (raw_id or "").strip()
    rid = ARXIV_VER_RE.sub("", rid)          # remove leading 'arXiv:'
    rid = re.sub(r"v\d+$", "", rid)          # remove trailing version
    return rid

def datacite_doi_for_arxiv(arxiv_id: str) -> str:
    core = normalize_arxiv_id(arxiv_id)
    return f"10.48550/arXiv.{core}"

def normalize_doi_prefix(url_or_doi: str) -> str:
    # strip https://doi.org/ and case-normalize
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", (url_or_doi or "").strip(), flags=re.IGNORECASE).lower()

PG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "medrag"),
    user=os.getenv("PGUSER", os.getenv("USER")),
    password=os.getenv("PGPASSWORD") or None,
)

# ---------- HTTP helpers ----------
HEADERS = {"User-Agent": "med-rag/0.1 (contact: you@example.com)"}
CROSSREF_WORKS = "https://api.crossref.org/works/"
DATACITE_DOIS  = "https://api.datacite.org/dois/"

CR_HEADERS = {"User-Agent": "med-rag/0.1 (contact: ltashebir77@gmail.com)"}
DC_HEADERS = {"User-Agent": "med-rag/0.1 (contact: ltashebir77@gmail.com)"}

# replace your fetch_json()
async def fetch_json(url: str) -> Optional[dict]:
    headers = {"User-Agent": "med-rag/0.1 (contact: you@example.com)", "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        r = await client.get(url, headers=headers)
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except Exception:
            # BioC sometimes returns JSON with text/plain; try manual fix
            txt = r.text.strip()
            if not txt:
                return None
            try:
                import json
                return json.loads(txt)
            except Exception:
                return None

async def fetch_text(url: str, params=None) -> Optional[str]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params, headers=HEADERS)
        if r.status_code != 200:
            return None
        return r.text

async def fetch_pmc_bioc(pmcid: str) -> Optional[dict]:
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
    return await fetch_json(url)

async def fetch_pmc_nxml(pmcid: str) -> Optional[str]:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pmc", "id": pmcid, "retmode": "xml"}
    if os.getenv("NCBI_API_KEY"):
        params["api_key"] = os.getenv("NCBI_API_KEY")
    return await fetch_text(url, params=params)

# in enrich_doi.py
async def fetch_arxiv_entry(arxiv_id: str):
    api = "https://export.arxiv.org/api/query"
    headers = {"User-Agent": "med-rag/0.1 (contact: you@example.com)"}
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        r = await client.get(api, params={"search_query": f"id:{normalize_arxiv_id(arxiv_id)}", "max_results": "1"}, headers=headers)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "lxml-xml")
        return soup.find("entry")

# ---------- DB helpers ----------
def documents_has_doi_column(con) -> bool:
    with con.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'documents' AND column_name = 'doi'
            LIMIT 1;
        """)
        return cur.fetchone() is not None

def ensure_document_ids_table(con):
    with con.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS document_ids (
          doc_id    bigint REFERENCES documents(doc_id) ON DELETE CASCADE,
          id_type   text CHECK (id_type IN ('pmid','pmcid','doi','arxiv','biorxiv','medrxiv')),
          id_value  text,
          UNIQUE (id_type, lower(id_value))
        );
        """)
        con.commit()

def update_documents_doi(con, doc_id: int, doi: str):
    with con.cursor() as cur:
        cur.execute("UPDATE documents SET doi = %s WHERE doc_id = %s;", (doi, doc_id))
        con.commit()

def upsert_document_ids(con, doc_id: int, doi: str):
    with con.cursor() as cur:
        cur.execute("""
            INSERT INTO document_ids (doc_id, id_type, id_value)
            VALUES (%s, 'doi', %s)
            ON CONFLICT (id_type, lower(id_value)) DO NOTHING;
        """, (doc_id, doi))
        con.commit()

def upsert_document_id(con, doc_id: int, id_type: str, id_value: str):
    with con.cursor() as cur:
        cur.execute("""
            INSERT INTO document_ids (doc_id, id_type, id_value)
            VALUES (%s, %s, %s)
            ON CONFLICT (id_type, lower(id_value)) DO NOTHING;
        """, (doc_id, id_type, id_value))
        con.commit()

def set_documents_doi_if_column(con, doc_id: int, doi: str):
    # only if you created documents.doi column
    try:
        with con.cursor() as cur:
            cur.execute("UPDATE documents SET doi = %s WHERE doc_id = %s;", (doi, doc_id))
        con.commit()
    except Exception:
        pass
async def resolve_journal_doi_from_arxiv_doi(arxiv_doi: str) -> str | None:
    """
    Given a DataCite arXiv DOI (e.g., 10.48550/arXiv.2510.09135) return a journal DOI if known.
    Strategy:
      1) Crossref: /works/<arxiv_doi> and check 'message.relation' fields (is-preprint-of / has-version).
      2) DataCite: /dois/<arxiv_doi> and scan 'relatedIdentifiers' with relationType 'IsPreprintOf'.
    """
    if not arxiv_doi:
        return None
    arxiv_doi_norm = normalize_doi_prefix(arxiv_doi)

    # --- Crossref ---
    try:
        url = CROSSREF_WORKS + urllib.parse.quote(arxiv_doi_norm, safe="")
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            r = await client.get(url, headers=CR_HEADERS)
        if r.status_code == 200:
            msg = r.json().get("message", {})
            # Crossref “relation” is a dict of lists keyed by relation type
            rel = msg.get("relation") or {}
            for key in ("is-preprint-of", "has-version"):
                if key in rel and isinstance(rel[key], list):
                    for item in rel[key]:
                        jdoi = (item.get("DOI") or "").strip()
                        if jdoi:
                            return normalize_doi_prefix(jdoi)
    except Exception:
        pass

    # --- DataCite fallback ---
    try:
        url = DATACITE_DOIS + urllib.parse.quote(arxiv_doi_norm, safe="")
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            r = await client.get(url, headers=DC_HEADERS)
        if r.status_code == 200:
            data = r.json().get("data", {})
            attrs = data.get("attributes", {})
            rels  = attrs.get("relatedIdentifiers") or []
            # look for IsPreprintOf / IsVersionOf pointing to a 10.x DOI
            for ri in rels:
                reltype = (ri.get("relationType") or "").lower()
                v = normalize_doi_prefix(ri.get("relatedIdentifier") or "")
                if reltype in ("ispreprintof", "isversionof") and v.startswith("10."):
                    return v
    except Exception:
        pass

    return None

# ---------- Worker ----------
async def enrich_row(row, write_to_documents: bool, con):
    doc_id   = row["doc_id"]
    source   = (row.get("source") or "").lower()
    source_id= (row.get("source_id") or "").strip()

    doi: Optional[str] = None

    try:
        if source == "pubmed central" and source_id.startswith("PMC"):
            bioc = await fetch_pmc_bioc(source_id)
            if bioc:
                doi = extract_doi_from_bioc(bioc)
            if not doi:
                nxml = await fetch_pmc_nxml(source_id)
                if nxml:
                    doi = extract_doi_from_nxml(nxml)

        elif source == "arxiv" and source_id:
            entry = await fetch_arxiv_entry(source_id)
            # First: try to extract journal DOI directly from the entry
            jdoi = extract_doi_from_arxiv_entry(entry) if entry else None

            # Always synthesize the DataCite arXiv DOI (exists for every e-print)
            arxiv_doi = datacite_doi_for_arxiv(source_id)
            arxiv_doi = normalize_doi_prefix(arxiv_doi)

            # If we didn’t get a journal DOI from the entry, try Crossref/DataCite resolution
            if not jdoi:
                jdoi = await resolve_journal_doi_from_arxiv_doi(arxiv_doi)

            # Write both (if available)
            if arxiv_doi:
                upsert_document_id(con, doc_id, "arxiv-doi", arxiv_doi)
            if jdoi:
                upsert_document_id(con, doc_id, "doi", jdoi)
                set_documents_doi_if_column(con, doc_id, jdoi)
                print(f"[ok] doc_id={doc_id} arXiv {source_id} → journal DOI {jdoi}")
            else:
                print(f"[ok] doc_id={doc_id} arXiv {source_id} → arXiv DOI only ({arxiv_doi})")


        if doi:
            if write_to_documents:
                update_documents_doi(con, doc_id, doi)
            else:
                upsert_document_ids(con, doc_id, doi)
            print(f"[ok] doc_id={doc_id} source={source} → doi={doi}")
        else:
            print(f"[skip] doc_id={doc_id} (no doi found)")

    except Exception as e:
        print(f"[err] doc_id={doc_id}: {type(e).__name__}: {e}")

async def main():
    limit = int(os.getenv("ENRICH_LIMIT", "1000"))

    with connect(**PG, row_factory=dict_row) as con:
        # Decide where to store DOIs
        write_to_documents = documents_has_doi_column(con)
        if not write_to_documents:
            ensure_document_ids_table(con)

        # Pull candidates that don’t have a DOI yet
        if write_to_documents:
            q = """
            SELECT doc_id, source, source_id
            FROM documents
            WHERE doi IS NULL
            ORDER BY doc_id DESC
            LIMIT %s;
            """
        else:
            q = """
            SELECT d.doc_id, d.source, d.source_id
            FROM documents d
            LEFT JOIN document_ids di
              ON di.doc_id = d.doc_id AND di.id_type = 'doi'
            WHERE di.doc_id IS NULL
            ORDER BY d.doc_id DESC
            LIMIT %s;
            """

        with con.cursor() as cur:
            cur.execute(q, (limit,))
            rows = [dict(r) for r in cur.fetchall()]

        # Process sequentially (easy to reason about). You can add concurrency later.
        for r in rows:
            await enrich_row(r, write_to_documents, con)

if __name__ == "__main__":
    asyncio.run(main())