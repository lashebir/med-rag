"""
Section-aware chunking utilities for RAG ingestion.
Preserves document structure while creating semantically coherent chunks.
"""
import re
from typing import List, Tuple

# Sentence splitter (crude but fast)
_SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+(?=[A-Z0-9(])')

def _sentences(text: str) -> List[str]:
    """Split text into sentences."""
    text = (text or "").strip()
    if not text:
        return []
    # Keep linebreaks as spaces
    text = re.sub(r'\s*\n+\s*', ' ', text)
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents if sents else [text]

def _tok_count(s: str) -> int:
    """Rough token count by whitespace splitting."""
    return len(s.split())

def detect_sections_from_text(text: str) -> List[Tuple[str, str]]:
    """
    Parse PDF-style text into sections.
    Returns list of (section_name, content) tuples.

    Section detection heuristics:
    1. Lines that are ALL CAPS or Title Case followed by newline
    2. Numbered sections like "1. Introduction" or "1 INTRODUCTION"
    3. Common section keywords (Abstract, Introduction, Methods, etc.)
    """
    lines = text.split('\n')
    sections: List[Tuple[str, str]] = []
    current_section = "Document"
    current_content: List[str] = []

    # Common section headers in academic papers
    SECTION_KEYWORDS = {
        'abstract', 'introduction', 'background', 'related work',
        'methods', 'methodology', 'materials and methods',
        'results', 'experiments', 'evaluation',
        'discussion', 'conclusion', 'conclusions',
        'references', 'bibliography', 'acknowledgments', 'acknowledgements',
        'appendix', 'supplementary', 'appendices'
    }

    # Patterns for section headers
    NUMBERED_SECTION = re.compile(r'^\s*(\d+\.?\s+)([A-Z][A-Za-z\s]+)$')
    ALL_CAPS = re.compile(r'^[A-Z\s\d\.\-]+$')
    TITLE_CASE = re.compile(r'^([A-Z][a-z]+\s+)+[A-Z][a-z]+$')

    def is_section_header(line: str) -> bool:
        """Check if a line is likely a section header."""
        line_stripped = line.strip()
        if not line_stripped or len(line_stripped) > 100:  # Too long to be header
            return False

        # Check for numbered sections
        if NUMBERED_SECTION.match(line_stripped):
            return True

        # Check for ALL CAPS headers (common in papers)
        if len(line_stripped) > 3 and ALL_CAPS.match(line_stripped):
            # Must be short enough and contain section keywords
            if any(kw in line_stripped.lower() for kw in SECTION_KEYWORDS):
                return True
            if len(line_stripped.split()) <= 5:  # Short ALL CAPS = likely header
                return True

        # Check for Title Case headers with keywords
        if TITLE_CASE.match(line_stripped):
            if any(kw in line_stripped.lower() for kw in SECTION_KEYWORDS):
                return True

        return False

    for line in lines:
        if is_section_header(line):
            # Save previous section
            if current_content:
                content = '\n'.join(current_content).strip()
                if content:
                    sections.append((current_section, content))

            # Start new section
            current_section = line.strip()
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_content:
        content = '\n'.join(current_content).strip()
        if content:
            sections.append((current_section, content))

    # Fallback: if no sections detected, create numbered sections by splitting roughly
    if not sections or len(sections) == 1:
        # Split into rough sections if text is long enough
        if len(text) > 1000:
            # Simple fallback: split by double newlines or length
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            if len(paragraphs) > 3:
                # Group paragraphs into sections - replace existing sections
                numbered_sections: List[Tuple[str, str]] = []
                section_size = max(1, len(paragraphs) // 3)
                for i in range(0, len(paragraphs), section_size):
                    section_num = (i // section_size) + 1
                    content = '\n\n'.join(paragraphs[i:i+section_size])
                    if content:
                        numbered_sections.append((f"Section {section_num}", content))
                return numbered_sections if numbered_sections else [("Section 1", text)]

    # Convert generic "Document" to "Section 1" when no meaningful sections found
    if sections and len(sections) == 1 and sections[0][0] == "Document":
        return [("Section 1", sections[0][1])]

    return sections if sections else [("Section 1", text)]

def chunk_sections(
    sections: List[Tuple[str, str]],
    max_tokens: int = 800,
    overlap_tokens: int = 80,
    include_section_name: bool = True
) -> List[Tuple[str, str]]:
    """
    Build chunks from sections, preserving section boundaries.

    Strategy:
    - Keep sections together when possible
    - Split large sections by sentences
    - Add overlap between chunks (sentence-based)
    - Optionally prefix chunks with section name

    Args:
        sections: List of (section_name, content) tuples
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Tokens to overlap between chunks
        include_section_name: Prepend section name to each chunk

    Returns:
        List of (chunk_text, section_name) tuples
    """
    chunks: List[Tuple[str, str]] = []  # List of (chunk_text, section_name)
    cur_sents: List[str] = []
    cur_tokens = 0
    cur_section_name: str = ""
    overlap_sents_last_chunk: List[str] = []

    def flush_chunk():
        nonlocal cur_sents, cur_tokens, overlap_sents_last_chunk
        if not cur_sents:
            return

        # Build chunk text
        if include_section_name and cur_section_name:
            chunk_text = f"[{cur_section_name}]\n" + "\n".join(cur_sents).strip()
        else:
            chunk_text = "\n".join(cur_sents).strip()

        if chunk_text:
            chunks.append((chunk_text, cur_section_name))

        # Compute overlap seed (tail sentences)
        toks = 0
        tail = []
        for s in reversed(cur_sents):
            toks += _tok_count(s)
            tail.append(s)
            if toks >= overlap_tokens:
                break
        overlap_sents_last_chunk = list(reversed(tail))

        # Reset buffer
        cur_sents = []
        cur_tokens = 0

    for section_name, content in sections:
        cur_section_name = section_name
        sents = _sentences(content)
        section_tokens = sum(_tok_count(s) for s in sents)

        # If whole section fits in remaining budget, add it
        if section_tokens <= max_tokens:
            if cur_tokens + section_tokens <= max_tokens:
                # Add section to current chunk
                if cur_sents:  # Add separator if continuing from previous content
                    cur_sents.append("")
                cur_sents.extend(sents)
                cur_tokens += section_tokens
            else:
                # Flush current chunk, start new one
                flush_chunk()
                # Add overlap from previous chunk
                if overlap_sents_last_chunk:
                    ov_tokens = sum(_tok_count(s) for s in overlap_sents_last_chunk)
                    if ov_tokens < max_tokens // 2:
                        cur_sents.extend(overlap_sents_last_chunk)
                        cur_tokens += ov_tokens
                # Add section
                cur_sents.extend(sents)
                cur_tokens += section_tokens

        else:
            # Section is too large - split by sentences
            flush_chunk()  # Flush current chunk first

            buf: List[str] = []
            btoks = 0
            for s in sents:
                stoks = _tok_count(s)
                if btoks + stoks <= max_tokens:
                    buf.append(s)
                    btoks += stoks
                else:
                    # Flush sub-chunk
                    if buf:
                        if include_section_name and section_name:
                            chunk_text = f"[{section_name}]\n" + "\n".join(buf).strip()
                        else:
                            chunk_text = "\n".join(buf).strip()
                        chunks.append((chunk_text, section_name))

                    # Compute overlap tail
                    toks = 0
                    tail = []
                    for ts in reversed(buf):
                        toks += _tok_count(ts)
                        tail.append(ts)
                        if toks >= overlap_tokens:
                            break

                    # Next sub-chunk starts with tail + current sentence
                    buf = list(reversed(tail))
                    btoks = sum(_tok_count(ts) for ts in buf)
                    buf.append(s)
                    btoks += stoks

            # Flush final sub-chunk
            if buf:
                if include_section_name and section_name:
                    chunk_text = f"[{section_name}]\n" + "\n".join(buf).strip()
                else:
                    chunk_text = "\n".join(buf).strip()
                chunks.append((chunk_text, section_name))

            # Clear overlap after giant section
            overlap_sents_last_chunk = []

    # Flush remainder
    flush_chunk()

    return chunks

def chunk_text_by_sections(
    text: str,
    max_tokens: int = 800,
    overlap_tokens: int = 80,
    include_section_names: bool = True
) -> List[Tuple[str, str]]:
    """
    High-level function: detect sections in text and chunk them.

    Args:
        text: The input text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap between chunks
        include_section_names: Include section name prefix in chunk text

    Returns:
        List of (chunk_text, section_name) tuples

    Usage:
        chunks_with_sections = chunk_text_by_sections(pdf_text, max_tokens=800, overlap_tokens=80)
        for chunk_text, section_name in chunks_with_sections:
            print(f"Section: {section_name}, Length: {len(chunk_text)}")
    """
    sections = detect_sections_from_text(text)
    return chunk_sections(sections, max_tokens, overlap_tokens, include_section_names)
