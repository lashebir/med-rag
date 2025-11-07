"""
NER extraction pipeline for medical/scientific documents.
Extracts entities in 4 discrete categories: disease, drug, device, statistical_significance
"""
import re
import spacy
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer

# Load models globally (lazy loading)
_nlp = None
_embedder = None

def get_nlp():
    """Lazy load the scispacy NER model."""
    global _nlp
    if _nlp is None:
        print("Loading en_core_sci_lg model...", flush=True)
        _nlp = spacy.load("en_core_sci_lg")
    return _nlp

def get_embedder():
    """Lazy load the sentence transformer for mention embeddings."""
    global _embedder
    if _embedder is None:
        print("Loading BAAI/bge-small-en-v1.5 for mention embeddings...", flush=True)
        _embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
    return _embedder

# Statistical significance patterns
STAT_SIG_PATTERNS = [
    # P-values
    r'\bp\s*[<>=≤≥]\s*0\.\d+\b',
    r'\bp-value\s*[<>=≤≥]\s*0\.\d+\b',
    r'\bp\s*=\s*\.\d+\b',
    # Confidence intervals
    r'\b\d+%\s*(?:confidence|CI)\s*interval',
    r'\bCI\s*[:\s]*\[?[\d\.\-\s,]+\]?',
    r'\b95%\s*CI\b',
    # Significance statements
    r'\bstatistically\s+significant\b',
    r'\bsignificant\s+(?:difference|effect|improvement|reduction|increase)\b',
    r'\bp\s*[<]\s*0\.05\b',
    r'\bp\s*[<]\s*0\.01\b',
    r'\bp\s*[<]\s*0\.001\b',
]

# Device patterns (medical devices, equipment)
DEVICE_PATTERNS = [
    r'\b(?:catheter|stent|pacemaker|defibrillator|implant|prosthesis|ventilator)s?\b',
    r'\b(?:MRI|CT|PET|ultrasound|X-ray|scanner|monitor)\b',
    r'\b(?:surgical|medical)\s+(?:device|instrument|equipment)\b',
    r'\b(?:insulin\s+pump|hearing\s+aid|wheelchair)\b',
]

# Compile patterns
STAT_SIG_REGEX = re.compile('|'.join(STAT_SIG_PATTERNS), re.IGNORECASE)
DEVICE_REGEX = re.compile('|'.join(DEVICE_PATTERNS), re.IGNORECASE)

# Entity label mapping from scispacy to our categories
LABEL_MAPPING = {
    # Scispacy labels that map to our categories
    'DISEASE': 'disease',
    'DISORDER': 'disease',
    'SYMPTOM': 'disease',
    'SYNDROME': 'disease',

    'CHEMICAL': 'drug',  # Most chemicals in medical text are drugs
    'DRUG': 'drug',
    'MEDICATION': 'drug',
    'PHARMACEUTICAL': 'drug',
}

def get_sentence_for_offset(text: str, start: int, end: int) -> Tuple[str, int, int]:
    """
    Get the full sentence containing the given character offsets.
    Returns (sentence_text, sentence_start_char, sentence_end_char)
    """
    nlp = get_nlp()
    doc = nlp(text)

    for sent in doc.sents:
        if sent.start_char <= start < sent.end_char or sent.start_char < end <= sent.end_char:
            return sent.text, sent.start_char, sent.end_char

    # Fallback: return a window around the mention
    window_start = max(0, start - 100)
    window_end = min(len(text), end + 100)
    return text[window_start:window_end], window_start, window_end

def extract_statistical_significance(text: str) -> List[Dict]:
    """Extract statistical significance mentions using regex patterns."""
    mentions = []
    for match in STAT_SIG_REGEX.finditer(text):
        mention_text = match.group(0)
        start_char = match.start()
        end_char = match.end()

        # Get context sentence
        context, _, _ = get_sentence_for_offset(text, start_char, end_char)

        mentions.append({
            'mention_text': mention_text,
            'label': 'statistical_significance',
            'start_char': start_char,
            'end_char': end_char,
            'context': context,
            'label_confidence': 0.9,  # High confidence for regex matches
        })

    return mentions

def extract_devices(text: str) -> List[Dict]:
    """Extract medical device mentions using regex patterns."""
    mentions = []
    for match in DEVICE_REGEX.finditer(text):
        mention_text = match.group(0)
        start_char = match.start()
        end_char = match.end()

        # Get context sentence
        context, _, _ = get_sentence_for_offset(text, start_char, end_char)

        mentions.append({
            'mention_text': mention_text,
            'label': 'device',
            'start_char': start_char,
            'end_char': end_char,
            'context': context,
            'label_confidence': 0.85,  # Good confidence for regex matches
        })

    return mentions

def extract_entities_from_text(
    text: str,
    section_name: Optional[str] = None
) -> List[Dict]:
    """
    Extract all entities from text using scispacy NER model + custom patterns.

    Args:
        text: The text to extract entities from
        section_name: Optional section name where text was found

    Returns:
        List of entity dictionaries with keys:
            - mention_text: the entity text
            - label: one of ['disease', 'drug', 'device', 'statistical_significance']
            - start_char: start character offset
            - end_char: end character offset
            - context: full sentence containing the mention
            - label_confidence: confidence score
            - section_name: section where found (if provided)
    """
    nlp = get_nlp()
    doc = nlp(text)

    mentions = []

    # Extract scispacy entities
    for ent in doc.ents:
        # Map scispacy label to our categories
        our_label = LABEL_MAPPING.get(ent.label_)

        if our_label:  # Only keep entities that map to our 4 categories
            # Get context sentence
            context, _, _ = get_sentence_for_offset(text, ent.start_char, ent.end_char)

            mention = {
                'mention_text': ent.text,
                'label': our_label,
                'start_char': ent.start_char,
                'end_char': ent.end_char,
                'context': context,
                'label_confidence': 0.8,  # Default confidence for NER
            }

            if section_name:
                mention['section_name'] = section_name

            mentions.append(mention)

    # Extract statistical significance patterns
    stat_mentions = extract_statistical_significance(text)
    for mention in stat_mentions:
        if section_name:
            mention['section_name'] = section_name
        mentions.append(mention)

    # Extract device patterns
    device_mentions = extract_devices(text)
    for mention in device_mentions:
        if section_name:
            mention['section_name'] = section_name
        mentions.append(mention)

    return mentions

def extract_entities_from_chunks(
    chunks: List[Tuple[str, str]]
) -> List[Dict]:
    """
    Extract entities from a list of (chunk_text, section_name) tuples.

    Args:
        chunks: List of (chunk_text, section_name) tuples

    Returns:
        List of entity dictionaries (same format as extract_entities_from_text)
    """
    all_mentions = []

    for chunk_text, section_name in chunks:
        # Remove source prefix if present (e.g., "arXiv:2301.12345\n\n")
        text_parts = chunk_text.split('\n\n', 1)
        if len(text_parts) > 1:
            actual_text = text_parts[1]
            offset = len(text_parts[0]) + 2  # +2 for '\n\n'
        else:
            actual_text = chunk_text
            offset = 0

        # Remove section name prefix if present (e.g., "[Introduction]\n")
        if actual_text.startswith('[') and '\n' in actual_text:
            first_line_end = actual_text.index('\n')
            if ']' in actual_text[:first_line_end]:
                actual_text = actual_text[first_line_end+1:]
                offset += first_line_end + 1

        mentions = extract_entities_from_text(actual_text, section_name)

        # Adjust character offsets to account for removed prefixes
        for mention in mentions:
            mention['start_char'] += offset
            mention['end_char'] += offset

        all_mentions.extend(mentions)

    return all_mentions

def embed_mentions(mentions: List[Dict]) -> List[List[float]]:
    """
    Generate embeddings for mention texts.

    Args:
        mentions: List of mention dictionaries

    Returns:
        List of 384-dimensional embeddings
    """
    if not mentions:
        return []

    embedder = get_embedder()
    mention_texts = [m['mention_text'] for m in mentions]
    embeddings = embedder.encode(mention_texts, normalize_embeddings=True, batch_size=32)
    return embeddings.tolist()

def deduplicate_mentions(mentions: List[Dict]) -> List[Dict]:
    """
    Remove duplicate mentions based on (mention_text, label, section_name).
    Keeps the first occurrence of each unique mention.
    """
    seen = set()
    unique_mentions = []

    for mention in mentions:
        key = (
            mention['mention_text'].lower().strip(),
            mention['label'],
            mention.get('section_name', '')
        )

        if key not in seen:
            seen.add(key)
            unique_mentions.append(mention)

    return unique_mentions
