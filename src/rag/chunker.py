"""Section-aware agricultural text chunker preserving provenance metadata.

Divides advisory documents into semantically coherent passages, tracking section categories,
parent document provenance, and contextual headers for high-fidelity dense retrieval.
"""

import re
from typing import List, Dict, Any, Optional
from src import config
from src.utils import get_logger
from src.rag.document_loader import AgriculturalDocument

logger = get_logger(__name__)


def infer_section_category(header_title: str) -> str:
    """Infer the agricultural information category from section header text."""
    lower = header_title.lower()
    if any(k in lower for k in ["symptom", "sign", "identification", "damage", "appearance"]):
        return "symptoms"
    elif any(k in lower for k in ["pathogen", "causal", "biology", "fungus", "oomycete", "cycle"]):
        return "causes_pathogens"
    elif any(k in lower for k in ["cultural", "sanitation", "pruning", "rotation", "spacing", "tillage", "mulch"]):
        return "cultural_practices"
    elif any(k in lower for k in ["treatment", "control", "fungicide", "spray", "chemical"]):
        return "treatment_control"
    elif any(k in lower for k in ["prevent", "resistance", "resistant", "management", "idm", "integrated"]):
        return "prevention"
    elif any(k in lower for k in ["stage", "calendar", "timeline", "window", "growth", "bloom", "tasseling"]):
        return "crop_stage_guidance"
    elif any(k in lower for k in ["safety", "phi", "interval", "pre-harvest", "pollinator", "hazard", "ppe"]):
        return "safety_usage_notes"
    elif any(k in lower for k in ["soil", "nutrition", "fertility", "irrigation", "water"]):
        return "cultural_practices"
    return "general_guidance"


class AgriculturalChunk:
    """Represents a discrete semantic text chunk with preserved provenance metadata."""

    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        text: str,
        section_title: str,
        category: str,
        crop: str,
        disease: str,
        disease_class: str,
        title: str,
        organization: str,
        author: str,
        source_url: str,
        publication_date: Optional[str],
        license: str,
        region: str,
        chunk_index: int,
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.text = text
        self.section_title = section_title
        self.category = category
        self.crop = crop
        self.disease = disease
        self.disease_class = disease_class
        self.title = title
        self.organization = organization
        self.author = author
        self.source_url = source_url
        self.publication_date = publication_date
        self.license = license
        self.region = region
        self.chunk_index = chunk_index

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization and vector store metadata."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "section_title": self.section_title,
            "category": self.category,
            "crop": self.crop,
            "disease": self.disease,
            "disease_class": self.disease_class,
            "title": self.title,
            "organization": self.organization,
            "author": self.author,
            "source_url": self.source_url,
            "publication_date": self.publication_date,
            "license": self.license,
            "region": self.region,
            "chunk_index": self.chunk_index,
            "char_count": len(self.text),
            "word_count": len(self.text.split()),
        }

    def __repr__(self) -> str:
        return f"<AgriculturalChunk id='{self.chunk_id}' crop='{self.crop}' cat='{self.category}'>"


def split_section_into_passages(
    text: str,
    max_chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> List[str]:
    """Split section body text into overlapping passages preserving sentence boundaries.

    Args:
        text: Raw section body string.
        max_chunk_size: Target maximum characters per chunk (default: 500).
        overlap: Character overlap between consecutive chunks (default: 50).

    Returns:
        List of chunk text strings.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    curr_chunk = ""

    for para in paragraphs:
        # If adding paragraph fits within max_chunk_size
        if len(curr_chunk) + len(para) + 2 <= max_chunk_size:
            if curr_chunk:
                curr_chunk += "\n\n" + para
            else:
                curr_chunk = para
        else:
            if curr_chunk:
                chunks.append(curr_chunk.strip())
                # Start new chunk with overlap from end of previous chunk if possible
                overlap_text = curr_chunk[-overlap:] if len(curr_chunk) > overlap else ""
                # Find space in overlap to avoid word truncation
                space_idx = overlap_text.find(" ")
                if space_idx != -1:
                    overlap_text = overlap_text[space_idx + 1:]
                curr_chunk = (overlap_text + " " + para).strip() if overlap_text else para
            else:
                # Single paragraph exceeds max_chunk_size; split by sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sub_chunk = ""
                for sent in sentences:
                    if len(sub_chunk) + len(sent) + 1 <= max_chunk_size:
                        sub_chunk = (sub_chunk + " " + sent).strip()
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk.strip())
                        sub_chunk = sent
                if sub_chunk:
                    curr_chunk = sub_chunk

    if curr_chunk:
        chunks.append(curr_chunk.strip())

    return [c for c in chunks if len(c) > 30]


def chunk_document(
    doc: AgriculturalDocument,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[AgriculturalChunk]:
    """Segment an AgriculturalDocument into semantic chunks preserving metadata and section context.

    Args:
        doc: AgriculturalDocument instance.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between chunks.

    Returns:
        List of AgriculturalChunk objects.
    """
    raw_content = doc.content
    # Split content by markdown section headers (## Section)
    section_pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    matches = list(section_pattern.finditer(raw_content))

    sections = []
    if not matches:
        sections.append(("General Overview", raw_content))
    else:
        # Intro before first section if present
        first_start = matches[0].start()
        if first_start > 0:
            intro_text = raw_content[:first_start].strip()
            if intro_text:
                sections.append(("Introduction", intro_text))

        for i, match in enumerate(matches):
            section_title = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_content)
            section_body = raw_content[start:end].strip()
            sections.append((section_title, section_body))

    chunks = []
    chunk_idx = 0

    for section_title, section_body in sections:
        category = infer_section_category(section_title)
        passages = split_section_into_passages(
            section_body, max_chunk_size=chunk_size, overlap=chunk_overlap
        )

        for p_idx, passage in enumerate(passages):
            chunk_id = f"{doc.doc_id}-C{chunk_idx:03d}"
            # Embed contextual header into passage text for dense semantic retrieval
            contextualized_text = (
                f"[{doc.crop} | {doc.disease} | Section: {section_title}]\n{passage}"
            )

            chunk = AgriculturalChunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                text=contextualized_text,
                section_title=section_title,
                category=category,
                crop=doc.crop,
                disease=doc.disease,
                disease_class=doc.disease_class,
                title=doc.title,
                organization=doc.organization,
                author=doc.author,
                source_url=doc.source_url,
                publication_date=doc.publication_date,
                license=doc.license,
                region=doc.region,
                chunk_index=chunk_idx,
            )
            chunks.append(chunk)
            chunk_idx += 1

    logger.info(f"Segmented [{doc.doc_id}] ({doc.title}) into {len(chunks)} chunks.")
    return chunks


def chunk_all_documents(
    documents: List[AgriculturalDocument],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[AgriculturalChunk]:
    """Chunk all provided documents into a unified list of AgriculturalChunks."""
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(doc_chunks)
    logger.info(f"Total chunks created across all {len(documents)} documents: {len(all_chunks)}")
    return all_chunks
