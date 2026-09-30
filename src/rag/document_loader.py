"""Document loader and parser for agricultural knowledge base documents.

Parses Markdown files with YAML frontmatter, text files, and PDFs, extracting
structured provenance metadata and cleaned text content.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


class AgriculturalDocument:
    """Represents a curated agricultural advisory document with full provenance metadata."""

    def __init__(
        self,
        doc_id: str,
        title: str,
        content: str,
        crop: str,
        disease: str,
        disease_class: str,
        organization: str,
        author: str,
        publication_date: Optional[str],
        source_url: str,
        region: str,
        license: str,
        language: str = "en",
        file_path: Optional[Path] = None,
        raw_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.doc_id = doc_id
        self.title = title
        self.content = content
        self.crop = crop
        self.disease = disease
        self.disease_class = disease_class
        self.organization = organization
        self.author = author
        self.publication_date = publication_date
        self.source_url = source_url
        self.region = region
        self.license = license
        self.language = language
        self.file_path = file_path
        self.raw_metadata = raw_metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "crop": self.crop,
            "disease": self.disease,
            "disease_class": self.disease_class,
            "organization": self.organization,
            "author": self.author,
            "publication_date": self.publication_date,
            "source_url": self.source_url,
            "region": self.region,
            "license": self.license,
            "language": self.language,
            "content_length": len(self.content),
            "file_path": str(self.file_path) if self.file_path else None,
        }

    def __repr__(self) -> str:
        return f"<AgriculturalDocument doc_id='{self.doc_id}' title='{self.title[:35]}...' crop='{self.crop}'>"


def clean_text(text: str) -> str:
    """Normalize whitespace and strip unwanted formatting artifacts."""
    # Replace carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple consecutive blank lines to double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing and leading whitespace
    return text.strip()


def parse_markdown_with_frontmatter(file_path: Path) -> AgriculturalDocument:
    """Parse a Markdown file containing YAML frontmatter.

    Args:
        file_path: Absolute path to the Markdown file.

    Returns:
        AgriculturalDocument instance with extracted metadata and body content.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    metadata = {}
    content = text

    # Check for YAML frontmatter between --- markers
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            raw_yaml = parts[1]
            content = parts[2]
            try:
                metadata = yaml.safe_load(raw_yaml) or {}
            except yaml.YAMLError as e:
                logger.warning(f"Error parsing YAML frontmatter in {file_path.name}: {e}")

    content = clean_text(content)

    doc_id = metadata.get("doc_id", file_path.stem.upper())
    title = metadata.get("title", file_path.stem.replace("_", " ").title())
    crop = metadata.get("crop", "Unknown")
    disease = metadata.get("disease", "General")
    disease_class = metadata.get("disease_class", "Unknown")
    organization = metadata.get("organization", "KP Department of Agriculture")
    author = metadata.get("author", "KP Agricultural Extension")
    publication_date = str(metadata.get("publication_date", "Undated"))
    source_url = metadata.get("source_url", "https://agrikp.gov.pk")
    region = metadata.get("region", "Khyber Pakhtunkhwa, Pakistan")
    license_info = metadata.get("license", "Public Domain")
    language = metadata.get("language", "en")

    return AgriculturalDocument(
        doc_id=doc_id,
        title=title,
        content=content,
        crop=crop,
        disease=disease,
        disease_class=disease_class,
        organization=organization,
        author=author,
        publication_date=publication_date,
        source_url=source_url,
        region=region,
        license=license_info,
        language=language,
        file_path=file_path,
        raw_metadata=metadata,
    )


def load_all_documents(
    documents_dir: Optional[Path] = None,
) -> List[AgriculturalDocument]:
    """Load and parse all curated agricultural documents from directory.

    Args:
        documents_dir: Directory containing documents. Defaults to config.KB_DOCUMENTS_DIR.

    Returns:
        List of AgriculturalDocument objects.
    """
    doc_dir = documents_dir or config.KB_DOCUMENTS_DIR
    if not doc_dir.exists():
        raise FileNotFoundError(f"Knowledge base documents directory not found: {doc_dir}")

    documents = []
    # Search for markdown files
    for file_path in sorted(doc_dir.glob("*.md")):
        if file_path.name.startswith("."):
            continue
        try:
            doc = parse_markdown_with_frontmatter(file_path)
            documents.append(doc)
            logger.info(f"Loaded document [{doc.doc_id}] - {doc.title} ({doc.crop})")
        except Exception as e:
            logger.error(f"Failed to parse document {file_path.name}: {e}")

    logger.info(f"Total documents successfully loaded: {len(documents)}")
    return documents
