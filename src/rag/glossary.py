"""Bilingual agricultural terminology glossary manager.

Provides term mapping, standardized English-to-Urdu translations, and preservation
of scientific binomials, chemical names, and citation tokens.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

DEFAULT_GLOSSARY_PATH = config.KB_METADATA_DIR / "agricultural_glossary_ur.json"


class AgriculturalGlossary:
    """Manages bilingual agricultural terms and domain-specific terminology mapping."""

    def __init__(self, glossary_path: Optional[Path] = None):
        """Load glossary definitions from JSON catalog."""
        self.glossary_path = glossary_path or DEFAULT_GLOSSARY_PATH
        self.en_to_ur: Dict[str, str] = {}
        self.ur_to_en: Dict[str, str] = {}
        self.term_metadata: Dict[str, Dict[str, Any]] = {}
        self.terms: List[Dict[str, Any]] = []
        self._load_glossary()

    def _load_glossary(self) -> None:
        """Load and index terms from glossary file."""
        if not self.glossary_path.exists():
            logger.warning(f"Glossary file not found at: {self.glossary_path}")
            return

        with open(self.glossary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.terms = data.get("terms", [])
        for entry in self.terms:
            en_term = entry.get("english", "").strip()
            ur_term = entry.get("urdu", "").strip()
            if en_term and ur_term:
                self.en_to_ur[en_term.lower()] = ur_term
                self.ur_to_en[ur_term] = en_term
                self.term_metadata[en_term.lower()] = entry

        logger.info(f"Loaded {len(self.en_to_ur)} bilingual agricultural terms into glossary.")

    def get_term(self, english_term: str) -> Optional[Dict[str, Any]]:
        """Retrieve full metadata entry for an English term."""
        return self.term_metadata.get(english_term.strip().lower())

    def get_urdu_term(self, english_term: str) -> Optional[str]:
        """Lookup standard Urdu translation for an agricultural term."""
        return self.en_to_ur.get(english_term.strip().lower())

    def get_english_term(self, urdu_term: str) -> Optional[str]:
        """Lookup English counterpart for an Urdu term."""
        return self.ur_to_en.get(urdu_term.strip())

    def apply_glossary_replacements(self, text: str) -> str:
        """Enforce standard agricultural terminology onto an Urdu translation text."""
        # Sort keys by length descending to replace longer composite phrases first
        sorted_keys = sorted(self.en_to_ur.keys(), key=len, reverse=True)
        for key in sorted_keys:
            ur_replacement = self.en_to_ur[key]
            # Word boundary regex match
            pattern = re.compile(r"\b" + re.escape(key) + r"\b", re.IGNORECASE)
            text = pattern.sub(ur_replacement, text)
        return text


# Global singleton instance
_GLOBAL_GLOSSARY: Optional[AgriculturalGlossary] = None


def get_glossary() -> AgriculturalGlossary:
    """Return singleton AgriculturalGlossary instance."""
    global _GLOBAL_GLOSSARY
    if _GLOBAL_GLOSSARY is None:
        _GLOBAL_GLOSSARY = AgriculturalGlossary()
    return _GLOBAL_GLOSSARY
