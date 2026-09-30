"""Multilingual translation and agricultural localization engine (English <-> Urdu).

Supports:
1. Citation and Latin binomial entity protection during translation.
2. Glossary-driven terminology substitution.
3. Disk-backed and memory-backed translation caching for latency optimization and offline reproducibility.
4. Right-to-Left (RTL) formatting and bidirectional marker handling.
5. Bidirectional translation (English -> Urdu and Urdu -> English query translation).
6. Deterministic template matching for safety guardrails and standard notices.
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import hashlib

from src import config
from src.utils import get_logger
from src.rag.glossary import get_glossary

logger = get_logger(__name__)

CACHE_FILE = config.KNOWLEDGE_BASE_DIR / "cache" / "translation_cache.json"

# Patterns to protect from machine translation
PROTECTED_PATTERNS = [
    r"\[KP-AGRI-[A-Z0-9\-]+\]",            # Citation tokens e.g. [KP-AGRI-POT-002]
    r"https?://[^\s\)]+",                   # URLs
    r"\b[A-Z][a-z]+ [a-z]+\b",             # Latin binomials e.g. Phytophthora infestans
    r"\bFRAC \d+\b",                        # FRAC resistance codes e.g. FRAC 4
    r"\b[A-Z0-9\-]+___[A-Za-z0-9_\-]+\b",  # Internal disease class labels
]

# Standard agronomic template translations (deterministic offline safety fallback)
STANDARD_TEMPLATES_UR = {
    "The requested agricultural guidance or specific treatment/dosage recommendation is not verified in the curated KP agricultural extension knowledge base. To prevent hazardous crop management, please consult your local KP Agriculture Extension Officer or certified agronomist.":
        "درخواست کردہ زرعی رہنمائی یا مخصوص علاج/خوراک کی سفارش خیبر پختونخوا کے تصدیق شدہ توسیعی علمی ذخیرے میں موجود نہیں ہے۔ فصلوں کے غیر محفوظ انتظام سے بچاؤ کے لیے براہ کرم اپنے مقامی محکمہ زراعت توسیع کے افسر سے رجوع کریں۔",
    "Guidance is strictly grounded in official Khyber Pakhtunkhwa extension and research publications. Pesticide availability and regulations may vary by district; verify product registrations with local extension officers.":
        "یہ رہنمائی مکمل طور پر خیبر پختونخوا کے زرعی توسیعی اور تحقیقی اداروں کی سرکاری اشاعتوں پر مبنی ہے۔ کیڑے مار ادویات کی دستیابی اور ضوابط مختلف اضلاع میں مختلف ہو سکتے ہیں؛ اپنے مقامی فیلڈ آفیسر سے تصدیق کریں۔",
    "Refer to primary cited bulletin for detailed foliar symptoms.":
        "تفصیلی علامات کے لیے بنیادی حوالہ شدہ زرعی بلیٹن سے رجوع کریں۔",
    "Consult local KP Agriculture Extension officer for verified registered treatments in your tehsil.":
        "اپنی تحصیل میں تصدیق شدہ رجسٹرڈ ادویات اور علاج کے لیے مقامی زرعی توسیعی افسر سے مشورہ کریں۔",
    "Implement crop rotation and use certified disease-free seed.":
        "فصلوں کا ہیر پھیر (کراپ روٹیشن) اپنائیں اور تصدیق شدہ بیماری سے پاک بیج استعمال کریں۔",
    "Monitor foliage during humid vegetative and reproductive stages.":
        "نم آلود بڑھوتری اور پیداواری مراحل کے دوران پتوں کی مسلسل نگرانی کریں۔",
    "No verified source documents were retrieved from the curated knowledge base.":
        "تصدیق شدہ علمی ذخیرے سے کوئی متعلقہ معلوماتی دستاویز حاصل نہیں ہو سکی۔",
    "Requested information is unavailable in curated sources.":
        "درخواست کردہ معلومات تصدیق شدہ زرعی ذرائع میں دستیاب نہیں ہیں۔",
}


class AgriculturalTranslator:
    """Handles high-fidelity English-to-Urdu and Urdu-to-English translation with domain preservation."""

    def __init__(self, cache_path: Optional[Path] = None, enable_online: bool = True):
        """Initialize translator with glossary, persistent cache, and online toggle."""
        self.glossary = get_glossary()
        self.cache_path = cache_path or CACHE_FILE
        self.enable_online = enable_online and (os.getenv("OFFLINE_TRANSLATION", "0") != "1")
        self.cache: Dict[str, str] = {}
        self._load_cache()
        self._translator_ur = None
        self._translator_en = None

    def _get_translator_to_ur(self):
        """Lazy load GoogleTranslator from deep_translator for English -> Urdu."""
        if not self.enable_online:
            return None
        if self._translator_ur is None:
            try:
                from deep_translator import GoogleTranslator
                self._translator_ur = GoogleTranslator(source="en", target="ur")
            except Exception as e:
                logger.warning(f"Could not initialize GoogleTranslator (en->ur): {e}. Fallback enabled.")
                self._translator_ur = None
        return self._translator_ur

    def _get_translator_to_en(self):
        """Lazy load GoogleTranslator from deep_translator for Urdu -> English."""
        if not self.enable_online:
            return None
        if self._translator_en is None:
            try:
                from deep_translator import GoogleTranslator
                self._translator_en = GoogleTranslator(source="ur", target="en")
            except Exception as e:
                logger.warning(f"Could not initialize GoogleTranslator (ur->en): {e}. Fallback enabled.")
                self._translator_en = None
        return self._translator_en

    def _load_cache(self) -> None:
        """Load persistent translation cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached translations from {self.cache_path}")
            except Exception as e:
                logger.warning(f"Failed to read translation cache: {e}")
                self.cache = {}
        else:
            self.cache = {}

    def _save_cache(self) -> None:
        """Save persistent translation cache to disk."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save translation cache: {e}")

    def _protect_entities(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Replace citations and Latin binomials with placeholder tokens."""
        placeholder_map = {}
        token_counter = 0

        # Protect bracketed citations and URLs first
        for pat in PROTECTED_PATTERNS:
            for match in re.finditer(pat, text):
                matched_str = match.group(0)
                placeholder = f"__ENTITY_TOKEN_{token_counter}__"
                placeholder_map[placeholder] = matched_str
                text = text.replace(matched_str, placeholder)
                token_counter += 1

        return text, placeholder_map

    def _restore_entities(self, text: str, placeholder_map: Dict[str, str]) -> str:
        """Restore protected entities from placeholder tokens."""
        for placeholder, original in placeholder_map.items():
            text = text.replace(placeholder, original)
        return text

    def translate_to_urdu(self, text: str) -> str:
        """Translate English text to fluent Urdu while preserving terminology and citations.

        Args:
            text: English text string.

        Returns:
            Urdu localized text string.
        """
        if not text or not text.strip():
            return text

        # If already Urdu script and contains no English prose words, return as-is
        if re.search(r"[\u0600-\u06FF]", text) and not re.search(r"[a-zA-Z]{4,}", text):
            return text

        raw_hash = hashlib.md5(text.strip().encode("utf-8")).hexdigest()
        cache_key = f"en2ur_{raw_hash}"

        if cache_key in self.cache:
            return self.cache[cache_key]
        if raw_hash in self.cache:
            return self.cache[raw_hash]

        # Check standard agronomic templates
        clean_key = text.strip()
        if clean_key in STANDARD_TEMPLATES_UR:
            res = STANDARD_TEMPLATES_UR[clean_key]
            self.cache[cache_key] = res
            return res

        # Check out-of-scope refusal pattern
        if "outside the supported 11-class diagnostic domain" in clean_key:
            m = re.search(r"The crop '([^']+)'", clean_key)
            crop_name = m.group(1) if m else "غیر معاون فصل"
            res = (
                f"فصل '{crop_name}' کے پی سمارٹ کراپ ڈاکٹر کے 11 معاون تشخیصی زمروں (سیب، مکئی، آلو) سے باہر ہے۔ "
                f"{crop_name} کے لیے تصدیق شدہ زرعی سفارشات فی الوقت اس علمی ذخیرے میں دستیاب نہیں ہیں۔"
            )
            self.cache[cache_key] = res
            return res

        # 1. Protect special entities
        protected_text, entity_map = self._protect_entities(text)

        # 2. Translate using deep-translator or fallback
        translator = self._get_translator_to_ur()
        translated_text = ""
        if translator:
            try:
                translated_text = translator.translate(protected_text)
            except Exception as e:
                logger.debug(f"Translation API error (en->ur): {e}. Using glossary substitution.")

        if not translated_text:
            translated_text = self.glossary.apply_glossary_replacements(protected_text)

        # 3. Restore protected entities
        restored_text = self._restore_entities(translated_text, entity_map)

        # 4. Enforce standard agricultural glossary terms
        final_text = self.glossary.apply_glossary_replacements(restored_text)

        # 5. Cache result
        self.cache[cache_key] = final_text
        self._save_cache()

        return final_text

    def translate_to_english(self, text: str) -> str:
        """Translate Urdu query or text to English for knowledge base retrieval.

        Args:
            text: Urdu text query string.

        Returns:
            English translated text string suitable for semantic vector retrieval.
        """
        if not text or not text.strip():
            return text

        # If text contains no Urdu script, return directly
        if not re.search(r"[\u0600-\u06FF]", text):
            return text

        raw_hash = hashlib.md5(text.strip().encode("utf-8")).hexdigest()
        cache_key = f"ur2en_{raw_hash}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        # 1. Protect special entities
        protected_text, entity_map = self._protect_entities(text)

        # 2. Try online translator
        translator = self._get_translator_to_en()
        translated_text = ""
        if translator:
            try:
                translated_text = translator.translate(protected_text)
            except Exception as e:
                logger.debug(f"Translation API error (ur->en): {e}. Using glossary mapping.")

        if not translated_text:
            # Fallback: Extract English terms mapped from Urdu glossary
            matched_en_terms = []
            for ur_term, en_term in self.glossary.ur_to_en.items():
                if ur_term in text:
                    matched_en_terms.append(en_term)
            translated_text = " ".join(matched_en_terms) if matched_en_terms else text

        restored_text = self._restore_entities(translated_text, entity_map)
        self.cache[cache_key] = restored_text
        self._save_cache()

        return restored_text

    @staticmethod
    def wrap_rtl(text: str) -> str:
        """Wrap Urdu text with Right-to-Left (RTL) unicode directional isolation markers."""
        # Unicode U+2067 (RIGHT-TO-LEFT ISOLATE) and U+2069 (POP DIRECTIONAL ISOLATE)
        return f"\u2067{text}\u2069"
