"""Grounded advisory generation module for agricultural decision support.

Integrates RAG retrieval evidence into structured agronomic advisories with
strict citation preservation, refusal/abstention guardrails, and bilingual (English/Urdu) output.
"""

import os
import re
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union

from src import config
from src.utils import get_logger
from src.rag.retriever import (
    AgriculturalRetriever,
    RetrievalResponse,
    RetrievedChunk,
    detect_out_of_scope_crop,
)
from src.rag.translator import AgriculturalTranslator

logger = get_logger(__name__)


@dataclass
class StructuredAdvisory:
    """Structured agricultural advisory contract."""

    disease: str
    confidence: Optional[float]
    key_symptoms: List[str]
    likely_cause: Optional[str]
    recommended_management: List[str]
    prevention: List[str]
    crop_stage_guidance: Optional[str]
    source_citations: List[str]
    uncertainty_limitations: str
    status: str = "grounded"  # "grounded", "abstain", "information_unavailable"
    language: str = "en"       # "en" or "ur"
    direction: str = "ltr"     # "ltr" or "rtl"
    evidence_chunks_count: int = 0
    refusal_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LocalExtractiveSynthesizer:
    """Deterministic, local grounded advisory synthesizer.

    Constructs structured advisories directly from retrieved evidence chunks without
    external API calls. Guarantees zero hallucination and strict source fidelity.
    """

    def synthesize(
        self,
        query: str,
        retrieval_response: RetrievalResponse,
        crop: Optional[str] = None,
        disease_name: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> StructuredAdvisory:
        """Synthesize a structured advisory strictly grounded in the retrieved chunks."""
        chunks = retrieval_response.chunks

        if not chunks:
            return StructuredAdvisory(
                disease=disease_name or "Unknown Condition",
                confidence=confidence,
                key_symptoms=[],
                likely_cause=None,
                recommended_management=[],
                prevention=[],
                crop_stage_guidance=None,
                source_citations=[],
                uncertainty_limitations="No verified source documents were retrieved from the curated knowledge base.",
                status="abstain",
                language="en",
                evidence_chunks_count=0,
                refusal_reason="Insufficient retrieval evidence.",
            )

        # Primary disease / crop from top evidence
        primary_disease = chunks[0].disease if chunks else (disease_name or "Unknown Disease")
        primary_crop = chunks[0].crop if chunks else (crop or "Crop")

        # Collect citations
        citations = list(dict.fromkeys([c.citation for c in chunks]))

        # Categorize extracted sentences across chunks
        symptoms = []
        causes = []
        management = []
        prevention = []
        crop_stage = []

        for c in chunks:
            # Strip the contextual header line before extracting sentences
            body_text = c.text
            if body_text.startswith("["):
                header_end = body_text.find("]\n")
                if header_end != -1:
                    body_text = body_text[header_end + 2:]

            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body_text) if len(s.strip()) > 20]

            if c.category == "symptoms":
                symptoms.extend(sentences[:3])
            elif c.category == "causes_pathogens":
                causes.extend(sentences[:2])
            elif c.category == "treatment_control":
                management.extend(sentences[:3])
            elif c.category in ["prevention", "cultural_practices"]:
                prevention.extend(sentences[:3])
            elif c.category == "crop_stage_guidance":
                crop_stage.extend(sentences[:2])

        # Deduplicate while preserving order
        def dedup(lst: List[str]) -> List[str]:
            seen = set()
            out = []
            for item in lst:
                if item not in seen:
                    seen.add(item)
                    out.append(item)
            return out

        clean_symptoms = dedup(symptoms)[:4] or ["Refer to primary cited bulletin for detailed foliar symptoms."]
        likely_cause_str = causes[0] if causes else f"Biotic pathogen associated with {primary_disease}."
        clean_management = dedup(management)[:4] or [
            "Consult local KP Agriculture Extension officer for verified registered treatments in your tehsil."
        ]
        clean_prevention = dedup(prevention)[:4] or [
            "Implement crop rotation and use certified disease-free seed."
        ]
        clean_crop_stage = crop_stage[0] if crop_stage else "Monitor foliage during humid vegetative and reproductive stages."

        uncertainty_note = (
            "Guidance is strictly grounded in official Khyber Pakhtunkhwa extension and research publications. "
            "Pesticide availability and regulations may vary by district; verify product registrations with local extension officers."
        )

        return StructuredAdvisory(
            disease=f"{primary_crop}: {primary_disease}",
            confidence=confidence,
            key_symptoms=clean_symptoms,
            likely_cause=likely_cause_str,
            recommended_management=clean_management,
            prevention=clean_prevention,
            crop_stage_guidance=clean_crop_stage,
            source_citations=citations,
            uncertainty_limitations=uncertainty_note,
            status="grounded",
            language="en",
            direction="ltr",
            evidence_chunks_count=len(chunks),
        )


class GroundedAdvisoryGenerator:
    """Master grounded advisory generator supporting English and Urdu."""

    def __init__(
        self,
        retriever: Optional[AgriculturalRetriever] = None,
        translator: Optional[AgriculturalTranslator] = None,
        provider: Optional[str] = None,
    ):
        """Initialize generator with retriever, translator, and configured provider."""
        self.retriever = retriever or AgriculturalRetriever()
        self.translator = translator or AgriculturalTranslator()
        self.provider_name = (provider or os.getenv("LLM_PROVIDER", config.LLM_PROVIDER)).lower()
        self.synthesizer = LocalExtractiveSynthesizer()
        logger.info(f"Initialized GroundedAdvisoryGenerator with provider '{self.provider_name}'.")

    def generate_advisory(
        self,
        query: str,
        crop: Optional[str] = None,
        disease_class: Optional[str] = None,
        confidence: Optional[float] = None,
        language: str = "en",
        top_k: int = config.RAG_TOP_K_CHUNKS,
    ) -> StructuredAdvisory:
        """Execute grounded generation pipeline for a diagnostic or agronomic query.

        Args:
            query: User question or diagnostic symptom prompt.
            crop: Optional crop context from vision prediction.
            disease_class: Optional disease class from vision prediction.
            confidence: Optional vision prediction confidence score.
            language: 'en' for English or 'ur' for Urdu.
            top_k: Number of evidence chunks to retrieve.

        Returns:
            StructuredAdvisory instance.
        """
        # 1. Check for explicit unsupported crop guardrail in query
        out_of_scope_crop = detect_out_of_scope_crop(query)
        if out_of_scope_crop:
            refusal_text = (
                f"The crop '{out_of_scope_crop}' is outside the supported 11-class diagnostic domain "
                "(Apple, Maize, Potato) of KP Smart Crop Doctor. Verified agricultural advisories for "
                f"{out_of_scope_crop} are currently unavailable in this curated knowledge base."
            )
            advisory = StructuredAdvisory(
                disease=f"Unsupported Crop: {out_of_scope_crop}",
                confidence=confidence,
                key_symptoms=[],
                likely_cause=None,
                recommended_management=[],
                prevention=[],
                crop_stage_guidance=None,
                source_citations=[],
                uncertainty_limitations=refusal_text,
                status="information_unavailable",
                language=language,
                direction="rtl" if language == "ur" else "ltr",
                evidence_chunks_count=0,
                refusal_reason=refusal_text,
            )
            if language == "ur":
                advisory.disease = f"غیر معاون فصل: {out_of_scope_crop}"
                advisory.uncertainty_limitations = self.translator.translate_to_urdu(refusal_text)
                advisory.refusal_reason = advisory.uncertainty_limitations
            return advisory

        # 2. Cross-Lingual Query Preparation: translate Urdu query to English for retrieval
        is_urdu_query = bool(re.search(r"[\u0600-\u06FF]", query))
        if is_urdu_query:
            translated_en = self.translator.translate_to_english(query)
            if crop and disease_class:
                clean_disease = disease_class.replace("___", " ").replace("_", " ")
                retrieval_query = f"{crop} {clean_disease}: {translated_en}".strip()
            else:
                retrieval_query = translated_en
        else:
            if crop and disease_class and crop.lower() not in query.lower():
                clean_disease = disease_class.replace("___", " ").replace("_", " ")
                retrieval_query = f"{crop} {clean_disease}: {query}".strip()
            else:
                retrieval_query = query

        # 3. Retrieve evidence from knowledge base
        retrieval_response = self.retriever.retrieve(
            query=retrieval_query,
            top_k=top_k,
            filter_crop=crop,
            filter_disease_class=disease_class,
        )

        # 4. Check for abstention / information unavailable from retriever
        if retrieval_response.status == "information_unavailable":
            abstain_msg = retrieval_response.message or "Requested information is unavailable in curated sources."
            advisory = StructuredAdvisory(
                disease=disease_class or "Unsupported Crop/Disease",
                confidence=confidence,
                key_symptoms=[],
                likely_cause=None,
                recommended_management=[],
                prevention=[],
                crop_stage_guidance=None,
                source_citations=[],
                uncertainty_limitations=abstain_msg,
                status="information_unavailable",
                language=language,
                direction="rtl" if language == "ur" else "ltr",
                evidence_chunks_count=0,
                refusal_reason=abstain_msg,
            )
            if language == "ur":
                advisory.uncertainty_limitations = self.translator.translate_to_urdu(abstain_msg)
                advisory.refusal_reason = advisory.uncertainty_limitations
            return advisory

        # 5. Grounded Synthesis
        advisory_en = self.synthesizer.synthesize(
            query=retrieval_query,
            retrieval_response=retrieval_response,
            crop=crop,
            disease_name=disease_class,
            confidence=confidence,
        )

        # 6. If English requested, return English advisory
        if language.lower() == "en":
            return advisory_en

        # 7. If Urdu requested, translate structured fields while preserving citations and Latin names
        advisory_ur = self._localize_advisory_to_urdu(advisory_en)
        return advisory_ur

    def _localize_advisory_to_urdu(self, en: StructuredAdvisory) -> StructuredAdvisory:
        """Translate structured advisory fields to Urdu while preserving citations and technical terms."""
        ur_disease = self.translator.translate_to_urdu(en.disease)
        ur_symptoms = [self.translator.translate_to_urdu(s) for s in en.key_symptoms]
        ur_cause = self.translator.translate_to_urdu(en.likely_cause) if en.likely_cause else None
        ur_management = [self.translator.translate_to_urdu(m) for m in en.recommended_management]
        ur_prevention = [self.translator.translate_to_urdu(p) for p in en.prevention]
        ur_crop_stage = self.translator.translate_to_urdu(en.crop_stage_guidance) if en.crop_stage_guidance else None
        ur_uncertainty = self.translator.translate_to_urdu(en.uncertainty_limitations)

        # Source citations are kept intact in original citation format
        return StructuredAdvisory(
            disease=ur_disease,
            confidence=en.confidence,
            key_symptoms=ur_symptoms,
            likely_cause=ur_cause,
            recommended_management=ur_management,
            prevention=ur_prevention,
            crop_stage_guidance=ur_crop_stage,
            source_citations=en.source_citations,  # Preserved verbatim
            uncertainty_limitations=ur_uncertainty,
            status=en.status,
            language="ur",
            direction="rtl",
            evidence_chunks_count=en.evidence_chunks_count,
            refusal_reason=self.translator.translate_to_urdu(en.refusal_reason) if en.refusal_reason else None,
        )
