"""Unit and integration tests for grounded multilingual advisory generation.

Tests:
1. StructuredAdvisory data schema and serialization contract.
2. Grounded context construction from retrieved chunks.
3. Strict citation preservation across synthesis and translation.
4. Out-of-scope crop/disease guardrail abstention (information_unavailable).
5. Zero-retrieval / low-similarity abstention behavior.
6. Agricultural glossary term mapping and replacement.
7. Translator entity masking (citations, Latin binomials) and RTL wrapping.
8. English and Urdu structural consistency and field parity.
9. LocalExtractiveSynthesizer zero-hallucination guarantee.
"""

import pytest
from pathlib import Path
from dataclasses import asdict

from src import config
from src.rag.retriever import AgriculturalRetriever, RetrievalResponse, RetrievedChunk
from src.rag.glossary import AgriculturalGlossary, get_glossary
from src.rag.translator import AgriculturalTranslator
from src.rag.generator import (
    StructuredAdvisory,
    LocalExtractiveSynthesizer,
    GroundedAdvisoryGenerator,
)


@pytest.fixture(scope="module")
def retriever():
    """Instantiate shared retriever."""
    return AgriculturalRetriever(min_similarity_threshold=0.45)


@pytest.fixture(scope="module")
def translator():
    """Instantiate shared translator configured for deterministic testing."""
    return AgriculturalTranslator(enable_online=False)


@pytest.fixture(scope="module")
def generator(retriever, translator):
    """Instantiate shared advisory generator."""
    return GroundedAdvisoryGenerator(retriever=retriever, translator=translator)


@pytest.fixture
def mock_retrieval_response():
    """Create a controlled mock retrieval response for deterministic testing."""
    chunks = [
        RetrievedChunk(
            chunk_id="KP-AGRI-POT-002_chunk_001",
            doc_id="KP-AGRI-POT-002",
            title="Management of Potato Late Blight in Khyber Pakhtunkhwa",
            crop="Potato",
            disease="Late Blight",
            disease_class="Potato___Late_blight",
            section_title="Symptoms and Diagnostic Identification",
            category="symptoms",
            text="[Potato | Late Blight | symptoms]\nWater-soaked lesions appear on leaf tips and margins. White velvety fungal sporulation develops on undersides during high humidity.",
            organization="KP Department of Agriculture Extension",
            author="Plant Pathology Section",
            source_url="https://agrikp.gov.pk/potato-late-blight",
            publication_date="2023-09",
            similarity_score=0.88,
            citation="[KP-AGRI-POT-002] Potato Late Blight Management Bulletin, KP Agri Extension (2023)",
        ),
        RetrievedChunk(
            chunk_id="KP-AGRI-POT-002_chunk_002",
            doc_id="KP-AGRI-POT-002",
            title="Management of Potato Late Blight in Khyber Pakhtunkhwa",
            crop="Potato",
            disease="Late Blight",
            disease_class="Potato___Late_blight",
            section_title="Chemical and Cultural Treatment",
            category="treatment_control",
            text="[Potato | Late Blight | treatment_control]\nApply protective sprays of mancozeb before disease onset. Under severe pressure, apply metalaxyl or cymoxanil formulated sprays according to local agricultural extension schedules.",
            organization="KP Department of Agriculture Extension",
            author="Plant Pathology Section",
            source_url="https://agrikp.gov.pk/potato-late-blight",
            publication_date="2023-09",
            similarity_score=0.85,
            citation="[KP-AGRI-POT-002] Potato Late Blight Management Bulletin, KP Agri Extension (2023)",
        ),
        RetrievedChunk(
            chunk_id="KP-AGRI-POT-002_chunk_003",
            doc_id="KP-AGRI-POT-002",
            title="Management of Potato Late Blight in Khyber Pakhtunkhwa",
            crop="Potato",
            disease="Late Blight",
            disease_class="Potato___Late_blight",
            section_title="Preventive Measures",
            category="prevention",
            text="[Potato | Late Blight | prevention]\nUse certified disease-free seed tubers from Swat or Kalam seed farms. Destroy cull piles and eliminate volunteer potato plants.",
            organization="KP Department of Agriculture Extension",
            author="Plant Pathology Section",
            source_url="https://agrikp.gov.pk/potato-late-blight",
            publication_date="2023-09",
            similarity_score=0.82,
            citation="[KP-AGRI-POT-002] Potato Late Blight Management Bulletin, KP Agri Extension (2023)",
        ),
    ]
    return RetrievalResponse(
        query="What are symptoms and control for potato late blight?",
        chunks=chunks,
        status="success",
        applied_crop_filter="Potato",
        applied_disease_filter="Potato___Late_blight",
    )


# =============================================================================
# 1. Structured Advisory Schema Tests
# =============================================================================

def test_structured_advisory_contract():
    """Verify that StructuredAdvisory satisfies the expected field schema."""
    advisory = StructuredAdvisory(
        disease="Potato: Late Blight",
        confidence=0.96,
        key_symptoms=["Water-soaked lesions on leaf tips"],
        likely_cause="Phytophthora infestans",
        recommended_management=["Apply protective fungicide"],
        prevention=["Use certified seed tubers"],
        crop_stage_guidance="Monitor during tuber bulking",
        source_citations=["[KP-AGRI-POT-002] KP Agri Extension"],
        uncertainty_limitations="Verify local district registrations.",
        status="grounded",
        language="en",
        direction="ltr",
        evidence_chunks_count=3,
    )
    d = advisory.to_dict()
    assert d["disease"] == "Potato: Late Blight"
    assert d["confidence"] == 0.96
    assert isinstance(d["key_symptoms"], list)
    assert len(d["key_symptoms"]) == 1
    assert d["status"] == "grounded"
    assert d["language"] == "en"
    assert d["direction"] == "ltr"
    assert d["evidence_chunks_count"] == 3


# =============================================================================
# 2. LocalExtractiveSynthesizer & Zero-Hallucination
# =============================================================================

def test_extractive_synthesizer_grounding(mock_retrieval_response):
    """Verify synthesizer extracts content directly from retrieved chunks without hallucination."""
    synthesizer = LocalExtractiveSynthesizer()
    advisory = synthesizer.synthesize(
        query="symptoms and control for late blight",
        retrieval_response=mock_retrieval_response,
        crop="Potato",
        disease_name="Potato___Late_blight",
        confidence=0.95,
    )

    assert advisory.status == "grounded"
    assert advisory.evidence_chunks_count == 3
    assert len(advisory.key_symptoms) > 0
    assert len(advisory.recommended_management) > 0
    assert len(advisory.prevention) > 0

    # Verify that symptom text originates from mock chunk text
    assert any("Water-soaked lesions" in s for s in advisory.key_symptoms)
    # Verify citations preserved
    assert "[KP-AGRI-POT-002]" in advisory.source_citations[0]


def test_extractive_synthesizer_empty_retrieval():
    """Verify synthesizer abstains when retrieval response has no chunks."""
    synthesizer = LocalExtractiveSynthesizer()
    empty_resp = RetrievalResponse(
        query="unknown query",
        chunks=[],
        status="information_unavailable",
        message="Insufficient retrieval evidence.",
    )
    advisory = synthesizer.synthesize(
        query="unknown query",
        retrieval_response=empty_resp,
    )
    assert advisory.status == "abstain"
    assert advisory.evidence_chunks_count == 0
    assert len(advisory.source_citations) == 0
    assert "Insufficient" in advisory.refusal_reason


# =============================================================================
# 3. Guardrail Abstention (Out-of-Scope)
# =============================================================================

@pytest.mark.parametrize("query,crop,disease", [
    ("How to treat cotton leaf curl virus?", "Cotton", "Cotton___Leaf_curl"),
    ("Fungicide for wheat stripe rust?", "Wheat", "Wheat___Stripe_rust"),
    ("How to control sugarcane red rot?", "Sugarcane", "Sugarcane___Red_rot"),
    ("Treatment for citrus canker in Mardan?", "Citrus", "Citrus___Canker"),
])
def test_out_of_scope_guardrail_abstention(generator, query, crop, disease):
    """Verify generator properly abstains on unsupported crops in English and Urdu."""
    # English
    adv_en = generator.generate_advisory(query=query, crop=crop, disease_class=disease, language="en")
    assert adv_en.status == "information_unavailable"
    assert adv_en.evidence_chunks_count == 0
    assert len(adv_en.source_citations) == 0
    assert adv_en.direction == "ltr"

    # Urdu
    adv_ur = generator.generate_advisory(query=query, crop=crop, disease_class=disease, language="ur")
    assert adv_ur.status == "information_unavailable"
    assert adv_ur.direction == "rtl"


# =============================================================================
# 4. Glossary and Multilingual Translation
# =============================================================================

def test_agricultural_glossary_lookup():
    """Verify agricultural glossary correctly loads and provides bilingual term pairs."""
    glossary = get_glossary()
    assert len(glossary.terms) >= 20

    # Verify key agronomic terms exist
    term = glossary.get_term("late blight")
    assert term is not None
    assert "پچھیتی جھلساؤ" in term["urdu"]

    term_fungicide = glossary.get_term("fungicide")
    assert term_fungicide is not None
    assert "پھپھوند کش" in term_fungicide["urdu"]


def test_translator_entity_protection(translator):
    """Verify citation tokens and Latin binomials are protected during translation."""
    sample = "Under high humidity, [KP-AGRI-POT-002] recommends Phytophthora infestans control."
    protected, entity_map = translator._protect_entities(sample)

    # Placeholders should replace bracketed citations and Latin binomials
    assert "[KP-AGRI-POT-002]" not in protected
    assert len(entity_map) >= 2

    # Restoration should restore original exact tokens
    restored = translator._restore_entities(protected, entity_map)
    assert restored == sample


def test_translator_rtl_wrapping():
    """Verify RTL unicode wrap adds standard directional markers."""
    urdu_text = "آلو کا پچھیتی جھلساؤ"
    wrapped = AgriculturalTranslator.wrap_rtl(urdu_text)
    assert wrapped.startswith("\u2067")
    assert wrapped.endswith("\u2069")
    assert urdu_text in wrapped


# =============================================================================
# 5. End-to-End Bilingual Advisory Parity
# =============================================================================

def test_end_to_end_bilingual_advisory_supported_class(generator):
    """Verify end-to-end advisory generation for a supported class in English and Urdu."""
    query = "What are the early symptoms of apple scab and how should Swat orchardists manage it?"

    # Generate English
    adv_en = generator.generate_advisory(
        query=query,
        crop="Apple",
        disease_class="Apple___Apple_scab",
        confidence=0.97,
        language="en",
    )
    assert adv_en.status == "grounded"
    assert adv_en.language == "en"
    assert adv_en.direction == "ltr"
    assert adv_en.evidence_chunks_count > 0
    assert len(adv_en.source_citations) > 0
    assert any("KP-AGRI-APP-001" in c for c in adv_en.source_citations)

    # Generate Urdu
    adv_ur = generator.generate_advisory(
        query=query,
        crop="Apple",
        disease_class="Apple___Apple_scab",
        confidence=0.97,
        language="ur",
    )
    assert adv_ur.status == "grounded"
    assert adv_ur.language == "ur"
    assert adv_ur.direction == "rtl"
    assert adv_ur.evidence_chunks_count == adv_en.evidence_chunks_count
    # Citations must be preserved verbatim
    assert adv_ur.source_citations == adv_en.source_citations
    # Urdu symptoms and management must match list length
    assert len(adv_ur.key_symptoms) == len(adv_en.key_symptoms)
    assert len(adv_ur.recommended_management) == len(adv_en.recommended_management)
