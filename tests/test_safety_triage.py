"""Tests for urgent-situation safety routing. No models, no network.

Covers the rule-based classifier itself, then each entry point (Generator,
FastAPI /query, MCP standards_ask) to prove an emergency never reaches
retrieval or the LLM.
"""
import json
import os
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.safety import ADVISORY, EMERGENCY, SafetyTriage, check_urgent


@pytest.fixture(autouse=True)
def _traces_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))


@pytest.fixture(scope="module")
def triage():
    return SafetyTriage()


# --- Classifier: live emergencies -----------------------------------------

EMERGENCIES = [
    # chest pain / cardiac
    ("My mum has chest pain and is sweating, what should I do?", "medical_emergency"),
    ("Dad says his chest feels tight and the pain is going down his arm", "medical_emergency"),
    ("Resident is complaining of crushing chest pressure", "medical_emergency"),
    ("I think my husband is having a heart attack", "medical_emergency"),
    # other life-threatening
    ("My father is not breathing", "medical_emergency"),
    ("Grandma's face is drooping and her speech is slurred", "medical_emergency"),
    ("Mum collapsed in the kitchen and won't wake up", "medical_emergency"),
    # fall with head injury
    ("A resident just fell and hit her head, she seems confused", "fall_head_injury"),
    ("Dad fell in the bathroom an hour ago and banged his head", "fall_head_injury"),
    ("She hit her head when she fell yesterday", "fall_head_injury"),
    ("Mum slipped on the stairs and there's blood coming from her head", "fall_head_injury"),
    ("My nan had a fall and is on warfarin, should I be worried?", "fall_head_injury"),
    ("Found him on the floor this morning, he has a head wound", "fall_head_injury"),
    # suspected elder abuse
    ("I think a carer is hitting my grandmother at her nursing home", "elder_abuse"),
    ("Mum has unexplained bruises on her arms after her stay in respite", "elder_abuse"),
    ("Someone is taking my mum's pension and won't let her see us", "elder_abuse"),
    ("Dad says he is scared of the night staff", "elder_abuse"),
    ("My brother is financially abusing our father", "elder_abuse"),
    ("I think my aunt is being neglected in her care home", "elder_abuse"),
]


@pytest.mark.parametrize("question,category", EMERGENCIES)
def test_emergencies_are_caught(triage, question, category):
    r = triage.check(question)
    assert r is not None, f"missed: {question!r}"
    assert r.mode == EMERGENCY
    assert category in r.categories


@pytest.mark.parametrize("question,category", EMERGENCIES)
def test_emergency_message_points_to_000(triage, question, category):
    r = triage.check(question)
    assert "000" in r.message
    assert "cannot assess emergencies" in r.message  # footer explains why


def test_abuse_guidance_includes_helpline(triage):
    r = triage.check("I think the carer is hitting mum")
    assert "1800 353 374" in r.message  # 1800ELDERHelp


def test_head_injury_guidance_covers_blood_thinners(triage):
    r = triage.check("Dad fell and banged his head")
    assert "blood thinners" in r.message


def test_multiple_categories_combine(triage):
    r = triage.check("Mum fell and hit her head and now has chest pain")
    assert r.mode == EMERGENCY
    assert r.categories == ["medical_emergency", "fall_head_injury"]


def test_hit_her_head_is_not_read_as_abuse(triage):
    r = triage.check("She hit her head when she fell")
    assert "elder_abuse" not in r.categories


# --- Classifier: ordinary questions pass through --------------------------

ORDINARY = [
    "What are the requirements for dignity and choice under Standard 1?",
    "How do we document medication administration?",
    "What are the falls prevention requirements under Standard 5?",
    "Resident fell but didn't hit her head and has no other injuries",
    "What does advance care planning involve?",
    "Who is the head of the Aged Care Quality and Safety Commission?",
    "How often should care plans be reviewed?",
    "",
]


@pytest.mark.parametrize("question", ORDINARY)
def test_ordinary_questions_not_flagged(triage, question):
    assert triage.check(question) is None


# --- Classifier: policy questions on urgent topics ------------------------

@pytest.mark.parametrize("question", [
    "What does the SIRS policy say about reporting suspected abuse?",
    "What training must staff receive on recognising stroke?",
    "What are the procedure requirements after a resident has a head injury from a fall?",
])
def test_policy_questions_are_advisory(triage, question):
    r = triage.check(question)
    assert r is not None and r.mode == ADVISORY
    assert "000" in r.message


@pytest.mark.parametrize("question", [
    # policy words, but clearly happening now: emergency wins
    "What is the policy? A resident just fell and hit her head",
    "My mum has chest pain, what does the standard say?",
    "Help, what's the procedure, a resident is not breathing",
])
def test_live_situation_overrides_policy_framing(triage, question):
    assert triage.check(question).mode == EMERGENCY


def test_negation_only_where_allowed(triage):
    # Falls allow negation; breathing emergencies never do ("not breathing" must fire).
    assert triage.check("He fell over but did not hit his head") is None
    assert triage.check("She is not breathing").mode == EMERGENCY


def test_whitespace_and_case_insensitive(triage):
    assert triage.check("MY   DAD\nHAS  CHEST\tPAIN").mode == EMERGENCY


def test_module_helper_matches_class():
    assert check_urgent("Dad has chest pain").categories == ["medical_emergency"]


def test_custom_config_is_honoured():
    cfg = {
        "categories": [{"id": "x", "label": "X", "guidance": "Call 000.", "any_of": [r"\bzebra\b"]}],
    }
    t = SafetyTriage(cfg)
    assert t.check("a zebra is here").categories == ["x"]
    assert t.check("a horse is here") is None


# --- Generator: no LLM call for emergencies -------------------------------

def _make_generator():
    from src.generation.generator import Generator

    with patch("src.generation.generator.anthropic.Anthropic") as client_cls:
        gen = Generator()
    gen.client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text="Policy answer [Source: Doc, s1, p.1]")]
    msg.usage.input_tokens, msg.usage.output_tokens = 100, 20
    gen.client.messages.create.return_value = msg
    return gen


def test_generator_short_circuits_emergency():
    gen = _make_generator()
    resp = gen.generate("My mum has chest pain right now", chunks=[])
    gen.client.messages.create.assert_not_called()
    assert resp.model == "safety-triage"
    assert resp.input_tokens == resp.output_tokens == 0 and resp.cost_usd == 0
    assert resp.safety.mode == EMERGENCY and "000" in resp.answer


def test_generator_prepends_advisory_banner():
    gen = _make_generator()
    resp = gen.generate("What does the SIRS policy say about suspected abuse?", chunks=[])
    gen.client.messages.create.assert_called_once()
    assert resp.answer.startswith("> **Safety note:**")
    assert resp.answer.endswith("Policy answer [Source: Doc, s1, p.1]")
    assert resp.safety.mode == ADVISORY


def test_generator_normal_question_untouched():
    gen = _make_generator()
    resp = gen.generate("How often should care plans be reviewed?", chunks=[])
    assert resp.answer == "Policy answer [Source: Doc, s1, p.1]"
    assert resp.safety is None


# --- FastAPI /query --------------------------------------------------------

@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient

    with patch.dict(os.environ, {"RAG_API_KEY": "test-key-12345"}):
        import importlib
        import src.api.security as sec
        import src.api.app as api_app

        importlib.reload(sec)
        importlib.reload(api_app)
        api_app.retriever = MagicMock()
        api_app.generator = MagicMock()
        yield TestClient(api_app.app, raise_server_exceptions=False), api_app


def test_api_returns_guidance_without_retrieval(api_client):
    client, api_app = api_client
    resp = client.post(
        "/query",
        json={"question": "A resident just fell and hit her head"},
        headers={"X-API-Key": "test-key-12345"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["safety_mode"] == "emergency"
    assert data["safety_categories"] == ["fall_head_injury"]
    assert data["model"] == "safety-triage" and data["cost_usd"] == 0
    assert "000" in data["answer"]
    api_app.retriever.retrieve.assert_not_called()
    api_app.generator.generate.assert_not_called()


def test_api_emergency_still_requires_auth(api_client):
    client, _ = api_client
    resp = client.post("/query", json={"question": "My mum has chest pain"})
    assert resp.status_code == 401


def test_api_trace_records_safety(api_client, tmp_path):
    client, _ = api_client
    client.post(
        "/query",
        json={"question": "My dad has chest pain"},
        headers={"X-API-Key": "test-key-12345"},
    )
    line = (tmp_path / "traces" / "traces.jsonl").read_text().strip().splitlines()[-1]
    trace = json.loads(line)
    assert trace["safety"] == {"mode": "emergency", "categories": ["medical_emergency"]}
    assert [s["name"] for s in trace["spans"]] == ["safety_check"]


# --- MCP standards_ask ----------------------------------------------------

def test_mcp_ask_returns_guidance_without_retrieval():
    from src.mcp_server import server as srv
    from src.mcp_server.server import AskInput, standards_ask

    srv._retriever = MagicMock()
    srv._generator = MagicMock()
    try:
        out = json.loads(standards_ask(AskInput(question="I think a carer is hitting my mum")))
        assert out["safety"] == {"mode": "emergency", "categories": ["elder_abuse"]}
        assert out["citations"] == [] and "1800 353 374" in out["answer"]
        srv._retriever.retrieve.assert_not_called()
        srv._generator.generate.assert_not_called()
    finally:
        srv._retriever = None
        srv._generator = None
