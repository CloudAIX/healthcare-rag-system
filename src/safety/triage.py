"""Urgent-situation safety routing.

A deterministic, rule-based check that runs before retrieval and generation.
If a question describes a possible emergency (chest pain, a fall with a head
injury, suspected elder abuse, ...), callers return fixed emergency guidance
instead of a corpus answer.

Why rules and not an LLM classifier: the check must be fast, free, testable,
and work when the Anthropic API is down. Patterns and guidance live in
config/safety_triage.yaml so they can be reviewed without touching code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "safety_triage.yaml"

EMERGENCY = "emergency"
ADVISORY = "advisory"


@dataclass
class TriageResult:
    """Outcome of a triggered safety check."""

    mode: str                      # EMERGENCY or ADVISORY
    categories: list[str]          # category ids, in config order
    labels: list[str]              # human-readable category labels
    matched_terms: list[str] = field(default_factory=list)
    message: str = ""              # guidance (emergency) or banner (advisory)

    @property
    def is_emergency(self) -> bool:
        return self.mode == EMERGENCY


@dataclass
class _Category:
    id: str
    label: str
    guidance: str
    any_of: list[re.Pattern]
    all_of: list[re.Pattern]
    allow_negation: bool


class SafetyTriage:
    def __init__(self, config: dict | None = None):
        if config is None:
            with open(CONFIG_PATH) as f:
                config = yaml.safe_load(f)
        self.config = config
        flags = re.IGNORECASE
        self.immediacy = [re.compile(p, flags) for p in config.get("immediacy_patterns", [])]
        self.policy = [re.compile(p, flags) for p in config.get("policy_patterns", [])]
        cues = sorted(config.get("negation_cues", []), key=len, reverse=True)
        self.negation = re.compile(r"\b(" + "|".join(re.escape(c) for c in cues) + r")\b", flags) if cues else None
        self.negation_window = int(config.get("negation_window_words", 3))
        self.categories = [
            _Category(
                id=c["id"],
                label=c["label"],
                guidance=c["guidance"].strip(),
                any_of=[re.compile(p, flags) for p in c.get("any_of", [])],
                all_of=[re.compile(p, flags) for p in c.get("all_of", [])],
                allow_negation=bool(c.get("allow_negation", False)),
            )
            for c in config["categories"]
        ]
        self.footer = config.get("footer", "").strip()
        self.banner = config.get("advisory_banner", "").strip()

    # --- matching helpers -------------------------------------------------

    def _negated(self, text: str, start: int) -> bool:
        """True if a negation cue sits within the few words before `start`."""
        if self.negation is None:
            return False
        before = text[:start].split()[-self.negation_window:]
        return bool(self.negation.search(" ".join(before)))

    def _find(self, pattern: re.Pattern, text: str, allow_negation: bool) -> str | None:
        for m in pattern.finditer(text):
            if allow_negation and self._negated(text, m.start()):
                continue
            # Report a short phrase, not a whole `.{0,40}` span.
            return " ".join(m.group(0).split()[:3])
        return None

    def _match_category(self, cat: _Category, text: str) -> list[str]:
        for p in cat.any_of:
            hit = self._find(p, text, cat.allow_negation)
            if hit:
                return [hit]
        if cat.all_of:
            hits = [self._find(p, text, cat.allow_negation) for p in cat.all_of]
            if all(hits):
                return hits
        return []

    # --- public API -------------------------------------------------------

    def check(self, question: str) -> TriageResult | None:
        """Return a TriageResult if the question may describe an emergency, else None."""
        text = " ".join((question or "").split())
        if not text:
            return None

        matched: list[tuple[_Category, list[str]]] = []
        for cat in self.categories:
            hits = self._match_category(cat, text)
            if hits:
                matched.append((cat, hits))
        if not matched:
            return None

        terms = [t for _, hits in matched for t in hits]
        mode = self._mode(text)
        terms_str = ", ".join(f'"{t}"' for t in dict.fromkeys(t.lower() for t in terms))
        if mode == EMERGENCY:
            body = "\n\n".join(cat.guidance for cat, _ in matched)
            message = body + ("\n\n" + self.footer.format(matched=terms_str) if self.footer else "")
        else:
            message = self.banner.format(matched=terms_str)
        return TriageResult(
            mode=mode,
            categories=[cat.id for cat, _ in matched],
            labels=[cat.label for cat, _ in matched],
            matched_terms=terms,
            message=message.strip(),
        )

    def _mode(self, text: str) -> str:
        """Decide whether to replace the answer (emergency) or annotate it (advisory).

        Emergency unless the question is clearly policy-framed AND shows no sign
        of a live situation. Any doubt resolves to emergency.
        """
        live = any(p.search(text) for p in self.immediacy)
        policy = any(p.search(text) for p in self.policy)
        return ADVISORY if (policy and not live) else EMERGENCY


@lru_cache(maxsize=1)
def get_triage() -> SafetyTriage:
    return SafetyTriage()


def check_urgent(question: str) -> TriageResult | None:
    """Module-level convenience wrapper around the shared SafetyTriage."""
    return get_triage().check(question)
