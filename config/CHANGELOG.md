# Configuration Changelog
## [1.1] - 2026-11-27
### Urgent-situation safety routing
- New `safety_triage.yaml`: rule-based check that runs before retrieval and generation
- Categories: medical emergency (chest pain, stroke signs, not breathing, unresponsive), fall with head injury, suspected elder abuse or neglect
- Live situations get fixed emergency guidance (Triple Zero 000, healthdirect, 1800ELDERHelp, ACQSC) instead of a generated answer; policy questions on these topics get a normal answer with a safety note
- Helpline numbers and clinical wording need review by a clinician before production use
## [1.0] - 2026-02-28
### Initial Configuration
- System prompt v1.0: Compliance assistant with citation enforcement and refusal behaviour
- Chunk size: 700 tokens with 100 token overlap
- Retrieval: top-5 vector + top-5 BM25, RRF fusion, top-3 after re-ranking
- Generation: Claude Sonnet, temperature 0.1
- Evaluation thresholds: faithfulness 0.85, citation accuracy 0.90
