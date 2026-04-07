"""Run evaluation against the golden dataset.

Usage:
  # Live mode (requires working RAG pipeline):
  python scripts/run_eval.py

  # Offline mode with precomputed results:
  python scripts/run_eval.py --offline eval/sample_responses.json

  # Specific items only:
  python scripts/run_eval.py --items eval-001 eval-002

  # Filter by category:
  python scripts/run_eval.py --category factual
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.dataset import GoldenDataset
from src.evaluation.evaluator import Evaluator


def make_rag_answer_fn():
    """Create a function that queries the full RAG pipeline."""
    from src.retrieval.retriever import Retriever
    from src.generation.generator import Generator

    retriever = Retriever()
    generator = Generator()

    def answer_fn(question: str) -> dict:
        chunks = retriever.retrieve(question)
        response = generator.generate(question, chunks)
        return {
            "answer": response.answer,
            "contexts": [c.text for c in chunks],
            "chunk_metadata": [
                {
                    "document_title": c.document_title,
                    "document_filename": c.document_filename,
                    "sections": ", ".join(c.sections) if c.sections else "",
                    "page_numbers": c.page_numbers,
                }
                for c in chunks
            ],
        }

    return answer_fn


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--offline", type=str, help="Path to precomputed responses JSON")
    parser.add_argument("--items", nargs="+", help="Specific item IDs to evaluate")
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument("--save", action="store_true", default=True, help="Save results")
    args = parser.parse_args()

    print("=" * 60)
    print("Healthcare RAG — Evaluation Runner")
    print("=" * 60)

    # Load golden dataset
    dataset = GoldenDataset()
    print(f"Loaded {len(dataset)} evaluation items")

    # Apply filters
    if args.items:
        dataset.items = [i for i in dataset.items if i.id in args.items]
        print(f"Filtered to {len(dataset)} items: {args.items}")
    if args.category:
        dataset.items = dataset.filter_by_category(args.category)
        print(f"Filtered to {len(dataset)} items in category: {args.category}")

    # Initialize evaluator with config thresholds
    evaluator = Evaluator()

    if args.offline:
        # Offline mode
        print(f"\nMode: OFFLINE (using {args.offline})")
        with open(args.offline) as f:
            precomputed = json.load(f)
        summary = evaluator.evaluate_dataset(dataset, precomputed=precomputed)
    else:
        # Live mode
        print("\nMode: LIVE (querying RAG pipeline)")
        try:
            answer_fn = make_rag_answer_fn()
        except Exception as e:
            print(f"\nFailed to initialize RAG pipeline: {e}")
            print("Use --offline mode with precomputed responses instead.")
            sys.exit(1)
        summary = evaluator.evaluate_dataset(dataset, answer_fn=answer_fn)

    # Report
    evaluator.print_report(summary)

    if args.save:
        evaluator.save_results(summary)

    # Exit code: 0 if pass rate > 80%, 1 otherwise
    sys.exit(0 if summary.pass_rate >= 0.8 else 1)


if __name__ == "__main__":
    main()
