from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from rag_pipeline import (
    build_corpus_from_files,
    build_vectorstore,
    compute_retrieval_metrics,
    get_embeddings,
    retrieve_context,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval with labeled chunk ids and compute actual Precision@k."
    )
    parser.add_argument(
        "--pdf-dir",
        default="evaluation_pdfs",
        help="Directory containing the PDFs used in evaluation.",
    )
    parser.add_argument(
        "--dataset",
        default="evaluation_dataset.json",
        help="JSON file with questions and relevant_chunk_ids.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k retrieval cutoff for Precision@k / Recall@k / F1@k.",
    )
    parser.add_argument(
        "--min-relevance",
        type=float,
        default=0.35,
        help="Minimum relevance threshold used by the retriever.",
    )
    return parser.parse_args()


def load_dataset(dataset_path: Path) -> list[dict]:
    with dataset_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required in .env to run evaluation.")

    args = parse_args()
    pdf_dir = Path(args.pdf_dir)
    dataset_path = Path(args.dataset)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDF files found in {pdf_dir}")

    evaluation_rows = load_dataset(dataset_path)
    documents = build_corpus_from_files(pdf_files)
    embeddings = get_embeddings()
    vectorstore, collection_name = build_vectorstore(
        documents=documents,
        embeddings=embeddings,
        persist_root=Path(".chroma"),
    )

    print(f"Indexed {len(pdf_files)} PDF(s) into collection {collection_name}")
    print(f"Loaded {len(evaluation_rows)} evaluation question(s)\n")

    totals = {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "f1_at_k": 0.0,
    }

    for index, row in enumerate(evaluation_rows, start=1):
        question = row["question"]
        relevant_chunk_ids = row["relevant_chunk_ids"]

        retrieved_chunks, _metrics = retrieve_context(
            vectorstore=vectorstore,
            question=question,
            top_k=args.top_k,
            min_relevance=args.min_relevance,
        )

        retrieved_chunk_ids = [
            str(chunk.document.metadata.get("chunk_id", ""))
            for chunk in retrieved_chunks
        ]

        scores = compute_retrieval_metrics(
            retrieved_chunk_ids=retrieved_chunk_ids,
            relevant_chunk_ids=relevant_chunk_ids,
            k=args.top_k,
        )

        totals["precision_at_k"] += scores["precision_at_k"]
        totals["recall_at_k"] += scores["recall_at_k"]
        totals["f1_at_k"] += scores["f1_at_k"]

        print(f"Q{index}: {question}")
        print(f"Relevant chunk ids: {', '.join(relevant_chunk_ids)}")
        print(f"Retrieved chunk ids: {', '.join(retrieved_chunk_ids) or 'None'}")
        print(
            f"Precision@{args.top_k}: {scores['precision_at_k']:.4f} | "
            f"Recall@{args.top_k}: {scores['recall_at_k']:.4f} | "
            f"F1@{args.top_k}: {scores['f1_at_k']:.4f}\n"
        )

    question_count = len(evaluation_rows)
    print("Average Metrics")
    print(
        f"Precision@{args.top_k}: {totals['precision_at_k'] / question_count:.4f}\n"
        f"Recall@{args.top_k}: {totals['recall_at_k'] / question_count:.4f}\n"
        f"F1@{args.top_k}: {totals['f1_at_k'] / question_count:.4f}"
    )


if __name__ == "__main__":
    main()
