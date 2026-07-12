from __future__ import annotations

import argparse
from pathlib import Path

from rag_pipeline import build_corpus_from_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List chunk ids with source, page, and excerpt for evaluation labeling."
    )
    parser.add_argument(
        "--pdf-dir",
        default="evaluation_pdfs",
        help="Directory containing PDFs to inspect.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=220,
        help="Maximum excerpt length per chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_dir = Path(args.pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        raise SystemExit(f"No PDF files found in {pdf_dir}")

    documents = build_corpus_from_files(pdf_files)

    for document in documents:
        chunk_id = document.metadata.get("chunk_id", "unknown")
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page", "?")
        excerpt = " ".join(document.page_content.split())[: args.limit]

        print(f"chunk_id: {chunk_id}")
        print(f"source:   {source}")
        print(f"page:     {page}")
        print(f"excerpt:  {excerpt}")
        print("-" * 80)


if __name__ == "__main__":
    main()
