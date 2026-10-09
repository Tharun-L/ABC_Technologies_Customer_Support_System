"""
knowledge_base.py
------------------
Task 6: RAG pipeline integration.

Loads the 4 company knowledge-base documents, splits them into chunks,
and builds a TF-IDF vector index for retrieval. TF-IDF is used (instead
of a hosted embedding API) so the whole RAG pipeline runs fully offline
and deterministically -- no API key required for retrieval. The LLM
provider used for *generation* is swapped independently (see
llm_provider.py), so this still demonstrates a real "retrieve relevant
chunks -> inject into prompt -> generate grounded answer" RAG pattern.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")

DOCUMENT_FILES = {
    "Company Policy Document": "company_policy.txt",
    "Pricing Guide": "pricing_guide.txt",
    "Technical Manual": "technical_manual.txt",
    "FAQ Document": "faq_document.txt",
}


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def _split_into_chunks(text: str, source: str, start_id: int) -> List[Chunk]:
    """Split a document into paragraph-level chunks (simple, explainable chunking)."""
    raw_paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks = []
    cid = start_id
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        # Keep numbered-section headers attached to their content; chunk is
        # already paragraph-sized which works well for these documents.
        chunks.append(Chunk(text=para, source=source, chunk_id=cid))
        cid += 1
    return chunks


class KnowledgeBase:
    """A minimal but real RAG retriever backed by TF-IDF cosine similarity."""

    def __init__(self, docs_dir: str = DOCS_DIR):
        self.docs_dir = docs_dir
        self.chunks: List[Chunk] = []
        self._load_documents()
        self._build_index()

    def _load_documents(self):
        cid = 0
        for doc_name, filename in DOCUMENT_FILES.items():
            path = os.path.join(self.docs_dir, filename)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            doc_chunks = _split_into_chunks(text, doc_name, cid)
            self.chunks.extend(doc_chunks)
            cid += len(doc_chunks)

    def _build_index(self):
        corpus = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus) if corpus else None

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.05) -> List[Tuple[Chunk, float]]:
        """Return the top_k most relevant chunks for the query, with scores."""
        if not self.chunks or self.matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        ranked_idx = scores.argsort()[::-1][:top_k]
        results = [(self.chunks[i], float(scores[i])) for i in ranked_idx if scores[i] >= min_score]
        return results

    def retrieve_for_department(self, query: str, department: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """
        Bias retrieval toward the document most relevant to a department,
        but still search the full corpus (a real refund question might also
        need the FAQ, etc.) Department -> preferred source mapping nudges
        ranking via a small boost rather than hard filtering.
        """
        preferred_source = {
            "Sales": "Pricing Guide",
            "Technical": "Technical Manual",
            "Billing": "Company Policy Document",
            "Account": "FAQ Document",
        }.get(department)

        results = self.retrieve(query, top_k=top_k * 2)
        if preferred_source:
            results = sorted(
                results,
                key=lambda pair: (pair[0].source != preferred_source, -pair[1]),
            )
        return results[:top_k]


# Module-level singleton so the index is built once per process.
_kb_instance: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance
