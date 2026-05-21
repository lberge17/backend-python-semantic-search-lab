from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Protocol


REQUIRED_DOCUMENT_FIELDS = ("id", "title", "category", "summary", "source")


DEFAULT_DOCUMENTS = [
    {
        "id": "DOC-101",
        "title": "Resetting a Forgotten Password",
        "category": "account",
        "summary": (
            "Explains how users can change or reset their account password "
            "after identity verification."
        ),
        "source": "platform-docs/account/password-reset",
        "tags": ["password", "account", "verification"],
    },
    {
        "id": "DOC-102",
        "title": "Fixing Invalid API Authentication Tokens",
        "category": "api",
        "summary": (
            "Explains how to resolve API calls blocked by expired, missing, "
            "or malformed bearer credentials."
        ),
        "source": "platform-docs/api/authentication-tokens",
        "tags": ["api", "authentication", "credentials"],
    },
    {
        "id": "DOC-103",
        "title": "Understanding Monthly Billing Limits",
        "category": "billing",
        "summary": (
            "Explains how usage limits affect invoices, monthly plans, "
            "and account upgrades."
        ),
        "source": "platform-docs/billing/monthly-limits",
        "tags": ["billing", "invoice", "plan"],
    },
    {
        "id": "DOC-104",
        "title": "Troubleshooting Slow Dashboard Loading",
        "category": "performance",
        "summary": (
            "Explains common causes of slow page loads, dashboard latency, "
            "and client-side rendering delays."
        ),
        "source": "platform-docs/performance/dashboard-loading",
        "tags": ["dashboard", "performance", "latency"],
    },
    {
        "id": "DOC-105",
        "title": "Updating User Profile Settings",
        "category": "account",
        "summary": (
            "Explains how users can update email, display name, "
            "notification preferences, and profile details."
        ),
        "source": "platform-docs/account/profile-settings",
        "tags": ["profile", "settings", "email"],
    },
]


class EmbeddingModel(Protocol):
    def embed(self, text: str) -> list[float]:
        """Return a vector embedding for the provided text."""
        ...


class OllamaEmbeddingModel:
    def __init__(self, model_name: str = "embeddinggemma"):
        self.model_name = model_name

    def embed(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string.")

        try:
            import ollama
        except ImportError as exc:
            raise ImportError(
                "The ollama package is required to use OllamaEmbeddingModel. "
                "Install dependencies with pipenv install."
            ) from exc

        response = ollama.embeddings(model=self.model_name, prompt=text)
        return response["embedding"]


def build_search_text(document: dict[str, Any]) -> str:

    if not isinstance(document, dict):
        raise ValueError("Document must be a dictionary.")

    parts: list[str] = []

    for field in ("title", "category", "summary"):
        value = document.get(field)

        if value is not None and str(value).strip():
            parts.append(f"{field}: {str(value).strip()}")

    tags = document.get("tags")

    if tags:
        if isinstance(tags, (list, tuple, set)):
            tag_text = ", ".join(
                str(tag).strip() for tag in tags if str(tag).strip()
            )
        else:
            tag_text = str(tags).strip()

        if tag_text:
            parts.append(f"tags: {tag_text}")

    if not parts:
        raise ValueError("Document does not contain searchable text.")

    return " | ".join(parts)


def prepare_documents(raw_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(raw_documents, list):
        raise ValueError("raw_documents must be a list of dictionaries.")

    if not raw_documents:
        raise ValueError("raw_documents must include at least one document.")

    prepared_documents: list[dict[str, Any]] = []

    for index, document in enumerate(raw_documents):
        if not isinstance(document, dict):
            raise ValueError(f"Document at index {index} must be a dictionary.")

        missing_fields = [
            field
            for field in REQUIRED_DOCUMENT_FIELDS
            if field not in document or document[field] is None
        ]

        if missing_fields:
            missing_text = ", ".join(missing_fields)
            raise ValueError(
                f"Document at index {index} is missing required field(s): "
                f"{missing_text}"
            )

        prepared_document = deepcopy(document)
        prepared_document["text"] = build_search_text(prepared_document)
        prepared_documents.append(prepared_document)

    return prepared_documents


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:

    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same number of dimensions.")

    if len(vector_a) == 0:
        return 0.0

    dot_product = 0.0
    magnitude_a = 0.0
    magnitude_b = 0.0

    for value_a, value_b in zip(vector_a, vector_b):
        numeric_a = float(value_a)
        numeric_b = float(value_b)

        dot_product += numeric_a * numeric_b
        magnitude_a += numeric_a * numeric_a
        magnitude_b += numeric_b * numeric_b

    magnitude_a = math.sqrt(magnitude_a)
    magnitude_b = math.sqrt(magnitude_b)

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def embed_documents(
    prepared_documents: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
) -> list[dict[str, Any]]:

    if not isinstance(prepared_documents, list):
        raise ValueError("prepared_documents must be a list of dictionaries.")

    embedded_documents: list[dict[str, Any]] = []

    for index, document in enumerate(prepared_documents):
        if not isinstance(document, dict):
            raise ValueError(f"Document at index {index} must be a dictionary.")

        text = document.get("text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Prepared document at index {index} must include a non-empty "
                "'text' field."
            )

        embedding = embedding_model.embed(text)

        embedded_document = deepcopy(document)
        embedded_document["embedding"] = list(embedding)
        embedded_documents.append(embedded_document)

    return embedded_documents


def rank_documents(
    query: str,
    embedded_documents: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
    top_k: int = 3,
) -> list[dict[str, Any]]:

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if not isinstance(embedded_documents, list):
        raise ValueError("embedded_documents must be a list of dictionaries.")

    query_embedding = embedding_model.embed(query.strip())

    ranked_results: list[dict[str, Any]] = []

    for index, document in enumerate(embedded_documents):
        if not isinstance(document, dict):
            raise ValueError(f"Document at index {index} must be a dictionary.")

        if "embedding" not in document:
            raise ValueError(
                f"Embedded document at index {index} is missing an 'embedding' field."
            )

        score = cosine_similarity(query_embedding, document["embedding"])

        result = {
            "id": document.get("id"),
            "title": document.get("title"),
            "category": document.get("category"),
            "summary": document.get("summary"),
            "source": document.get("source"),
            "score": float(score),
        }

        ranked_results.append(result)

    ranked_results.sort(key=lambda result: result["score"], reverse=True)

    return ranked_results[:top_k]


def semantic_search(
    query: str,
    raw_documents: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
    top_k: int = 3,
) -> list[dict[str, Any]]:

    prepared_documents = prepare_documents(raw_documents)
    embedded_documents = embed_documents(prepared_documents, embedding_model)

    return rank_documents(
        query=query,
        embedded_documents=embedded_documents,
        embedding_model=embedding_model,
        top_k=top_k,
    )


def main() -> None:
    embedder = OllamaEmbeddingModel()
    query = "Why does the mobile app say my token is expired?"

    results = semantic_search(
        query=query,
        raw_documents=DEFAULT_DOCUMENTS,
        embedding_model=embedder,
        top_k=3,
    )

    for index, result in enumerate(results, start=1):
        print(f"{index}. {result['title']} | score={result['score']:.4f}")
        print(f"   Source: {result['source']}")
        print(f"   Summary: {result['summary']}")


if __name__ == "__main__":
    main()