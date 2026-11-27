"""Embed policy passages and retrieval queries with the OpenAI Embeddings API."""
import math
import os

import httpx


MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
ENDPOINT = "https://api.openai.com/v1/embeddings"
BATCH_SIZE = 32
MAX_INPUT_BYTES = 8192


class PolicyEmbeddingError(RuntimeError):
    pass


def embed_texts(texts: list[str]) -> list[list[float]]:
    # The vector store imports this module's model constants during database setup.
    from backend.assurance.investigation_tracing import trace_span

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise PolicyEmbeddingError("Set OPENAI_API_KEY before indexing or searching policy wording.")

    # UTF-8 length bounds the token count, even for unusually long OCR tokens.
    inputs = [text.encode("utf-8", errors="replace")[:MAX_INPUT_BYTES].decode("utf-8", errors="ignore").strip()
              for text in texts]
    if not all(inputs):
        raise ValueError("Policy embedding input must contain text.")

    vectors = []
    with httpx.Client(timeout=30, headers={"Authorization": f"Bearer {key}"}) as client:
        for start in range(0, len(inputs), BATCH_SIZE):
            batch = inputs[start:start + BATCH_SIZE]
            request = {"model": MODEL, "input": batch, "encoding_format": "float"}
            with trace_span("model.openai_embeddings", kind="embedding", metadata={"batch_size": len(batch)},
                            content=request, retain_content=False, generation={"model": MODEL}) as span:
                try:
                    response = client.post(ENDPOINT, json=request)
                    response.raise_for_status()
                    data = response.json()
                    usage = data.get("usage") if isinstance(data, dict) else None
                    tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
                    if type(tokens) is int and tokens >= 0:
                        span.usage = {"input": tokens}
                    rows = sorted(data["data"], key=lambda row: row["index"])
                    if data["model"] != MODEL or [row["index"] for row in rows] != list(range(len(batch))):
                        raise ValueError("Unexpected embedding model or result indexes")
                    batch_vectors = [row["embedding"] for row in rows]
                    if any(len(vector) != DIMENSIONS or not all(math.isfinite(value) for value in vector) for vector in batch_vectors):
                        raise ValueError("Invalid embedding vector")
                except httpx.HTTPStatusError as error:
                    raise PolicyEmbeddingError(f"OpenAI embeddings returned HTTP {error.response.status_code}.") from error
                except httpx.RequestError as error:
                    raise PolicyEmbeddingError("Could not reach the OpenAI embeddings API.") from error
                except (KeyError, TypeError, ValueError) as error:
                    raise PolicyEmbeddingError("OpenAI returned an invalid embedding response.") from error
                span.summary = {"result_count": len(batch_vectors)}
            vectors.extend(batch_vectors)
    return vectors
