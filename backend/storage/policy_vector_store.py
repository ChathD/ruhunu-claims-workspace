"""Store and rank policy embeddings in SQLite using sqlite-vec."""
import re
import json

import sqlite_vec
from sqlalchemy import text

from backend.evidence.policy_embeddings import DIMENSIONS, MODEL


STORAGE = "sqlite-vec"
POSTGRES_STORAGE = "pgvector"
CANDIDATE_LIMIT = 10
WORDS = re.compile(r"[a-z]{3,}")


def load_policy_vectors(connection):
    connection.enable_load_extension(True)
    try:
        sqlite_vec.load(connection)
    finally:
        connection.enable_load_extension(False)
    connection.create_function("policy_keyword_matches", 2, keyword_matches, deterministic=True)


def initialize_policy_vectors(engine, *, connection=None):
    if connection is not None:
        connection.exec_driver_sql(f"""
            CREATE TABLE IF NOT EXISTS policy_chunk_vectors (
                chunk_id text PRIMARY KEY REFERENCES policy_chunks(id) ON DELETE CASCADE,
                document_id text NOT NULL REFERENCES policy_documents(id),
                embedding vector({DIMENSIONS}) NOT NULL
            )
        """)
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS policy_chunk_vectors_document ON policy_chunk_vectors(document_id)")
        return
    with engine.begin() as connection:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        connection.exec_driver_sql(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS policy_chunk_vectors USING vec0(
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT partition key,
                embedding FLOAT[{DIMENSIONS}] distance_metric=cosine,
                chunk_size=64
            )
        """)
        columns = connection.exec_driver_sql("PRAGMA table_info(policy_chunks)").all()
        if not any(column.name == "vector" for column in columns):
            return

        # Existing OpenAI vectors can be moved without calling the provider again.
        compatible = connection.execute(text("""
            SELECT d.id
            FROM policy_documents AS d JOIN policy_chunks AS c ON c.document_id = d.id
            WHERE json_extract(d.embedding, '$.model') = :model
              AND json_extract(d.embedding, '$.dimension') = :dimension
            GROUP BY d.id
            HAVING COUNT(*) = COUNT(CASE WHEN json_array_length(c.vector) = :dimension THEN 1 END)
        """), {"model": MODEL, "dimension": DIMENSIONS}).scalars().all()
        for identifier in compatible:
            connection.exec_driver_sql("""
                INSERT INTO policy_chunk_vectors (chunk_id, document_id, embedding)
                SELECT id, document_id, vec_f32(vector) FROM policy_chunks WHERE document_id = ?
            """, (identifier,))
            connection.exec_driver_sql("""
                UPDATE policy_documents SET embedding = json_set(embedding, '$.storage', ?) WHERE id = ?
            """, (STORAGE, identifier))
        connection.exec_driver_sql("ALTER TABLE policy_chunks DROP COLUMN vector")


def replace_policy_vectors(session, identifier, chunks, vectors):
    postgres = session.get_bind().dialect.name == "postgresql"
    serialize = json.dumps if postgres else sqlite_vec.serialize_float32
    rows = [{"id": chunk.id, "document": identifier, "embedding": serialize(vector)}
            for chunk, vector in zip(chunks, vectors, strict=True)]
    session.execute(text("DELETE FROM policy_chunk_vectors WHERE document_id = :document"), {"document": identifier})
    if rows:
        value = "CAST(:embedding AS vector)" if postgres else ":embedding"
        session.execute(text(f"""INSERT INTO policy_chunk_vectors (chunk_id, document_id, embedding)
            VALUES (:id, :document, {value})"""), rows)


def keyword_matches(keywords, passage):
    return len(set(keywords.split()).intersection(WORDS.findall(passage.casefold()))) if keywords else 0


def rank_policy_passages(session, identifier, vector, keyword_query):
    keywords = " ".join(sorted(set(WORDS.findall(keyword_query.casefold()))))
    if session.get_bind().dialect.name == "postgresql":
        return session.execute(text("""
            SELECT c.id, c.page, c.text, 1.0 - (v.embedding <=> CAST(:query AS vector))
                + 0.02 * (SELECT count(DISTINCT word) FROM regexp_split_to_table(lower(c.text), '[^a-z]+') AS words(word)
                          WHERE word = ANY(string_to_array(:keywords, ' ')) AND word <> '') AS score
            FROM policy_chunk_vectors AS v
            JOIN policy_chunks AS c ON c.id = v.chunk_id AND c.document_id = v.document_id
            WHERE v.document_id = :document
            ORDER BY score DESC, c.page, length(c.id), c.id
            LIMIT :limit
        """), {"document": identifier, "query": json.dumps(vector),
               "keywords": keywords, "limit": CANDIDATE_LIMIT}).all()
    return session.execute(text("""
        SELECT c.id, c.page, c.text, 1.0 - vec_distance_cosine(v.embedding, :query)
            + 0.02 * policy_keyword_matches(:keywords, c.text) AS score
        FROM policy_chunk_vectors AS v
        JOIN policy_chunks AS c ON c.id = v.chunk_id AND c.document_id = v.document_id
        WHERE v.document_id = :document
        ORDER BY score DESC, c.rowid
        LIMIT :limit
    """), {"document": identifier, "query": sqlite_vec.serialize_float32(vector),
           "keywords": keywords, "limit": CANDIDATE_LIMIT}).all()
