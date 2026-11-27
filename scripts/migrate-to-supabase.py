"""Copy a stopped SQLite workspace into the private Supabase schema."""
import argparse
import json
import os
import sqlite3
import struct
import sys
from contextlib import closing
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import JSON, func, select, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.storage.policy_vector_store import POSTGRES_STORAGE, STORAGE, load_policy_vectors
from backend.storage.postgres import lock_workspace
from backend.storage.sqlite_store import Base, EvaluationLabel, Store
from backend.workflow.claims_workflow import ClaimsWorkflow


def open_source(path):
    connection = sqlite3.connect(path.resolve(strict=True).as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("BEGIN")
    return connection


def copy_tables(source, labels, destination):
    counts = {}
    for table in Base.metadata.sorted_tables:
        if destination.scalar(select(func.count()).select_from(table)):
            raise ValueError(f"The destination already has data in {table.name}. Use an empty ruhunu schema.")
        database = labels if table is EvaluationLabel.__table__ else source
        if not database.execute("SELECT 1 FROM sqlite_master WHERE name=? AND type='table'", (table.name,)).fetchone():
            if table is not EvaluationLabel.__table__:
                raise ValueError(f"Missing source table {table.name}.")
            continue
        cursor = database.execute(f'SELECT * FROM "{table.name}"')
        json_fields = [column.name for column in table.columns if isinstance(column.type, JSON)]
        count = 0
        while rows := cursor.fetchmany(100):
            values = []
            for raw in rows:
                item = {column.name: raw[column.name] if column.name in raw.keys() else None for column in table.columns}
                for field in json_fields:
                    if item[field] is not None:
                        item[field] = json.loads(item[field])
                if table.name == "policy_documents" and (embedding := item["embedding"]).get("storage") == STORAGE:
                    embedding["storage"] = POSTGRES_STORAGE
                values.append(item)
            destination.execute(table.insert(), values)
            count += len(values)
        if destination.scalar(select(func.count()).select_from(table)) != count:
            raise ValueError(f"Row counts differ in {table.name}.")
        counts[table.name] = count
        for column in table.primary_key.columns:
            sequence = destination.scalar(text("SELECT pg_get_serial_sequence(:table, :column)"),
                                          {"table": f"ruhunu.{table.name}", "column": column.name})
            if sequence and count:
                latest = destination.scalar(select(func.max(column)))
                destination.execute(text("SELECT setval(CAST(:sequence AS regclass), :latest)"),
                                    {"sequence": sequence, "latest": latest})
    return counts


def copy_vectors(source, destination):
    load_policy_vectors(source)
    cursor = source.execute("SELECT chunk_id,document_id,embedding FROM policy_chunk_vectors")
    count = 0
    while rows := cursor.fetchmany(100):
        destination.execute(text("""INSERT INTO policy_chunk_vectors (chunk_id,document_id,embedding)
            VALUES (:chunk_id,:document_id,CAST(:embedding AS vector))"""), [
                {"chunk_id": row[0], "document_id": row[1],
                 "embedding": json.dumps(struct.unpack(f"<{len(row[2]) // 4}f", row[2]))} for row in rows])
        count += len(rows)
    if destination.scalar(text("SELECT count(*) FROM policy_chunk_vectors")) != count:
        raise ValueError("Policy vector counts differ.")
    return count


def copy_checkpoints(source, target):
    from langgraph.checkpoint.sqlite import SqliteSaver

    saver = SqliteSaver(source)
    source.execute("PRAGMA query_only=ON")
    # SQLite schema setup issues CREATE TABLE IF NOT EXISTS and commits, even on reads.
    saver.is_setup = True
    for item in saver.list(None):
        keys = dict(item.config["configurable"])
        keys.pop("checkpoint_id")
        parent = item.parent_config or {"configurable": keys}
        target.put(parent, item.checkpoint, item.metadata, item.checkpoint["channel_versions"])
    with target.conn.cursor() as cursor:
        for row in source.execute("SELECT thread_id,checkpoint_ns,checkpoint_id,task_id,idx,channel,type,value,task_path FROM writes"):
            cursor.execute("""INSERT INTO checkpoint_writes (thread_id,checkpoint_ns,checkpoint_id,task_id,idx,channel,type,blob,task_path)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""", tuple(row))
    for sqlite_name, postgres_name in (("checkpoints", "checkpoints"), ("writes", "checkpoint_writes")):
        count = source.execute(f"SELECT count(*) FROM {sqlite_name}").fetchone()[0]
        stored = target.conn.execute(f"SELECT count(*) AS n FROM {postgres_name}").fetchone()["n"]
        if count != stored:
            raise ValueError(f"Checkpoint counts differ for {sqlite_name}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / ".local" / "workspace")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    if not (url := os.getenv("SUPABASE_DATABASE_URL")):
        parser.error("Set SUPABASE_DATABASE_URL in .env.")
    directory = args.source
    with closing(open_source(directory / "claims.sqlite3")) as source, closing(open_source(directory / "labels.sqlite3")) as labels, \
            closing(open_source(directory / "checkpoints.sqlite3")) as checkpoints:
        if source.execute("SELECT 1 FROM assessments WHERE status IN ('queued', 'processing') LIMIT 1").fetchone():
            raise ValueError("Wait for active investigations before switching storage.")
        store = Store(directory / "claims.sqlite3", database_url=url)
        try:
            store.initialize_postgres()
            workflow = ClaimsWorkflow(store, directory / "checkpoints.sqlite3")
            try:
                with store.engine.begin() as destination:
                    lock_workspace(destination)
                    counts = copy_tables(source, labels, destination)
                    counts["policy_chunk_vectors"] = copy_vectors(source, destination)
                    copy_checkpoints(checkpoints, workflow.checkpointer)
            finally:
                workflow.close()
        finally:
            store.engine.dispose()
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
