"""Private PostgreSQL storage shared by the workspace and migration utility."""
from contextlib import contextmanager
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url


def postgres_engine(database_url):
    url = make_url(database_url)
    if url.get_backend_name() != "postgresql":
        raise ValueError("SUPABASE_DATABASE_URL must be a PostgreSQL connection string.")
    url = url.set(drivername="postgresql+psycopg").update_query_dict({"sslmode": url.query.get("sslmode", "require")})
    engine = create_engine(url, pool_size=5, max_overflow=5, pool_pre_ping=True, hide_parameters=True,
                           connect_args={"connect_timeout": 10})

    @event.listens_for(engine, "connect")
    def select_schema(connection, _record):
        connection.execute("SET search_path TO ruhunu, extensions")
        connection.commit()

    return engine


def lock_workspace(connection):
    connection.execute(text("SELECT pg_advisory_xact_lock(9032486500254246)"))


@contextmanager
def worker_lease(store):
    if not store.database_url:
        yield True
        return
    with store.engine.connect() as connection:
        acquired = connection.scalar(text("SELECT pg_try_advisory_lock(9032486500254247)"))
        connection.commit()
        try:
            yield acquired
        finally:
            if acquired:
                connection.execute(text("SELECT pg_advisory_unlock(9032486500254247)"))
                connection.commit()


def begin_write(session):
    connection = session.connection()
    if connection.dialect.name == "postgresql":
        lock_workspace(connection)
    else:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
