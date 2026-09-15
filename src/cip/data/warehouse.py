"""Analytics warehouse.

DuckDB by default: a single file, no server, and fast enough to hold every
result this platform produces. Estimates are written as rows rather than left
in a notebook so that a later run can be compared against an earlier one, and
so a dashboard reads results rather than recomputing them.

PostgreSQL is supported for a shared warehouse; the schema is identical and is
created with the same DDL, so the local database cannot drift from the shared
one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..errors import DataError
from ..logging_utils import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS estimates (
    run_name        VARCHAR NOT NULL,
    created_at      TIMESTAMP NOT NULL,
    dataset         VARCHAR NOT NULL,
    estimand        VARCHAR NOT NULL,   -- ATE, ATT, CATE
    method          VARCHAR NOT NULL,
    estimate        DOUBLE  NOT NULL,
    standard_error  DOUBLE,
    ci_lower        DOUBLE,
    ci_upper        DOUBLE,
    n_units         BIGINT,
    n_treated       BIGINT,
    truth           DOUBLE,             -- known only for simulations and benchmarks
    bias            DOUBLE,
    covers_truth    BOOLEAN,
    notes           VARCHAR
);

CREATE TABLE IF NOT EXISTS diagnostics (
    run_name        VARCHAR NOT NULL,
    created_at      TIMESTAMP NOT NULL,
    dataset         VARCHAR NOT NULL,
    check_name      VARCHAR NOT NULL,
    passed          BOOLEAN NOT NULL,
    statistic       DOUBLE,
    threshold       DOUBLE,
    detail          VARCHAR
);

CREATE TABLE IF NOT EXISTS decisions (
    run_name        VARCHAR NOT NULL,
    created_at      TIMESTAMP NOT NULL,
    dataset         VARCHAR NOT NULL,
    recommendation  VARCHAR NOT NULL,
    effect          DOUBLE,
    probability     DOUBLE,
    reason          VARCHAR NOT NULL
);
"""


class Warehouse:
    """Result storage, backed by DuckDB."""

    def __init__(self, path: str = "data/processed/cip.duckdb") -> None:
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - a hard dependency
            raise DataError("duckdb is not installed; run `make install`") from exc

        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._connection: Any = duckdb.connect(path)
        self._connection.execute(SCHEMA)

    def write(self, table: str, rows: pd.DataFrame) -> int:
        """Append rows to a table."""
        if rows.empty:
            return 0
        if table not in {"estimates", "diagnostics", "decisions"}:
            raise DataError(f"unknown table {table!r}")
        # Registered as a view so DuckDB reads the frame directly rather than
        # going through a row-by-row insert.
        self._connection.register("_incoming", rows)
        columns = ", ".join(rows.columns)
        self._connection.execute(f"INSERT INTO {table} ({columns}) SELECT {columns} FROM _incoming")
        self._connection.unregister("_incoming")
        logger.info("warehouse_write", table=table, rows=len(rows))
        return len(rows)

    def read(self, table: str, *, run_name: str | None = None) -> pd.DataFrame:
        if table not in {"estimates", "diagnostics", "decisions"}:
            raise DataError(f"unknown table {table!r}")
        if run_name:
            return self._connection.execute(
                f"SELECT * FROM {table} WHERE run_name = ? ORDER BY created_at", [run_name]
            ).df()
        return self._connection.execute(f"SELECT * FROM {table} ORDER BY created_at").df()

    def query(self, sql: str, parameters: list[Any] | None = None) -> pd.DataFrame:
        """Run a read-only query against the warehouse."""
        return self._connection.execute(sql, parameters or []).df()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Warehouse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
