# db_adapter.py
# Database abstraction layer: supports both MySQL (production) and SQLite (standalone/dev).
# When USE_SQLITE=True, the system is fully independent of KaopuVIP.

import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

# --- Toggle: set True to use local SQLite fake DB, False for MySQL ---
USE_SQLITE = os.environ.get("USE_SQLITE", "true").lower() == "true"
SQLITE_DB_PATH = os.environ.get("SQLITE_DB_PATH",
                                 os.path.join(os.path.dirname(__file__), "fake_database.db"))


class DatabaseConnection:
    """Unified DB connection wrapper that works with both MySQL and SQLite."""

    def __init__(self, config, use_sqlite=None):
        self._use_sqlite = USE_SQLITE if use_sqlite is None else use_sqlite
        self._config = config
        self._conn = None

    def connect(self):
        if self._use_sqlite:
            self._conn = sqlite3.connect(SQLITE_DB_PATH)
            self._conn.row_factory = sqlite3.Row  # dict-like rows
        else:
            import mysql.connector
            self._conn = mysql.connector.connect(**self._config)
        return self

    def cursor(self, dictionary=False):
        if self._use_sqlite:
            return SQLiteCursorWrapper(self._conn.cursor())
        else:
            return self._conn.cursor(dictionary=dictionary)

    def close(self):
        if self._conn:
            self._conn.close()

    def is_connected(self):
        if self._use_sqlite:
            return self._conn is not None
        else:
            return self._conn.is_connected()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SQLiteCursorWrapper:
    """Wraps a sqlite3.Cursor to return list-of-dict like MySQL's dictionary=True cursor."""

    def __init__(self, cursor):
        self._cursor = cursor

    def execute(self, query, params=None):
        # Translate MySQL-specific syntax to SQLite equivalents
        query = self._translate_query(query)
        if params:
            self._cursor.execute(query, params)
        else:
            self._cursor.execute(query)

    def fetchall(self):
        rows = self._cursor.fetchall()
        if rows and isinstance(rows[0], sqlite3.Row):
            return [dict(row) for row in rows]
        # Fallback: if row_factory didn't apply (e.g. plain cursor)
        if rows and self._cursor.description:
            cols = [d[0] for d in self._cursor.description]
            return [dict(zip(cols, row)) for row in rows]
        return rows

    def close(self):
        self._cursor.close()

    @staticmethod
    def _translate_query(query):
        # Replace MySQL %s placeholders with SQLite ? placeholders
        query = query.replace("%s", "?")
        # Replace information_schema query with sqlite_master equivalent
        if "information_schema.TABLES" in query:
            query = _translate_information_schema_query(query)
        return query


def _translate_information_schema_query(query):
    """
    Translate:
      SELECT TABLE_NAME FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME IN (?, ?, ...)
    into SQLite's:
      SELECT name FROM sqlite_master WHERE type='table' AND name IN (?, ?, ...)
    """
    # Count the number of placeholders in the original query
    import re
    placeholders = re.findall(r'\?', query)
    # First ? is TABLE_SCHEMA (not needed in SQLite), rest are table names
    num_table_placeholders = len(placeholders) - 1
    in_clause = ", ".join(["?"] * num_table_placeholders)
    return f"SELECT name AS TABLE_NAME FROM sqlite_master WHERE type='table' AND name IN ({in_clause})"


def get_connection(config):
    """Factory function: returns a DatabaseConnection."""
    return DatabaseConnection(config, USE_SQLITE)


def check_db_reachable(config):
    """Check if database is reachable. Used for environment auto-detection."""
    if USE_SQLITE:
        return os.path.exists(SQLITE_DB_PATH)
    else:
        try:
            import mysql.connector
            conn = mysql.connector.connect(**config)
            conn.close()
            return True
        except Exception:
            return False


def execute_info_schema_query(cursor, db_config, possible_tables):
    """
    Execute table existence check compatible with both MySQL and SQLite.
    For SQLite, the TABLE_SCHEMA param is stripped automatically.
    """
    if USE_SQLITE:
        if not possible_tables:
            return []
        format_strings = ','.join(['?'] * len(possible_tables))
        query = f"SELECT name AS TABLE_NAME FROM sqlite_master WHERE type='table' AND name IN ({format_strings})"
        cursor.execute(query, tuple(possible_tables))
    else:
        if not possible_tables:
            return []
        format_strings = ','.join(['%s'] * len(possible_tables))
        query = f"""
            SELECT TABLE_NAME
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME IN ({format_strings})
        """
        params = [db_config['database']] + possible_tables
        cursor.execute(query, tuple(params))
    return cursor.fetchall()
