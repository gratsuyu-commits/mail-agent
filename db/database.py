import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "self_ai.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def initialize():
    conn = get_connection()
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS learning_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            mail_id    TEXT NOT NULL,
            task_type  TEXT NOT NULL,
            features   TEXT NOT NULL,
            action     TEXT NOT NULL,
            reward     INTEGER,
            created_at DATETIME DEFAULT (datetime('now', 'localtime'))
        );

        CREATE TABLE IF NOT EXISTS trust_scores (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            task_type  TEXT NOT NULL,
            score      REAL NOT NULL DEFAULT 0.0,
            mode       TEXT NOT NULL DEFAULT 'confirm',
            updated_at DATETIME DEFAULT (datetime('now', 'localtime'))
        );

        CREATE TABLE IF NOT EXISTS sender_master (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT UNIQUE NOT NULL,
            total_count   INTEGER NOT NULL DEFAULT 0,
            deleted_count INTEGER NOT NULL DEFAULT 0,
            kept_count    INTEGER NOT NULL DEFAULT 0,
            moved_count   INTEGER NOT NULL DEFAULT 0,
            updated_at    DATETIME DEFAULT (datetime('now', 'localtime'))
        );

        CREATE TABLE IF NOT EXISTS pending_confirmations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            mail_id    TEXT NOT NULL UNIQUE,
            subject    TEXT,
            sender     TEXT,
            summary    TEXT,
            action     TEXT NOT NULL,
            folder     TEXT,
            status     TEXT NOT NULL DEFAULT 'waiting',
            created_at DATETIME DEFAULT (datetime('now', 'localtime')),
            updated_at DATETIME DEFAULT (datetime('now', 'localtime'))
        );

        CREATE TABLE IF NOT EXISTS experience_buffer (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            mail_id    TEXT NOT NULL,
            features   TEXT NOT NULL,
            action     TEXT NOT NULL,
            reward     INTEGER NOT NULL,
            created_at DATETIME DEFAULT (datetime('now', 'localtime'))
        );

        CREATE TABLE IF NOT EXISTS model_weights (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            task_type  TEXT NOT NULL,
            weights    BLOB NOT NULL,
            version    INTEGER NOT NULL DEFAULT 1,
            saved_at   DATETIME DEFAULT (datetime('now', 'localtime'))
        );

        CREATE TABLE IF NOT EXISTS system_state (
            key        TEXT PRIMARY KEY,
            value      TEXT,
            updated_at DATETIME DEFAULT (datetime('now', 'localtime'))
        );

        -- Bug A: mail_id の重複挿入を防ぐ（既存DBへの適用）
        CREATE UNIQUE INDEX IF NOT EXISTS idx_pending_mail_id
            ON pending_confirmations(mail_id);
    """)

    # trust_scores の重複行を削除してから UNIQUE インデックスを付与
    c.execute("""
        DELETE FROM trust_scores
        WHERE id NOT IN (
            SELECT MAX(id) FROM trust_scores GROUP BY task_type
        )
    """)
    c.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_trust_scores_task_type
            ON trust_scores(task_type)
    """)

    # 初期レコード
    c.execute("""
        INSERT OR IGNORE INTO trust_scores (task_type, score, mode)
        VALUES ('email', 0.0, 'confirm')
    """)
    c.execute("""
        INSERT OR IGNORE INTO system_state (key, value)
        VALUES ('last_run_at', NULL)
    """)

    conn.commit()
    conn.close()
    print(f"DB initialized: {DB_PATH}")


if __name__ == "__main__":
    initialize()
