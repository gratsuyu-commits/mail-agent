import json
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import get_connection
from modules.task_executor import execute_action
from modules.policy import PolicyNetwork
from modules import trust_score as ts

logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"

# Python 3.14 の Jinja2 キャッシュ非互換を回避するためキャッシュ無効化
_jinja_env = Environment(
    loader=FileSystemLoader(str(BASE_DIR / "webui" / "templates")),
    cache_size=0,
)
TEMPLATES = Jinja2Templates(env=_jinja_env)

app = FastAPI()

_policy = None  # メインエンジンから注入する（別プロセス起動時はon_startupでロード）


def set_policy(policy):
    global _policy
    _policy = policy


def _get_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _get_policy() -> PolicyNetwork:
    global _policy
    if _policy is None:
        config = _get_config()
        _policy = PolicyNetwork(config["dqn"])
    return _policy


def _add_experience(conn, mail_id: str, features_json: str, action: str, reward: int):
    config = _get_config()
    buf_size = config["dqn"]["buffer_size"]
    conn.execute(
        "INSERT INTO experience_buffer (mail_id, features, action, reward) VALUES (?, ?, ?, ?)",
        (mail_id, features_json, action, reward)
    )
    conn.execute(
        """DELETE FROM experience_buffer WHERE id IN (
               SELECT id FROM experience_buffer ORDER BY id ASC
               LIMIT MAX(0, (SELECT COUNT(*) FROM experience_buffer) - ?)
           )""",
        (buf_size,)
    )


# ------------------------------------------------------------------
# 確認モード：一覧表示
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    conn = get_connection()
    mode  = ts.get_current_mode()
    score = ts.get_score()

    if mode == "autonomous":
        rows = conn.execute(
            """SELECT * FROM pending_confirmations
               WHERE status='autonomous' AND date(created_at)=date('now','localtime')
               ORDER BY id"""
        ).fetchall()
        conn.close()
        return TEMPLATES.TemplateResponse(request, "result.html", {
            "items": [dict(r) for r in rows],
            "score": score,
        })

    rows = conn.execute(
        "SELECT * FROM pending_confirmations WHERE status='waiting' ORDER BY id"
    ).fetchall()
    conn.close()
    return TEMPLATES.TemplateResponse(request, "confirm.html", {
        "items": [dict(r) for r in rows],
        "folders": _get_config().get("folders", []),
        "score": score,
        "total": len(rows),
    })


# ------------------------------------------------------------------
# 確認モード：1件承認
# ------------------------------------------------------------------

@app.post("/approve/{pending_id}")
async def approve(pending_id: int):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM pending_confirmations WHERE id=? AND status='waiting'", (pending_id,)
    ).fetchone()
    if not row:
        conn.close()
        return JSONResponse({"error": "not found or already processed"}, status_code=404)

    # experience_bufferの仮登録(reward=0)を正しいreward=1で上書き追加
    feat_row = conn.execute(
        "SELECT features FROM experience_buffer WHERE mail_id=? ORDER BY id DESC LIMIT 1",
        (row["mail_id"],)
    ).fetchone()
    feat = feat_row["features"] if feat_row else "[]"

    conn.execute(
        "UPDATE experience_buffer SET reward=1 WHERE mail_id=? AND reward=0",
        (row["mail_id"],)
    )
    conn.execute(
        "UPDATE pending_confirmations SET status='confirmed', updated_at=datetime('now','localtime') WHERE id=?",
        (pending_id,)
    )
    conn.execute(
        """INSERT INTO learning_history (mail_id, task_type, features, action, reward)
           VALUES (?, 'email', ?, ?, 1)""",
        (row["mail_id"], feat, row["action"])
    )
    _update_sender(conn, row["sender"], row["action"])
    conn.commit()
    conn.close()

    ts.update()
    policy = _get_policy()
    policy.train_step()
    policy.save_weights()

    execute_action(row["mail_id"], row["action"], row["folder"])
    return JSONResponse({"status": "approved"})


# ------------------------------------------------------------------
# 確認モード：1件却下 → 正解行動で上書き
# ------------------------------------------------------------------

@app.post("/reject/{pending_id}")
async def reject(pending_id: int, correct_action: str = Form(...), correct_folder: str = Form(None)):
    if correct_action not in ("delete", "keep", "move"):
        return JSONResponse({"error": "invalid action"}, status_code=400)

    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM pending_confirmations WHERE id=? AND status='waiting'", (pending_id,)
    ).fetchone()
    if not row:
        conn.close()
        return JSONResponse({"error": "not found or already processed"}, status_code=404)

    feat_row = conn.execute(
        "SELECT features FROM experience_buffer WHERE mail_id=? ORDER BY id DESC LIMIT 1",
        (row["mail_id"],)
    ).fetchone()

    # 仮登録(reward=0)を削除
    conn.execute("DELETE FROM experience_buffer WHERE mail_id=? AND reward=0", (row["mail_id"],))

    same_action = (row["action"] == correct_action)

    if feat_row:
        feat = feat_row["features"]
        if not same_action:
            # actionの種類が違う → 元の判断を -1 として記録
            _add_experience(conn, row["mail_id"], feat, row["action"], -1)
            conn.execute(
                """INSERT INTO learning_history (mail_id, task_type, features, action, reward)
                   VALUES (?, 'email', ?, ?, -1)""",
                (row["mail_id"], feat, row["action"])
            )
        # 正解行動を +1 として記録（フォルダだけの訂正でも +1 は追加）
        _add_experience(conn, row["mail_id"], feat, correct_action, 1)
        conn.execute(
            """INSERT INTO learning_history (mail_id, task_type, features, action, reward)
               VALUES (?, 'email', ?, ?, 1)""",
            (row["mail_id"], feat, correct_action)
        )
    else:
        logger.warning(f"experience_bufferに特徴量なし（DQN学習スキップ）: {row['mail_id']}")
    conn.execute(
        """UPDATE pending_confirmations
           SET status='confirmed', action=?, folder=?, updated_at=datetime('now','localtime')
           WHERE id=?""",
        (correct_action, correct_folder, pending_id)
    )
    _update_sender(conn, row["sender"], correct_action)
    conn.commit()
    conn.close()

    ts.update()
    policy = _get_policy()
    policy.train_step()
    policy.save_weights()

    execute_action(row["mail_id"], correct_action, correct_folder)
    return JSONResponse({"status": "rejected", "correct_action": correct_action})


# ------------------------------------------------------------------
# すべて承認して実行
# ------------------------------------------------------------------

@app.post("/approve_all")
async def approve_all():
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM pending_confirmations WHERE status='waiting'"
    ).fetchall()

    for row in rows:
        feat_row = conn.execute(
            "SELECT features FROM experience_buffer WHERE mail_id=? ORDER BY id DESC LIMIT 1",
            (row["mail_id"],)
        ).fetchone()
        feat = feat_row["features"] if feat_row else "[]"

        conn.execute("UPDATE experience_buffer SET reward=1 WHERE mail_id=? AND reward=0", (row["mail_id"],))
        conn.execute(
            "UPDATE pending_confirmations SET status='confirmed', updated_at=datetime('now','localtime') WHERE id=?",
            (row["id"],)
        )
        conn.execute(
            """INSERT INTO learning_history (mail_id, task_type, features, action, reward)
               VALUES (?, 'email', ?, ?, 1)""",
            (row["mail_id"], feat, row["action"])
        )
        _update_sender(conn, row["sender"], row["action"])

    conn.commit()
    conn.close()

    for row in rows:
        execute_action(row["mail_id"], row["action"], row["folder"])

    if rows:
        ts.update()
        policy = _get_policy()
        policy.train_step()
        policy.save_weights()

    return JSONResponse({"status": "all_approved", "count": len(rows)})


# ------------------------------------------------------------------
# ステータス確認API
# ------------------------------------------------------------------

@app.get("/status")
async def status():
    return JSONResponse({
        "mode":  ts.get_current_mode(),
        "score": ts.get_score(),
    })


# ------------------------------------------------------------------
# ヘルパー
# ------------------------------------------------------------------

def _update_sender(conn, sender_email: str, action: str):
    conn.execute(
        "INSERT OR IGNORE INTO sender_master (email) VALUES (?)", (sender_email,)
    )
    col_map = {"delete": "deleted_count", "keep": "kept_count", "move": "moved_count"}
    col = col_map.get(action, "kept_count")
    conn.execute(
        f"UPDATE sender_master SET {col}={col}+1, updated_at=datetime('now','localtime') WHERE email=?",
        (sender_email,)
    )
