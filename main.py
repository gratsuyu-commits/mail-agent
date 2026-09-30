"""
メール判断AI メインエンジン
スケジューラから毎朝9時に呼び出される。
"""
import json
import logging
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from db.database import get_connection, initialize
from modules.mail_fetcher import fetch_mails
from modules.llm import extract_features
from modules.policy import PolicyNetwork
from modules.task_executor import execute_action
from modules import trust_score as ts
from webui.app import set_policy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "logs" / "app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    return yaml.safe_load((BASE_DIR / "config.yaml").read_text(encoding="utf-8"))


def get_last_run_at(conn) -> datetime | None:
    row = conn.execute("SELECT value FROM system_state WHERE key='last_run_at'").fetchone()
    if row and row["value"]:
        return datetime.fromisoformat(row["value"])
    return None


def update_last_run_at(conn):
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "UPDATE system_state SET value=?, updated_at=datetime('now','localtime') WHERE key='last_run_at'",
        (now,)
    )
    conn.commit()


def timeout_pending(conn):
    """前日以前のwaitingレコードをtimeoutに更新し、対応するexperience_bufferも削除する"""
    timed_out = conn.execute(
        """SELECT mail_id FROM pending_confirmations
           WHERE status='waiting' AND date(created_at) < date('now','localtime')"""
    ).fetchall()

    if timed_out:
        ids = [r["mail_id"] for r in timed_out]
        conn.execute(
            f"""UPDATE pending_confirmations
               SET status='timeout', updated_at=datetime('now','localtime')
               WHERE mail_id IN ({','.join('?' * len(ids))})""",
            ids
        )
        conn.execute(
            f"DELETE FROM experience_buffer WHERE mail_id IN ({','.join('?' * len(ids))}) AND reward=0",
            ids
        )
        logger.info(f"タイムアウト処理: {len(ids)}件")

    conn.commit()


def update_sender_total(conn, sender_email: str):
    """差出人の受信総数を+1する（なければ新規作成）"""
    conn.execute("INSERT OR IGNORE INTO sender_master (email) VALUES (?)", (sender_email,))
    conn.execute(
        "UPDATE sender_master SET total_count=total_count+1, updated_at=datetime('now','localtime') WHERE email=?",
        (sender_email,)
    )


def get_sender_history(conn, sender_email: str) -> dict:
    row = conn.execute(
        "SELECT total_count, deleted_count, kept_count, moved_count FROM sender_master WHERE email=?",
        (sender_email,)
    ).fetchone()
    if not row:
        return {"total": 1, "deleted": 0, "kept": 0, "moved": 0}
    return {
        "total":   row["total_count"],
        "deleted": row["deleted_count"],
        "kept":    row["kept_count"],
        "moved":   row["moved_count"],
    }


def get_relationship(total_count: int) -> str:
    if total_count >= 10:
        return "frequent"
    elif total_count >= 3:
        return "occasional"
    return "first_time"


def add_to_experience_buffer(conn, mail_id: str, features: list, action: str, reward: int, buffer_size: int):
    conn.execute(
        "INSERT INTO experience_buffer (mail_id, features, action, reward) VALUES (?, ?, ?, ?)",
        (mail_id, json.dumps(features), action, reward)
    )
    # 上限を超えた古いレコードを削除
    conn.execute(
        """DELETE FROM experience_buffer WHERE id IN (
               SELECT id FROM experience_buffer ORDER BY id ASC
               LIMIT MAX(0, (SELECT COUNT(*) FROM experience_buffer) - ?)
           )""",
        (buffer_size,)
    )


def run():
    config = load_config()
    initialize()

    conn = get_connection()

    # タイムアウト処理
    timeout_pending(conn)

    # 前回実行日時を取得してすぐ更新（クラッシュ時の再取得を防ぐ）
    last_run_at = get_last_run_at(conn)
    update_last_run_at(conn)

    # メール取得
    logger.info("メール取得開始")
    mails = fetch_mails(since=last_run_at, initial_fetch_days=config["mail"]["initial_fetch_days"])
    if not mails:
        logger.info("新着メールなし。終了します。")
        conn.close()
        return

    logger.info(f"{len(mails)}件のメールを処理します")

    # ポリシーNW初期化
    policy = PolicyNetwork(config["dqn"])
    set_policy(policy)

    mode = ts.get_current_mode()
    logger.info(f"現在のモード: {mode}")

    # フィードバック件数取得（ε計算用）
    fb_count_row = conn.execute(
        "SELECT COUNT(*) as cnt FROM learning_history WHERE task_type='email' AND reward IS NOT NULL"
    ).fetchone()
    policy.update_epsilon(fb_count_row["cnt"])

    for mail in mails:
        mail_id = mail["mail_id"]
        sender  = mail["sender"]
        subject = mail["subject"]

        # LLM特徴量抽出
        llm_result = extract_features(subject, sender, mail["body"])
        if llm_result is None:
            logger.warning(f"LLMスキップ: {mail_id}")
            continue

        # 差出人カウント更新（LLM成功後のみ）
        update_sender_total(conn, sender)
        conn.commit()

        # sender_history・relationship取得
        history = get_sender_history(conn, sender)
        relationship = get_relationship(history["total"])
        llm_result["relationship"] = relationship

        # 特徴ベクトル構築
        features = policy.build_feature_vector(llm_result, history)

        # 行動選択
        action = policy.select_action(features, autonomous=(mode == "autonomous"))

        # フォルダ移動の場合はLLMで移動先を決定
        folder = None
        if action == "move":
            folder = _select_folder(subject, llm_result.get("summary", ""), config.get("folders", []))

        if mode == "confirm":
            # 確認待ちテーブルに保存（featuresも一緒に持たせる）
            conn.execute(
                """INSERT OR IGNORE INTO pending_confirmations
                   (mail_id, subject, sender, summary, action, folder, status)
                   VALUES (?, ?, ?, ?, ?, ?, 'waiting')""",
                (mail_id, subject, sender, llm_result.get("summary", ""), action, folder)
            )
            # featuresをexperience_bufferに仮登録（reward=0、フィードバック時に正しいrewardで再登録）
            add_to_experience_buffer(conn, mail_id, features, action, 0, config["dqn"]["buffer_size"])
            conn.commit()
        else:
            # 自律モード：即時実行
            conn.execute(
                """INSERT OR IGNORE INTO pending_confirmations
                   (mail_id, subject, sender, summary, action, folder, status)
                   VALUES (?, ?, ?, ?, ?, ?, 'autonomous')""",
                (mail_id, subject, sender, llm_result.get("summary", ""), action, folder)
            )
            conn.execute(
                """INSERT INTO learning_history (mail_id, task_type, features, action, reward)
                   VALUES (?, 'email', ?, ?, NULL)""",
                (mail_id, json.dumps(features), action)
            )
            # sender_master更新
            col_map = {"delete": "deleted_count", "keep": "kept_count", "move": "moved_count"}
            col = col_map.get(action, "kept_count")
            conn.execute(
                f"UPDATE sender_master SET {col}={col}+1, updated_at=datetime('now','localtime') WHERE email=?",
                (sender,)
            )
            conn.commit()
            execute_action(mail_id, action, folder)

    policy.save_weights()
    conn.close()

    # WebUI起動
    _launch_webui()


def _select_folder(subject: str, summary: str, folders: list[str]) -> str:
    """LLMを使って移動先フォルダを選択する"""
    if not folders:
        return None
    from modules.llm import _get_client, get_model
    folder_list = "、".join(folders)
    prompt = f"以下のメールを最も適切なフォルダに振り分けてください。フォルダ名のみ返してください。\n件名：{subject}\n要約：{summary}\nフォルダ一覧：{folder_list}"
    try:
        resp = _get_client().chat.completions.create(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
            reasoning_effort="low",  # gpt-oss 系は推論トークンを消費するため最小にする
        )
        result = (resp.choices[0].message.content or "").strip().strip("「」\"'`* 。")
        if result in folders:
            return result
        # 余計な語が付いた場合は、返答に含まれるフォルダ名を採用する
        for f in folders:
            if f in result:
                return f
        logger.warning(f"フォルダ選択の返答が一覧に無いため先頭を使用: {result!r}")
        return folders[0]
    except Exception:
        return folders[0]


def _kill_port(port: int):
    """指定ポートを使用中のプロセスを終了する"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) != 0:
            return  # 使用されていない
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess"],
            capture_output=True, text=True
        )
        for pid in set(result.stdout.strip().splitlines()):
            if pid.strip().isdigit():
                subprocess.run(["taskkill", "/PID", pid.strip(), "/F"], capture_output=True)
                logger.info(f"ポート{port}のプロセス(PID={pid.strip()})を終了しました")
    except Exception as e:
        logger.warning(f"ポート解放失敗: {e}")


def _launch_webui():
    """FastAPIサーバーを別プロセスで起動してブラウザを開く"""
    import time
    _kill_port(8000)
    time.sleep(1)
    python = str(BASE_DIR / ".venv" / "Scripts" / "python.exe")
    subprocess.Popen([
        python, "-m", "uvicorn", "webui.app:app",
        "--host", "127.0.0.1", "--port", "8000"
    ], cwd=str(BASE_DIR))
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:8000")
    logger.info("WebUI起動: http://127.0.0.1:8000")


if __name__ == "__main__":
    run()
