import logging

from db.database import get_connection

logger = logging.getLogger(__name__)

TASK_TYPE = "email"


def get_current_mode() -> str:
    conn = get_connection()
    row = conn.execute(
        "SELECT mode FROM trust_scores WHERE task_type=? ORDER BY id DESC LIMIT 1",
        (TASK_TYPE,)
    ).fetchone()
    conn.close()
    return row["mode"] if row else "confirm"


def update():
    """フィードバック1件ごとに信頼スコアを再計算して保存する。
    却下時はlearning_historyに2件記録されるが、1通=1フィードバックとして数えるため
    pending_confirmations の確認済み件数・承認件数を使う。
    """
    conn = get_connection()

    # 直近50件の確認済みメール（pending_confirmationsベース）
    confirmed = conn.execute(
        """SELECT pc.mail_id,
                  MAX(CASE WHEN lh.reward = -1 THEN 1 ELSE 0 END) AS was_rejected
           FROM pending_confirmations pc
           LEFT JOIN learning_history lh
               ON lh.mail_id = pc.mail_id AND lh.task_type = ?
           WHERE pc.status = 'confirmed'
           GROUP BY pc.mail_id
           ORDER BY pc.updated_at DESC
           LIMIT 50""",
        (TASK_TYPE,)
    ).fetchall()

    total = len(confirmed)
    if total < 50:
        score = 0.0
        mode  = "confirm"
    else:
        approvals = sum(1 for r in confirmed if not r["was_rejected"])
        score = approvals / 50
        mode  = "autonomous" if score >= 0.90 else "confirm"

    conn.execute(
        """
        UPDATE trust_scores SET score=?, mode=?, updated_at=datetime('now','localtime')
        WHERE task_type=?
        """,
        (score, mode, TASK_TYPE)
    )
    conn.commit()
    conn.close()
    logger.info(f"信頼スコア更新 score={score:.2f} mode={mode}")
    return score, mode


def get_score() -> float:
    conn = get_connection()
    row = conn.execute(
        "SELECT score FROM trust_scores WHERE task_type=? ORDER BY id DESC LIMIT 1",
        (TASK_TYPE,)
    ).fetchone()
    conn.close()
    return row["score"] if row else 0.0
