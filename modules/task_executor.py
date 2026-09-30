"""
Gmail上でのメール操作を実行するモジュール。
- delete: ゴミ箱へ移動
- keep:   何もしない（受信トレイに残す）
- move:   指定ラベルを付与して受信トレイから外す
"""
import logging

from googleapiclient.errors import HttpError

from modules.mail_fetcher import _get_service

logger = logging.getLogger(__name__)


def _get_or_create_label(service, label_name: str) -> str:
    """ラベルが存在しなければ作成してIDを返す"""
    result = service.users().labels().list(userId="me").execute()
    for label in result.get("labels", []):
        if label["name"] == label_name:
            return label["id"]

    created = service.users().labels().create(
        userId="me",
        body={"name": label_name, "labelListVisibility": "labelShow", "messageListVisibility": "show"}
    ).execute()
    logger.info(f"ラベル作成: {label_name} (id={created['id']})")
    return created["id"]


def execute_action(mail_id: str, action: str, folder: str | None = None) -> bool:
    """
    メールに対してアクションを実行する。
    成功時 True、失敗時 False を返す。
    """
    try:
        service = _get_service()

        if action == "delete":
            service.users().messages().trash(userId="me", id=mail_id).execute()
            logger.info(f"削除（ゴミ箱）: {mail_id}")

        elif action == "move" and folder:
            label_id = _get_or_create_label(service, folder)
            service.users().messages().modify(
                userId="me",
                id=mail_id,
                body={
                    "addLabelIds":    [label_id],
                    "removeLabelIds": ["INBOX"],
                }
            ).execute()
            logger.info(f"フォルダ移動: {mail_id} → {folder}")

        elif action == "keep":
            logger.info(f"保持（操作なし）: {mail_id}")

        return True

    except HttpError as e:
        logger.error(f"Gmail APIエラー: {mail_id} / {e}")
        return False
    except Exception as e:
        logger.error(f"実行エラー: {mail_id} / {e}")
        return False
