import base64
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
BASE_DIR = Path(__file__).parent.parent
CREDENTIALS_PATH = BASE_DIR / "credentials.json"
TOKEN_PATH = BASE_DIR / "token.json"


def _get_service():
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        refreshed = False
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                refreshed = True
            except Exception as e:
                # リフレッシュトークン失効（invalid_grant など）→ ブラウザで再認証する
                logger.warning(f"Gmailトークンの更新に失敗したため再認証します: {e}")
        if not refreshed:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=8080, open_browser=True)
        TOKEN_PATH.write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def _decode_body(payload: dict) -> str:
    """メール本文を再帰的に取得してデコードする"""
    if payload.get("body", {}).get("data"):
        data = payload["body"]["data"]
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain":
            data = part.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    for part in payload.get("parts", []):
        result = _decode_body(part)
        if result:
            return result

    return ""


def fetch_mails(since: datetime | None, initial_fetch_days: int = 7) -> list[dict]:
    """
    Gmailから未処理メールを取得する。
    since が None（初回）の場合は initial_fetch_days 日分を取得する。
    """
    service = _get_service()

    if since is None:
        after_dt = datetime.now() - timedelta(days=initial_fetch_days)
    else:
        after_dt = since

    query = f"in:inbox after:{int(after_dt.timestamp())}"
    logger.info(f"Gmail fetch query: {query}")

    messages = []
    page_token = None
    while True:
        kwargs = {"userId": "me", "q": query, "maxResults": 100}
        if page_token:
            kwargs["pageToken"] = page_token
        results = service.users().messages().list(**kwargs).execute()
        messages.extend(results.get("messages", []))
        page_token = results.get("nextPageToken")
        if not page_token:
            break

    mails = []
    for msg in messages:
        detail = service.users().messages().get(
            userId="me", id=msg["id"], format="full"
        ).execute()

        headers = {h["name"]: h["value"] for h in detail["payload"]["headers"]}
        subject = headers.get("Subject", "（件名なし）")
        raw_from = headers.get("From", "")
        match = re.search(r"<([^>]+)>", raw_from)
        sender = match.group(1).lower() if match else raw_from.strip().lower()
        body    = _decode_body(detail["payload"])[:500]

        mails.append({
            "mail_id": msg["id"],
            "subject": subject,
            "sender":  sender,
            "body":    body,
        })

    logger.info(f"取得メール数: {len(mails)}")
    return mails
