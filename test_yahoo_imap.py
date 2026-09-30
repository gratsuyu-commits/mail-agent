"""
Yahoo! Mail IMAP接続テストスクリプト
実行前に .env ファイルに認証情報を設定してください。
"""
import imaplib
import os
from dotenv import load_dotenv

load_dotenv()

YAHOO_ADDRESS  = os.getenv("YAHOO_MAIL_ADDRESS")
YAHOO_PASSWORD = os.getenv("YAHOO_MAIL_PASSWORD")

IMAP_HOST = "imap.mail.yahoo.com"
IMAP_PORT = 993


def test_connection():
    if not YAHOO_ADDRESS or not YAHOO_PASSWORD:
        print("[ERROR] .env に YAHOO_MAIL_ADDRESS / YAHOO_MAIL_PASSWORD が設定されていません")
        return

    print(f"接続先: {IMAP_HOST}:{IMAP_PORT}")
    print(f"アカウント: {YAHOO_ADDRESS}")
    print("-" * 40)

    try:
        # SSL接続
        print("[1/3] SSL接続中...")
        mail = imaplib.IMAP4_SSL(host=IMAP_HOST, port=IMAP_PORT)
        print("      OK")

        # ログイン
        print("[2/3] ログイン中...")
        mail.login(YAHOO_ADDRESS, YAHOO_PASSWORD)
        print("      OK")

        # フォルダ一覧取得
        print("[3/3] フォルダ一覧取得中...")
        status, folders = mail.list()
        if status == "OK":
            print("      OK")
            print("\n--- 取得できたフォルダ ---")
            for f in folders:
                print(" ", f.decode())
        else:
            print(f"      FAILED: {folders}")

        mail.logout()
        print("\n[SUCCESS] Yahoo! Mail IMAPアクセス確認完了")

    except imaplib.IMAP4.error as e:
        print(f"\n[FAILED] IMAP認証エラー: {e}")
        print("ヒント: Yahoo!アカウントのセキュリティ設定で")
        print("  「IMAP/POP/SMTPアクセスを許可する」を有効にしているか確認してください。")
        print("  また、2段階認証が有効な場合はアプリパスワードを使用してください。")
    except ConnectionRefusedError:
        print(f"\n[FAILED] 接続拒否: {IMAP_HOST}:{IMAP_PORT} に接続できませんでした")
    except Exception as e:
        print(f"\n[FAILED] 予期しないエラー: {type(e).__name__}: {e}")


if __name__ == "__main__":
    test_connection()
