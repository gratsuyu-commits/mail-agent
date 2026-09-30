"""Gmail API接続テスト（直近7日分のメールを最大5件表示）"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from modules.mail_fetcher import fetch_mails

mails = fetch_mails(since=None, initial_fetch_days=7)
print(f"\n取得件数: {len(mails)}\n")
for m in mails[:5]:
    print(f"  ID     : {m['mail_id']}")
    print(f"  件名   : {m['subject']}")
    print(f"  差出人 : {m['sender']}")
    print(f"  本文冒頭: {m['body'][:80].replace(chr(10), ' ')}")
    print()
