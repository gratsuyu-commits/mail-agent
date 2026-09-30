"""Groq API 特徴量抽出テスト"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from modules.llm import extract_features

samples = [
    {
        "subject": "【重要】請求書送付のご案内",
        "sender": "billing@example.com",
        "body": "お世話になっております。今月分の請求書をお送りします。支払期限は2026年5月10日です。",
    },
    {
        "subject": "週刊ニュースレター Vol.42",
        "sender": "newsletter@news.example.com",
        "body": "今週のテックニュースをお届けします。AIの最新動向、クラウド市場の変化など盛りだくさんです。",
    },
]

for s in samples:
    print(f"件名: {s['subject']}")
    result = extract_features(s["subject"], s["sender"], s["body"])
    if result:
        for k, v in result.items():
            print(f"  {k}: {v}")
    else:
        print("  [FAILED] 特徴量抽出失敗")
    print()
