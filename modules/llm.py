import json
import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)

_client = None
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
DEFAULT_MODEL = "openai/gpt-oss-20b"


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def get_model() -> str:
    """config.yaml の llm.model を返す（未設定なら DEFAULT_MODEL）"""
    try:
        cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        return (cfg.get("llm") or {}).get("model") or DEFAULT_MODEL
    except Exception as e:
        logger.warning(f"config.yaml 読み込み失敗（既定モデルを使用）: {e}")
        return DEFAULT_MODEL


PROMPT_TEMPLATE = """
以下のメールを分析し、JSON形式で返してください。JSONのみ出力し、説明文は不要です。

件名: {subject}
差出人: {sender}
本文: {body}

出力形式（キーはこの5つだけ。has_deadline と is_automated は引用符なしの真偽値 true / false で書く）:
{{
  "summary": "本文の要約（50文字以内）",
  "category": "newsletter | work | personal | ads | other のいずれか",
  "urgency": "high | medium | low のいずれか",
  "has_deadline": false,
  "is_automated": false
}}
""".strip()


def extract_features(subject: str, sender: str, body: str) -> dict | None:
    """
    メール情報をLLMに渡して特徴量を抽出する。
    失敗時は None を返す（呼び出し側でスキップ処理）。
    """
    prompt = PROMPT_TEMPLATE.format(subject=subject, sender=sender, body=body[:500])

    try:
        response = _get_client().chat.completions.create(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
            reasoning_effort="low",  # gpt-oss 系は推論トークンを消費するため最小にする
            response_format={"type": "json_object"},  # JSONモードで構文を保証する
        )
        raw = response.choices[0].message.content.strip()

        # JSON部分だけ抽出（コードブロックで囲まれている場合に対応）
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)

        # 必須キーの検証
        required = {"summary", "category", "urgency", "has_deadline", "is_automated"}
        if not required.issubset(result.keys()):
            logger.warning(f"LLMレスポンスにキー不足: {result}")
            return None

        # 値の正規化
        valid_categories = {"newsletter", "work", "personal", "ads", "other"}
        valid_urgencies  = {"high", "medium", "low"}
        if result["category"] not in valid_categories:
            result["category"] = "other"
        if result["urgency"] not in valid_urgencies:
            result["urgency"] = "medium"
        result["has_deadline"] = bool(result["has_deadline"])
        result["is_automated"] = bool(result["is_automated"])

        return result

    except json.JSONDecodeError as e:
        logger.error(f"LLM JSONパース失敗: {e} / raw={raw!r}")
        return None
    except Exception as e:
        logger.error(f"Groq APIエラー: {e}")
        return None
