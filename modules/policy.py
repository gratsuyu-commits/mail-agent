import io
import json
import logging
import random

import torch
import torch.nn as nn
import torch.optim as optim

from db.database import get_connection

logger = logging.getLogger(__name__)

ACTIONS = ["delete", "keep", "move"]
STATE_DIM  = 16
ACTION_DIM = 3


class DQNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, ACTION_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNetwork:
    def __init__(self, config: dict):
        self.epsilon       = config["epsilon_start"]
        self.epsilon_end   = config["epsilon_end"]
        self.decay_steps   = config["epsilon_decay_steps"]
        self.batch_size    = config["batch_size"]
        self.buffer_size   = config["buffer_size"]

        self.model     = DQNNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

        self._load_weights()

    # ------------------------------------------------------------------
    # 特徴量ベクトル構築
    # ------------------------------------------------------------------

    def build_feature_vector(self, llm_result: dict, sender_history: dict) -> list[float]:
        category_map     = {"newsletter": 0, "work": 1, "personal": 2, "ads": 3, "other": 4}
        urgency_map      = {"high": 0, "medium": 1, "low": 2}
        relationship_map = {"frequent": 0, "occasional": 1, "first_time": 2}

        cat = [0.0] * 5
        cat[category_map.get(llm_result.get("category", "other"), 4)] = 1.0

        urg = [0.0] * 3
        urg[urgency_map.get(llm_result.get("urgency", "medium"), 1)] = 1.0

        rel = [0.0] * 3
        rel[relationship_map.get(llm_result.get("relationship", "first_time"), 2)] = 1.0

        has_deadline = [1.0 if llm_result.get("has_deadline") else 0.0]
        is_automated = [1.0 if llm_result.get("is_automated") else 0.0]

        total   = sender_history.get("total", 0)
        deleted = sender_history.get("deleted", 0)
        kept    = sender_history.get("kept", 0)
        past    = max(total - 1, 1)

        sender_deleted_rate = [deleted / past]
        sender_kept_rate    = [kept / past]
        sender_total_norm   = [min(total, 20) / 20]

        return cat + urg + rel + has_deadline + is_automated + sender_deleted_rate + sender_kept_rate + sender_total_norm

    # ------------------------------------------------------------------
    # 行動選択（ε-greedy）
    # ------------------------------------------------------------------

    def select_action(self, features: list[float], autonomous: bool = False) -> str:
        if not autonomous and random.random() < self.epsilon:
            return random.choice(ACTIONS)

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(x)
            action_idx = q_values.argmax().item()
        return ACTIONS[action_idx]

    # ------------------------------------------------------------------
    # εの更新
    # ------------------------------------------------------------------

    def update_epsilon(self, feedback_count: int):
        ratio = min(feedback_count / self.decay_steps, 1.0)
        self.epsilon = max(
            self.epsilon_end,
            1.0 - (1.0 - self.epsilon_end) * ratio,
        )

    # ------------------------------------------------------------------
    # DQN学習
    # ------------------------------------------------------------------

    def train_step(self) -> bool:
        conn = get_connection()
        rows = conn.execute(
            "SELECT features, action, reward FROM experience_buffer WHERE reward != 0 ORDER BY RANDOM() LIMIT ?",
            (self.batch_size,)
        ).fetchall()
        conn.close()

        # 特徴量の次元が不正なレコードを除外（DBが壊れた場合のクラッシュ防止）
        rows = [r for r in rows if len(json.loads(r["features"])) == STATE_DIM]

        if len(rows) < self.batch_size:
            return False

        self.model.train()
        states  = torch.tensor([json.loads(r["features"]) for r in rows], dtype=torch.float32)
        rewards = torch.tensor([r["reward"] for r in rows], dtype=torch.float32)
        actions = [ACTIONS.index(r["action"]) for r in rows]

        q_all     = self.model(states)
        q_selected = q_all[range(self.batch_size), actions]
        loss = self.criterion(q_selected, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        logger.info(f"DQN更新 loss={loss.item():.4f}")
        return True

    # ------------------------------------------------------------------
    # 重みの保存・読み込み
    # ------------------------------------------------------------------

    def save_weights(self):
        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        weights_blob = buf.getvalue()

        conn = get_connection()
        last = conn.execute(
            "SELECT version FROM model_weights WHERE task_type='email' ORDER BY version DESC LIMIT 1"
        ).fetchone()
        version = (last["version"] + 1) if last else 1

        conn.execute(
            "INSERT INTO model_weights (task_type, weights, version) VALUES (?, ?, ?)",
            ("email", weights_blob, version)
        )
        # 最新1件だけ残して古いバージョンを削除
        conn.execute(
            """DELETE FROM model_weights
               WHERE task_type='email' AND version < (
                   SELECT MAX(version) FROM model_weights WHERE task_type='email'
               )"""
        )
        conn.commit()
        conn.close()
        logger.info(f"DQN重み保存 version={version}")

    def _load_weights(self):
        conn = get_connection()
        row = conn.execute(
            "SELECT weights FROM model_weights WHERE task_type='email' ORDER BY version DESC LIMIT 1"
        ).fetchone()
        conn.close()

        if row:
            try:
                buf = io.BytesIO(row["weights"])
                self.model.load_state_dict(torch.load(buf, weights_only=True))
                logger.info("DQN重みをDBから復元しました")
            except Exception as e:
                logger.warning(f"DQN重みの読み込み失敗（初期値で起動します）: {e}")
