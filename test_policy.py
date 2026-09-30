"""DQN ポリシーネットワーク動作テスト"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from modules.policy import PolicyNetwork

config = {
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 200,
    "batch_size": 32,
    "buffer_size": 1000,
}

policy = PolicyNetwork(config)

llm_result = {
    "category": "newsletter",
    "urgency": "low",
    "has_deadline": False,
    "is_automated": True,
    "relationship": "frequent",
}
sender_history = {"total": 20, "deleted": 18, "kept": 1, "moved": 1}

features = policy.build_feature_vector(llm_result, sender_history)
print(f"特徴ベクトル（{len(features)}次元）: {features}")

# ε=1.0 なのでランダム行動になる
action = policy.select_action(features)
print(f"選択された行動: {action}")

# εの減衰確認
policy.update_epsilon(100)
print(f"ε(フィードバック100件後): {policy.epsilon:.3f}")
policy.update_epsilon(200)
print(f"ε(フィードバック200件後): {policy.epsilon:.3f}")

print("\n[OK] DQNポリシーネットワーク正常動作")
