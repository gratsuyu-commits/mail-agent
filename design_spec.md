# メール判断AI 設計仕様書

## 1. システム概要

ユーザーの判断基準・価値観をRLで学習し、ユーザーの代わりにタスクを実行するエージェント型AI。
言語理解にオープンソースLLMを活用し、その上にDQNベースのポリシーネットワークを乗せることで、フィードバックを通じて「ユーザーらしい判断」を育てていく。確認・フィードバックはPC上のローカルWebUIで完結する。

---

## 2. システム全体構成

```
┌─────────────────────────────────────────────────┐
│                  データソース層                    │
│  Gmail (公式API)  ファイルシステム  スケジューラ  │
└───────────────────────┬─────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────┐
│                  共通エンジン層                    │
│                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │    LLM       │→│  ポリシーNW   │→│ 信頼スコア│ │
│  │ (Groq API)  │  │    (DQN)     │  │   管理   │ │
│  └─────────────┘  └──────────────┘  └──────────┘ │
│                          │                        │
│              ┌───────────┴──────────┐             │
│              ▼                      ▼             │
│       確認モード                自律モード          │
└───────┬───────────────────────────┬──────────────┘
        │                           │
┌───────▼──────────┐    ┌───────────▼──────────────┐
│    タスク層       │    │      インターフェース層      │
│  メール管理タスク  │    │   ローカルWebUI             │
│  ファイル整理タスク│    │  （FastAPI + ブラウザ）      │
│  （将来拡張可能） │    └───────────────────────────┘
└──────────────────┘
        │
┌───────▼──────────┐
│    データ層       │
│    SQLite DB     │
└──────────────────┘
```

---

## 3. コンポーネント詳細

### 3.1 スケジューラ

- 定期的にメインエンジンを起動する
- 設定：毎朝9時に実行（Windowsタスクスケジューラ）
- `StartWhenAvailable=true`：9時以降にPCが起動した場合は起動直後に実行する
- `DisallowStartIfOnBatteries=false`：バッテリー駆動中も実行する
- 将来的には常駐型に移行

### 3.2 メール取得モジュール

| サービス | 接続方式 | 取得内容 |
|---------|---------|---------|
| Gmail | 公式API（OAuth2） | 受信トレイのメール一覧、件名・差出人・本文 |

- 取得対象は **受信トレイのみ**（クエリに `in:inbox` を付与。送信済み・スパム・下書きは除外）
- 取得範囲：前回実行日時以降に受信したメールのみを対象とする（重複処理を防ぐため）
- ページネーション対応：`nextPageToken` を使って100件超のメールも全件取得する
- 最終実行日時はDBの `system_state` テーブルに保存し、次回起動時に参照する
- `last_run_at` はメール取得開始時に更新する（処理途中でクラッシュした場合の再取得を防ぐため）
- 初回起動時はconfig.yamlの `initial_fetch_days` で指定した日数分を取得する
- **差出人の正規化**：Fromヘッダーが `"表示名 <address@example.com>"` 形式の場合、`<>` 内のメールアドレス部分のみを小文字で抽出して保存する。`<>` がない場合はFromヘッダー全体を小文字にして使う
- 本文は text/plain の部分を優先して再帰的に探してデコードし、先頭500文字だけを使う
- 件名がない場合は「（件名なし）」とする

### 3.3 LLM（Groq API）

- モデル：config.yaml の `llm.model` で指定（現在 `openai/gpt-oss-20b`、Groq API経由）。旧 `llama-3.1-8b-instant` は 2026年9月時点で Groq から廃止済み
- 役割：メールの内容を理解し、構造化された特徴量を生成する
- 入出力はJSON形式
- パラメータ：temperature 0.1、最大512トークン、`reasoning_effort="low"`（gpt-oss 系の推論トークン消費を抑える）、JSONモード（`response_format={"type": "json_object"}`）で構文を保証
- 本文は先頭500文字だけを渡し、summary は50文字以内で生成させる
- 応答がコードブロック（```）で囲まれている場合は、中身だけを取り出してからJSONとして読む
- `has_deadline` / `is_automated` は真偽値に変換する
- LLMのレスポンスが失敗・JSON不正・必須キー不足の場合は該当メールをスキップする
- `category` / `urgency` の値が定義外の場合はスキップせず `other` / `medium` にフォールバックする
- **差出人カウントの更新（`sender_master.total_count +1`）はLLM成功後に行う**（スキップ時に誤カウントされるのを防ぐため）

### 3.4 ポリシーネットワーク（DQN）

- 役割：LLMが抽出した特徴量を受け取り、行動を決定する
- アルゴリズム：DQN（Deep Q-Network）アーキテクチャを採用
- 将来的にはPPOへの移行も検討

> **設計上の注意：Contextual Bandit としての実装**
>
> 各メールの判断は互いに独立しており（メールAの削除がメールBに影響しない）、報酬もユーザーの即時フィードバックで完結する。そのため本実装ではベルマン方程式（`Q(s,a) = r + γ * max Q(s', a')`）を用いず、Q値を即時報酬 `r` に直接近づける学習（`loss = MSE(Q(s,a), r)`）を行っている。これは厳密な逐次意思決定（MDP）ではなく **Contextual Bandit** 問題に相当し、この用途においては理論上問題ない。DQNのネットワーク構造と経験再生バッファのみを活用している形になる。

**学習方式：経験再生バッファ（Experience Replay）**

- フィードバックを受けるたびに経験バッファに追加する
- バッファが32件以上溜まったら、ランダムに32件サンプリングしてDQNを更新する
- バッファの最大保持件数：1000件（古いものから上書き）
- **学習対象は `reward != 0` のエントリのみ**（`reward=0` は未評価の仮登録であり学習に使わない）
- 最適化手法は Adam（学習率 0.001）、損失関数は平均二乗誤差（MSE）
- 1回のフィードバックにつき、ミニバッチ1回分だけ更新する
- サンプリングしたレコードのうち、特徴量が16次元でないものは除外する（DB破損時のクラッシュ防止）。除外の結果32件に満たなくなった場合は更新しない

```
フィードバック受信 → 経験バッファに追加（最大1000件）
                        ↓
              バッファのうち reward != 0 が32件以上溜まったら
                        ↓
          ランダムに32件サンプリングしてDQN更新
```

**入力特徴量（16次元ベクトル）**

テキストの意味理解はLLMが担うため、DQNにはLLMが構造化した数値ベクトルのみを渡す。
summaryはWebUIの表示用としてのみ使用し、DQNには渡さない。

| 特徴量 | エンコード方式 | 次元数 |
|-------|-------------|-------|
| category（newsletter/work/personal/ads/other） | one-hot | 5 |
| urgency（high/medium/low） | one-hot | 3 |
| relationship（frequent/occasional/first_time） | one-hot | 3 |
| has_deadline | 0/1 | 1 |
| is_automated | 0/1 | 1 |
| sender_deleted_rate（過去の削除率） | 実数（0.0〜1.0） | 1 |
| sender_kept_rate（過去の保持率） | 実数（0.0〜1.0） | 1 |
| sender_total_norm（受信総数を正規化、上限20通） | 実数（0.0〜1.0） | 1 |
| **合計** | | **16** |

※ `sender_deleted_rate` と `sender_kept_rate` の分母は `total_count - 1`（現在処理中のメールを除く過去の受信数）を使用する。初回受信時（total_count=1）は両値とも `0.0` とする。

```python
past_count = max(total_count - 1, 1)  # ゼロ除算防止
sender_deleted_rate = deleted_count / past_count
sender_kept_rate    = kept_count    / past_count
```

**入力ベクトルの例**
```
category=newsletter → [1,0,0,0,0]
urgency=low         → [0,0,1]
relationship=frequent → [1,0,0]
has_deadline=false  → [0]
is_automated=true   → [1]
sender_deleted_rate → [0.9]
sender_kept_rate    → [0.1]
sender_total_norm   → [0.2]

最終ベクトル：[1,0,0,0,0, 0,0,1, 1,0,0, 0, 1, 0.9, 0.1, 0.2]
```

**探索戦略：ε-greedy**

学習初期にDQNが偏った行動（例：全てのメールを削除）を選び続けるのを防ぐため、ε-greedy による探索を行う。

```
初期：ε = 1.0（100%ランダムに行動を選択）
      ↓ フィードバックが蓄積されるにつれてεを減衰
最終：ε = 0.05（95%は学習した行動、5%はランダム探索）
```

- εの減衰はフィードバック件数に応じて線形に行う
- フィードバック件数は `learning_history` のうち reward が NULL でない件数を使う（却下で2件記録された場合は2件と数える）。メインエンジン起動時に1回だけ計算する
- 減衰式：`ε = max(ε_end, 1.0 - (1.0 - ε_end) × (フィードバック件数 / decay_steps))`
- 初期値はconfig.yamlで管理（epsilon_start: 1.0 / epsilon_end: 0.05 / epsilon_decay_steps: 200）
- 確認モード中はユーザーの確認が安全網になるため、ランダム行動が実害になることはない
- 自律モード移行後は ε = 0 に固定し、完全にDQNの学習した判断に従う（確認なしで実行されるためランダム行動を排除する）

**ネットワーク構造**
```
入力層（16次元）→ 中間層（64ユニット）→ 中間層（32ユニット）→ 出力層（3次元）
```

**出力（Q値・行動の選択肢）**
- 削除（ゴミ箱へ移動）
- 保持（受信トレイに残す）
- フォルダ移動（指定フォルダへ振り分け）

Q値が最も高い行動を選択する。

**DQN重みの保存**
- フィードバックのたびに `model_weights` テーブルに保存する
- 常に最新バージョン1件のみ保持し、古いバージョンは保存時に自動削除する

### 3.5 信頼スコア管理

- タスク種別ごとにスコアを管理する
- 自律モード移行後も実行結果をWebUIで報告

**計算式**

```
信頼スコア = 直近50件のメール確認のうち承認件数 / 50
例）直近50件中45件承認 → 信頼スコア = 0.90
```

- 「1通のメール確認 = 1フィードバック」として数える
- 却下は1回の不正解として数える（learning_historyに2件記録されるが、スコア計算では `pending_confirmations` を基準に `mail_id` でグループ化して1件扱い）
- 閾値：0.90以上で自律モードに移行
- 直近N件を使うことで「最近の成長」が反映されやすい設計
- フィードバックが50件に満たない初期は確認モードを維持する（スコアは0.0として保存）
- 信頼スコアの再計算は、承認・却下・一括承認のときだけ行う（一括承認では全件処理後に1回）

**現状の実装についての補足**

- 閾値0.90と直近50件は、現在 `modules/trust_score.py` に直接書かれており、config.yaml の `trust_score`（threshold / window_size）の値は読み込んでいない
- 却下かどうかは、そのメールの `learning_history` に reward=-1 の記録があるかどうかで判定している。フォルダだけを訂正した却下は -1 を記録しないため、スコア計算上は承認と同じ扱いになる
- 自律モードに移行するとフィードバックが発生しないため、スコアは再計算されず、自動で確認モードに戻る仕組みはない

### 3.6 ローカルWebUI（FastAPI + ブラウザ）

- FastAPIでローカルサーバーを起動し、ブラウザで操作する（localhost:8000）
- 外部サーバー・ngrok不要。PC上で完結する

**サーバー起動の詳細**

- 待ち受けアドレスは 127.0.0.1:8000
- 起動前に、ポート8000を使っているプロセスがあれば終了させる（PowerShell の `Get-NetTCPConnection` でプロセスIDを調べ、`taskkill` で終了）。前日のサーバーが残っていても新しいサーバーに置き換わる
- `.venv\Scripts\python.exe -m uvicorn webui.app:app` を別プロセスで起動し、2秒待ってからブラウザで http://127.0.0.1:8000 を開く
- WebUIは別プロセスのため、メインエンジンのポリシーNWは引き継がない。最初に必要になった時点で、DBの最新のDQN重みを読み込む
- 画面テンプレートは Jinja2 のキャッシュを無効にして読み込む（Python 3.14 と Jinja2 のキャッシュの非互換を避けるため）

**起動・終了フロー**
```
スケジューラがメール処理・判断生成を完了
    ↓
FastAPIサーバーを別プロセスとして起動・ブラウザを自動オープン（localhost:8000）
    ↓
スケジューラは終了（FastAPIは継続稼働）
    ↓
├─ 確認モード：ユーザーが好きなタイミングでWebUIを確認・操作
│              → 1件承認・却下するたびに DQN更新 → DQN重みをmodel_weightsテーブルに保存
│
└─ 自律モード：即時実行 → 実行結果をWebUIに表示
```

**確認モード時の画面**

```
【本日の判断一覧】
信頼スコア：〇〇%（自律モード移行まで90%が必要）［バー表示］
確認待ち：〇件

No.1 差出人：xxxx@example.com
     件名：〇〇〇〇
     要約：〇〇〇〇〇〇〇〇...
     判断：削除
     [✅ 承認]  [❌ 却下 → 正解を選ぶ]

No.2 件名：〇〇〇〇
     ...

[すべて承認して実行]
```

- 各メールには本文ではなく、LLMが作った要約を表示する
- 判断は「削除」「保持」「フォルダ移動（フォルダ名）」を色分けしたラベルで表示する
- 却下を押すと、正しい操作（保持 / 削除 / フォルダ移動）を選ぶ欄が開く。初期選択は「保持」。フォルダ移動を選ぶと、config.yaml のフォルダ一覧から移動先を選ぶ欄が出る
- 承認・却下が終わったメールは薄く表示され、「完了」と出る
- 「すべて承認して実行」は確認ダイアログを出してから実行し、処理件数を表示して画面を再読み込みする

**自律モード時の画面**

```
【本日の実行結果】

削除：5件
保持：3件
フォルダ移動：2件

[詳細を見る]
```

- 「詳細を見る」を開くと、各メールの行動（フォルダ移動の場合は移動先）・件名・差出人を一覧表示する

- 自律モードで実行したメールは `pending_confirmations` に `status='autonomous'` で保存する
- WebUIは `status='autonomous' AND date(created_at)=date('now','localtime')` で当日分を取得して表示する

**二重送信防止**

- 承認・却下エンドポイントは `status='waiting'` のレコードのみ処理する
- すでに `confirmed` のレコードに対するリクエストは 404 を返して無視する

**却下時の正解行動バリデーション**

- `correct_action` は `delete / keep / move` のいずれかのみ受け付ける
- 不正な値の場合は 400 を返す

**タイムアウト設計**

- WebUIを開いて確認するのはユーザーの任意のタイミング
- 確認しないまま翌日スケジューラが起動した場合、前日の未確認分は `timeout` 扱いにする
- タイムアウト時は対応する `experience_buffer` の `reward=0` エントリも合わせて削除する
- DQN更新は行わない

### 3.7 タスク実行エンジン

- 共通エンジンにプラグイン形式でタスクを追加する設計
- 初期実装タスク：メール管理

**フォルダ移動の宛先設計**

- DQNは「削除/保持/フォルダ移動」の3択のみを担当する
- 「フォルダ移動」が選ばれた場合、LLMがメール内容を見て移動先フォルダを決定する
- フォルダ一覧はユーザーが設定ファイル（config.yaml）で管理する。変更は即時反映される
- Gmail上にラベルが存在しない場合は自動作成する
- 移動先の選択は Groq API（`llm.model`、temperature 0.0、最大256トークン、`reasoning_effort="low"`）に件名・要約・フォルダ一覧を渡し、フォルダ名だけを返させる。返答に余計な語が付いた場合は、返答に含まれるフォルダ名を採用する
- LLMの返したフォルダ名が一覧にない場合や、APIエラーの場合は、一覧の先頭のフォルダ（現在は「仕事」）にする
- フォルダ一覧が空の場合は移動先なし（null）となり、Gmail上では何もしない

**Gmail上の操作**

| 行動 | Gmail上の操作 |
|-----|-------------|
| 削除（delete） | ゴミ箱へ移動する |
| 保持（keep） | 何もしない（受信トレイに残す） |
| フォルダ移動（move） | 移動先のラベルを付け、受信トレイ（INBOX）ラベルを外す |

```yaml
# 現在のフォルダ設定
folders:
  - 仕事
  - プライベート
  - 領収書
  - スニーカー
  - ライブ
  - 野球
  - claude
  - KSK
```

**フォルダ移動時のフロー**
```
DQNが「フォルダ移動」を選択
    ↓
LLMが設定ファイルのフォルダ一覧から最適なフォルダを選択
    ↓
確認モード：WebUIに「〇〇フォルダに移動」と表示
    ↓
承認 → 実行
却下 → WebUIでフォルダを選び直す → 正解フォルダを実行 → DQN更新（バッファ32件以上の場合）
```

※ フォルダの選択（LLM担当）とアクション種別の選択（DQN担当）は独立している。`experience_buffer` はアクション種別（delete/keep/move）のみを学習対象とし、フォルダ情報は保持しない。

### 3.8 SQLite データベース

- 接続時は `check_same_thread=False` を指定する（FastAPIの非同期処理との共存のため）
- 後述のデータベース設計参照

---

## 4. API / インターフェース設計

### 4.1 ① スケジューラ → メインエンジン

```
トリガー：Windowsタスクスケジューラ（毎朝9時、StartWhenAvailable）
内容：起動シグナル（引数なし）
```

### 4.2 ② メール取得 → LLM

**リクエスト**
```json
{
  "subject": "件名",
  "sender": "差出人メールアドレス（正規化済み）",
  "body": "本文（先頭500文字程度）"
}
```

**レスポンス**
```json
{
  "summary": "本文の要約",
  "category": "newsletter | work | personal | ads | other",
  "urgency": "high | medium | low",
  "has_deadline": true,
  "is_automated": false
}
```

※ `relationship` はLLMでは判断できないため、sender_masterテーブルのtotal_countから以下のルールで算出する。

```python
if total_count >= 10:
    relationship = "frequent"
elif total_count >= 3:
    relationship = "occasional"
else:
    relationship = "first_time"
```

※ `sender_total_norm` は上限20通として正規化する。

```python
sender_total_norm = min(total_count, 20) / 20
# 例）5通 → 0.25、10通 → 0.50、20通以上 → 1.00
```

### 4.3 ③ LLM → ポリシーNW

```json
{
  "mail_id": "一意のID",
  "subject": "件名",
  "sender": "差出人メールアドレス",
  "summary": "本文の要約",
  "category": "newsletter",
  "urgency": "low",
  "has_deadline": false,
  "is_automated": true,
  "relationship": "frequent",
  "sender_history": {
    "total": 20,
    "deleted": 18,
    "kept": 1,
    "moved": 1
  }
}
```

※ `relationship` と `sender_history` はLLMのレスポンスではなく、sender_masterテーブルから取得・算出した値をここに追加する。

### 4.4 ④ ポリシーNW → WebUI

**WebUIに表示するデータ形式**
```json
{
  "mail_id": "一意のID",
  "subject": "件名",
  "sender": "差出人メールアドレス",
  "summary": "本文の要約",
  "action": "delete | keep | move",
  "folder": "仕事 | null"
}
```

### 4.5 ⑤ WebUIフィードバック → DQN

**承認時**
```json
{
  "mail_id": "一意のID",
  "action": "delete | keep | move",
  "folder": null,
  "reward": 1
}
```

**却下時（アクション種別が異なる場合）**
- 元の判断：reward: -1
- 正解行動：reward: +1
- 両方を `experience_buffer` と `learning_history` に記録する

**却下時（アクション種別が同じ・フォルダのみ違う場合）**
- 正解行動：reward: +1 のみ記録する
- 元の判断に reward: -1 は記録しない（同一の特徴量・同一アクションに矛盾した報酬が入るのを防ぐため）

### 4.6 WebUIのエンドポイント一覧

| メソッド・パス | 内容 |
|--------------|------|
| GET `/` | 確認モードなら確認画面（confirm.html）、自律モードなら実行結果画面（result.html）を返す |
| POST `/approve/{pending_id}` | 1件承認する。`pending_id` は `pending_confirmations` の id |
| POST `/reject/{pending_id}` | 1件却下する。フォームで `correct_action`（必須）と `correct_folder`（フォルダ移動の場合）を送る |
| POST `/approve_all` | 確認待ちを全件承認して実行する。処理件数を返す |
| GET `/status` | 現在のモードと信頼スコアを JSON で返す |

**各エンドポイントの処理順**

- 承認・却下：DB更新（experience_buffer・pending_confirmations・learning_history・sender_master）→ 信頼スコア更新 → DQN更新 → DQN重み保存 → Gmail操作
- 一括承認：全件のDB更新 → 全件のGmail操作 → 信頼スコア更新・DQN更新・DQN重み保存を1回ずつ

---

## 5. データベース設計

### 5.1 学習履歴テーブル（learning_history）

**役割**：全フィードバック履歴を永続的に記録する。件数制限なし。信頼スコア計算・ε減衰カウント・デバッグに使用する。

| カラム名 | 型 | 説明 |
|---------|---|------|
| id | INTEGER PRIMARY KEY | 一意のID |
| mail_id | TEXT | メールの一意ID |
| task_type | TEXT | タスク種別（email） |
| features | TEXT | 入力特徴量（JSON文字列） |
| action | TEXT | その経験における行動 |
| reward | INTEGER | フィードバック（1 / -1 / NULL） |
| created_at | DATETIME | 記録日時 |

※ reward は確認モードでのフィードバック値（1 / -1）を記録する。自律モード実行時はフィードバックがないため NULL とする。
※ 却下時でアクション種別が異なる場合のみ2件記録する（元の判断: reward:-1、正解行動: reward:+1）。アクション種別が同じ（フォルダのみ違う）場合は正解行動の1件のみ記録する。

### 5.2 信頼スコアテーブル（trust_scores）

| カラム名 | 型 | 説明 |
|---------|---|------|
| id | INTEGER PRIMARY KEY | 一意のID |
| task_type | TEXT UNIQUE | タスク種別（UNIQUE制約あり） |
| score | REAL | 信頼スコア（0.0〜1.0） |
| mode | TEXT | 現在のモード（confirm / autonomous） |
| updated_at | DATETIME | 更新日時 |

※ `task_type` に UNIQUE インデックスを付与し、行が増え続けないようにする。`INSERT OR IGNORE` で初期レコードを1件のみ保証する。

### 5.3 差出人マスタテーブル（sender_master）

| カラム名 | 型 | 説明 |
|---------|---|------|
| id | INTEGER PRIMARY KEY | 一意のID |
| email | TEXT UNIQUE | メールアドレス（正規化済み、小文字） |
| total_count | INTEGER | 受信総数（LLM成功後に更新） |
| deleted_count | INTEGER | 削除した数（実行後に更新） |
| kept_count | INTEGER | 保持した数（実行後に更新） |
| moved_count | INTEGER | 移動した数（実行後に更新） |
| updated_at | DATETIME | 更新日時 |

### 5.4 確認待ちテーブル（pending_confirmations）

**役割**：AIが判断したが、まだユーザーが確認していない項目を管理する。WebUIの表示に必要な情報も保持する。

| カラム名 | 型 | 説明 |
|---------|---|------|
| id | INTEGER PRIMARY KEY | 一意のID |
| mail_id | TEXT UNIQUE | メールの一意ID（UNIQUE制約あり） |
| subject | TEXT | 件名 |
| sender | TEXT | 差出人メールアドレス |
| summary | TEXT | 本文要約（LLM生成） |
| action | TEXT | AIが判断した行動 |
| folder | TEXT | フォルダ移動の場合の移動先（nullable） |
| status | TEXT | waiting / confirmed / timeout / autonomous |
| created_at | DATETIME | 判断を生成した時刻 |
| updated_at | DATETIME | 更新日時 |

※ `mail_id` に UNIQUE 制約を付与し、同一メールの重複登録を防ぐ。`INSERT OR IGNORE` により再実行時も安全に処理できる。

**タイムアウト処理フロー**
```
スケジューラが起動
    ↓
pending_confirmationsをチェック
    ↓
created_atが前日以前 かつ status=waiting のレコードを検索
    ↓
該当レコードのstatusをtimeoutに更新（DQN更新なし）
    ↓
対応する experience_buffer の reward=0 エントリも削除する
```

### 5.5 経験バッファテーブル（experience_buffer）

**役割**：DQN更新専用の作業用テーブル。直近1000件のみ保持し、古いものから上書きする。learning_historyとは目的が異なる。

| カラム名 | 型 | 説明 |
|---------|---|------|
| id | INTEGER PRIMARY KEY | 一意のID |
| mail_id | TEXT | メールの一意ID |
| features | TEXT | 16次元入力ベクトル（JSON文字列） |
| action | TEXT | 実行した行動 |
| reward | INTEGER | フィードバック（1 / -1 / 0） |
| created_at | DATETIME | 記録日時 |

- 確認待ち中は `reward=0` の仮エントリとして登録し、承認・却下後に正しい値に更新または置換する
- **DQN学習には `reward != 0` のエントリのみ使用する**（`reward=0` の仮エントリは学習対象外）
- 最大1000件を保持し、超えた場合は古いものから削除する
- プロセス再起動後もバッファが維持され、学習が蓄積される
- DQN更新時はこのテーブルからランダムに32件サンプリングする

### 5.6 DQN重みテーブル（model_weights）

| カラム名 | 型 | 説明 |
|---------|---|------|
| id | INTEGER PRIMARY KEY | 一意のID |
| task_type | TEXT | タスク種別 |
| weights | BLOB | シリアライズされたモデルパラメータ |
| version | INTEGER | バージョン番号 |
| saved_at | DATETIME | 保存日時 |

※ 保存のたびにバージョンを +1 し、最新バージョン1件のみ保持する（古いバージョンは自動削除）。

### 5.7 システム状態テーブル（system_state）

**役割**：システム全体の状態を管理する。最終実行日時など、起動をまたいで保持が必要な値を格納する。

| カラム名 | 型 | 説明 |
|---------|---|------|
| key | TEXT PRIMARY KEY | 状態キー |
| value | TEXT | 状態値 |
| updated_at | DATETIME | 更新日時 |

主なレコード例：
```
key: "last_run_at",  value: "2026-04-30 09:00:00"
```

---

## 6. 学習フロー

```
1. スケジューラが起動
2. pending_confirmationsをチェック → 前日以前のwaitingをtimeout処理
   → 対応するexperience_bufferのreward=0エントリも削除する
3. system_stateから last_run_at を取得 → system_state の last_run_at を現在日時で更新
   → 受信トレイ（in:inbox）から前回実行日時以降のメールを全件取得（ページネーション対応）
   （初回起動時はconfig.yamlのinitial_fetch_daysで指定した日数分を取得）
   → 0件の場合はそのまま終了
4. 各メールに対してLLMで特徴量を抽出（category / urgency / has_deadline / is_automated / summary）
   → LLM失敗時はそのメールをスキップ
   → LLM成功後に sender_master の total_count を +1
5. DBのsender_masterから差出人の過去行動履歴を取得（sender_history・relationship算出）
6. ポリシーNWが行動を決定（4 + 5を結合した16次元ベクトルを入力）
7. 信頼スコアを確認
   ├─ 確認モード → 判断結果をpending_confirmationsに保存（mail_id UNIQUE、重複はIGNORE）
   │               → experience_bufferにreward=0で仮登録
   │               → （全件処理後）DQN重みを保存し、WebUIを起動してブラウザを自動オープン
   │     ユーザーが確認
   │     ├─ 承認 → experience_bufferのreward=0を+1に更新
   │     │         → learning_historyに記録（reward:1）
   │     │         → sender_master更新（実行行動のカウント+1）
   │     │         → pending_confirmations更新（confirmed）
   │     │         → 信頼スコア更新 → DQN更新（reward!=0が32件以上の場合）→ DQN重み保存 → 実行
   │     │
   │     └─ 却下 → experience_bufferのreward=0を削除
   │                → アクション種別が異なる場合：元の判断をreward:-1で追加
   │                → 正解行動をreward:+1で追加
   │                → learning_historyに記録（アクション異なる場合のみ-1を追加、+1は常に追加）
   │                → sender_master更新（正解行動のカウント+1）
   │                → pending_confirmations更新（confirmed、actionとfolderを正解で上書き）
   │                → 信頼スコア更新 → DQN更新（reward!=0が32件以上の場合）→ DQN重み保存 → 正解行動を実行
   │
   └─ 自律モード → pending_confirmationsに保存（status='autonomous'）
                  → learning_historyに記録（reward:NULL）→ sender_master更新 → 即時実行
                  → 全件処理後にDQN重みをmodel_weightsに保存 → WebUIで結果表示（当日のautonomousレコードを表示）

8. 確認モードの場合：承認・却下・一括承認のたびに DQN重みを model_weights に保存する
```

### 却下時のWebUI表示例
```
「削除」は違うんですね。正しい操作を選んでください：

[保持]  [フォルダ移動 ▼]
```

---

## 7. 自律性の成長フロー

```
初期状態
  └─ 全アクションで確認モード
        ↓ フィードバックを積み重ねる
  信頼スコアが閾値（0.90）を超える（直近50件中45件以上承認）
        ↓
  該当タスクが自律モードに移行
        ↓
  確認なしで実行 → WebUIで結果表示
```

- タスク種別ごとに独立してスコアを管理
- 自律モード移行後も実行結果をWebUIで確認可能
- 将来的には常駐型に移行し、イベント駆動で動作

---

## 8. セキュリティ・設定管理

**APIキー管理**

- APIキーは環境変数（`.env`ファイル）で管理し、コードには直接書かない
- `.env`ファイルはGitの`.gitignore`に追加してリポジトリに含めない

```
# .env（Gitに含めない）
GROQ_API_KEY=xxxxx
```

- `.env.example` には現在 Yahoo!メール用の項目（YAHOO_MAIL_ADDRESS / YAHOO_MAIL_PASSWORD）だけが書かれており、GROQ_API_KEY は含まれていない
- `.gitignore` の対象：.env、*.db、logs/、__pycache__/、*.pyc、token.json、credentials.json

**Gmail の認証情報**

- credentials.json（OAuthクライアント情報）と token.json（取得したトークン）で管理する
- token.json の期限が切れている場合はリフレッシュトークンで自動更新する。無い場合や更新に失敗した場合（リフレッシュトークン失効 `invalid_grant` など）は、ブラウザを開いて認証をやり直す（ローカルのポート8080で受け取る）

**設定ファイル**

- フォルダ一覧・閾値などのパラメータは`config.yaml`で管理する（ただし信頼スコアの閾値と件数は、現状コード側の値が使われる。§3.5参照）
- `config.yaml`はGitに含めてOK
- フォルダ一覧はWebUIサーバー再起動なしで即時反映される

```yaml
# config.yaml（現在の設定）
trust_score:
  threshold: 0.90
  window_size: 50

mail:
  initial_fetch_days: 7  # 初回起動時に取得する過去メールの日数

llm:
  model: openai/gpt-oss-20b  # Groq で利用するモデル名

dqn:
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay_steps: 200  # 200件のフィードバックでε=0.05に到達
  batch_size: 32             # DQN更新のミニバッチサイズ
  buffer_size: 1000          # 経験バッファの最大保持件数

folders:
  - 仕事
  - プライベート
  - 領収書
  - スニーカー
  - ライブ
  - 野球
  - claude
  - KSK
```

---

## 9. 技術スタック

| 領域 | 技術 |
|-----|-----|
| 言語 | Python 3.14（uv で管理、仮想環境 `.venv`） |
| LLM | Groq API（config.yaml の `llm.model`、現在 openai/gpt-oss-20b） |
| RL | DQN（PyTorch） |
| メール取得 | Gmail API（OAuth2、`gmail.modify` スコープ） |
| WebUI | FastAPI + Jinja2（ローカル、localhost:8000） |
| DB | SQLite（`check_same_thread=False`） |
| スケジューラ | Windowsタスクスケジューラ（毎朝9時、StartWhenAvailable） |
| ログ | Python標準の logging（logs/app.log と標準出力、INFOレベル） |
| 依存パッケージ | requirements.txt（python-dotenv、pyyaml、google-api-python-client、google-auth-httplib2、google-auth-oauthlib、groq、torch、fastapi、uvicorn、jinja2） |

- Yahoo!メール（IMAP）は接続テスト（test_yahoo_imap.py）のみで、本体にはまだ組み込まれていない

---

## 10. エラーハンドリング方針

| エラー種別 | 挙動 |
|-----------|------|
| Groq API タイムアウト・エラー | 対象メールをスキップし、次のメールへ進む。スキップしたmail_idをログに記録する |
| Gmail API 認証エラー | トークン期限切れは自動更新し、更新に失敗した場合（リフレッシュトークン失効を含む）はブラウザで再認証を求める。それ以外の例外は捕捉しておらず、処理はそのまま異常終了する（last_run_at は取得前に更新済みのため、その回の取得範囲は次回も取得されない） |
| Gmail 操作（削除・移動）の失敗 | ログに記録する。呼び出し側は成否を見ていないため、DB上は確認済み・実行済みのままになる |
| フォルダ選択の LLM 失敗 | 一覧の先頭のフォルダにする |
| LLM レスポンスの JSON パース失敗 | 対象メールをスキップし、スキップしたmail_idをログに記録する |
| DQN 更新中の例外 | 方針：DQN更新をスキップし、バッファはそのまま保持する（次回更新で再試行）。現状の実装では例外を捕捉しておらず、承認・却下のDB更新は確定済みのまま、DQN重みの保存とGmail操作が行われずにエラー応答になる |
| WebUI への二重送信 | `status='waiting'` のレコードのみ処理し、確認済みレコードへのリクエストは 404 を返す |
| 不正な correct_action | `delete / keep / move` 以外の値は 400 を返す |

- エラーが発生してもシステム全体は止めず、処理可能なメールは処理を続ける
- エラーログはファイルに書き出し、後からデバッグできるようにする

---

## 11. ファイル構成

```
self_ai/
├── main.py                 メインエンジン（スケジューラから起動。取得→判断→保存→WebUI起動）
├── config.yaml             設定（信頼スコア・取得日数・DQN・フォルダ一覧）
├── requirements.md         要件のたたき台
├── design_spec.md          本書
├── requirements.txt        依存パッケージ
├── .env / .env.example     APIキーなど（.env はGitに含めない）
├── credentials.json        Gmail OAuthクライアント情報
├── token.json              Gmail アクセストークン
├── self_ai.db              SQLite データベース
├── modules/
│   ├── mail_fetcher.py     Gmail からのメール取得・認証
│   ├── llm.py              Groq API による特徴量抽出
│   ├── policy.py           ポリシーNW（DQN）・特徴ベクトル構築・学習・重み保存
│   ├── task_executor.py    Gmail 上の操作（ゴミ箱・ラベル付け）
│   └── trust_score.py      信頼スコアの計算とモード判定
├── db/
│   └── database.py         DB接続とテーブル作成・初期レコード投入
├── webui/
│   ├── app.py              FastAPI（確認・承認・却下・一括承認・状態確認）
│   └── templates/
│       ├── confirm.html    確認モード画面
│       └── result.html     自律モードの実行結果画面
├── logs/app.log            実行ログ
├── test_gmail.py           Gmail 接続テスト
├── test_llm.py             LLM 特徴量抽出のテスト
├── test_policy.py          ポリシーNWのテスト
└── test_yahoo_imap.py      Yahoo!メール（IMAP）接続テスト
```

---

*作成日：2026-04-27*
*更新日：2026-04-29（実装に基づき全面更新）*
*更新日：2026-05-06（以下の変更を反映）*
- 自律モードの実行結果を `pending_confirmations`（`status='autonomous'`）に保存する仕様を追加（§3.6、§5.4、§6）
- WebUI自律モード時は `status='autonomous' AND date(created_at)=当日` で表示する旨を明記（§3.6）
- LLMの `category` / `urgency` が定義外の値の場合はスキップせず `other` / `medium` にフォールバックする挙動を追記（§3.3）
- ポリシーNWがベルマン方程式を使わないContextual Bandit実装である旨を補足（§3.4）

*更新日：2026-09-30（ソースコードとの照合結果を反映）*
- フォルダ一覧から「副業」を削除（config.yaml に合わせる）（§3.7、§8）
- DQN重みの保存タイミングを「全件確認完了時」から「承認・却下のたび」に修正（§3.6、§6）
- 承認・却下時の処理順を実装に合わせて修正（DQN更新と重み保存の後にGmail操作）（§4.6、§6）
- 確認画面は本文ではなく要約を表示する旨など、画面の説明を実装に合わせて追記（§3.6）
- WebUIサーバーの起動手順（ポート解放・別プロセス起動）を追記（§3.6）
- WebUIのエンドポイント一覧を追加（§4.6）
- LLMのパラメータ、フォルダ選択のフォールバック、Gmail上の操作を追記（§3.3、§3.7）
- DQNの最適化手法・学習率、ε計算に使う件数の数え方を追記（§3.4）
- 信頼スコアについて、閾値がconfig.yamlから読まれていない点などの現状を補足（§3.5）
- Gmail認証、.env.example、.gitignore の内容を追記（§8）
- エラーハンドリング表を実装に合わせて更新・追加（§10）
- ファイル構成を追加（§11）

*更新日：2026-09-30（運用再開に伴う修正）*
- Groq の `llama-3.1-8b-instant` 廃止に伴い、モデル名を config.yaml の `llm.model` で管理する方式に変更（現在 `openai/gpt-oss-20b`）（§3.3、§3.7、§8、§9）
- gpt-oss 系向けに `reasoning_effort="low"`・JSONモードを追加し、max_tokens を引き上げ（§3.3、§3.7）
- Gmail のリフレッシュトークン失効時にブラウザ再認証へフォールバックするよう変更（§8、§10）
