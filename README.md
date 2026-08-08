# movement-coach

輸入一段人體動作影片，用本地視覺語言模型判斷這是什麼動作、找出動作的問題、推論需要強化的環節，再從一個 1,324 筆的動作資料庫檢索出實際存在的訓練處方與作法。全程在本地執行，不呼叫雲端 API。

設計原則有兩條：**推理與接地分離**——模型可以自由推理，但最終輸出的訓練動作必須來自真實資料庫，不得由模型生成；**核心與介面分離**——核心功能是純 Python 模組，網頁介面只是其中一種呼叫方式。

> 狀態：規劃中

## Overview

- 研究問題：在沒有任何訓練資料的前提下，能否從單機 2D 影片產出可信的動作診斷與訓練處方？既有做法（監督式動作分類、手寫關節角度規則）都需要先有標註資料或逐動作的專家規則，換一個動作領域就得重來。
- 方法摘要：VLM 負責動作描述、問題評估與成因推論三段自然語言推理，過程中不受資料庫內容限制；推論結果經一層約束映射到 19 個可檢索的肌群詞彙；最後以貪婪集合覆蓋演算法從資料庫選出能覆蓋所有弱點的最小動作組合，並附上該動作的分步驟作法。
- 目前結論：端到端流程已可執行。以本機 `qwen2.5vl:7b` 對一段拳擊影片實測，四階段約 10 秒完成，產出的處方動作編號全數可在資料庫中驗證存在。該動作（上鉤拳）不在資料庫的 1,324 筆之內，流程仍完整產出診斷與處方，驗證了辨識與資料庫解耦的設計。輸出品質尚未評估。

詳細範圍與成功標準見 `docs/spec.md`；實作、演算法與驗證設計見 `docs/architecture.md`。

## Requirements

- Runtime / language：Python `>=3.10`（見 `pyproject.toml`），開發與測試用 conda 環境 `movement-coach`（Python 3.12）
- Package manager / build tool：pip（`pyproject.toml` + setuptools），開發環境用 conda 管理
- External services：本機 [Ollama](https://ollama.com) 服務，預設 `http://localhost:11434`，模型 `qwen2.5vl:7b`
- Hardware：NVIDIA GPU。實測於 RTX 3070 8GB
- Data：[exercises-dataset](https://github.com/hasaneyldrm/exercises-dataset) 的 `data/exercises.json`（1,324 筆，約 17 MB）

## Setup

```bash
conda create -y -n movement-coach python=3.12
conda activate movement-coach
pip install -e ".[api]" pytest
```

`[api]` 是選用相依（FastAPI/uvicorn）。只要當函式庫用的話 `pip install -e .` 即可，核心層不依賴任何 web 套件。

模型需事先備妥：`ollama pull qwen2.5vl:7b`。

資料取得步驟如下。

### 取得動作資料庫

上游 repository 會持續更新，本專案以 commit 固定版本，避免資料變動影響結果重現。

```bash
mkdir -p data
curl -L -o data/exercises.json \
  https://raw.githubusercontent.com/hasaneyldrm/exercises-dataset/7455efae41b330c265e7cd4b78dfa848e7ce5ebd/data/exercises.json
```

- 固定版本：`7455efae41b330c265e7cd4b78dfa848e7ce5ebd`（2026-07-16）
- 預期路徑：`data/exercises.json`
- SHA-256：`656634224b8977b99a6d765470ee123260d4979715eaa4e7c0b7c8bb0d79f93d`

驗證：

```bash
sha256sum -c <<< "656634224b8977b99a6d765470ee123260d4979715eaa4e7c0b7c8bb0d79f93d  data/exercises.json"
```

`data/` 不納入版本控制（見 `.gitignore`）。只使用 `exercises.json`，不下載該資料集的圖片與 GIF。

## Run

網頁介面：

```bash
uvicorn movement_coach.api:app --host 127.0.0.1 --port 8000
```

開啟 http://127.0.0.1:8000 。預設只綁定本機。

作為函式庫：

```python
from movement_coach import MovementCoach, format_report

coach = MovementCoach.from_path("data/exercises.json")
coach.check_ready()                       # 確認模型服務可用
print(format_report(coach.diagnose("squat.mp4", equipment=["body weight"])))
```

## Verify

```bash
pytest
```

105 個測試，全部不需要 GPU 或模型服務（VLM 以 stub 取代）。標記 `real_db` 的測試在 `data/exercises.json` 未下載時自動略過。

## Usage

網頁介面：

1. 上傳一段人體動作影片
2. 檢視系統判斷的動作描述，不正確時直接修改
3. 檢視問題診斷與弱點肌群（含系統無法對應到資料庫的項目）
4. 檢視訓練處方，展開查看每個動作的分步驟作法

第 2 步的修正機會是必要的：系統從影片取樣 6 張畫面判讀，取樣點未必落在關鍵動作上。

## Project Structure

- `src/movement_coach/dataset.py`：載入與驗證 `exercises.json`，正規化 `secondary_muscles`
- `src/movement_coach/muscles.py`：肌群詞彙正規化表（40+ 詞 → 19 個 `target`）
- `src/movement_coach/prescribe.py`：計分與貪婪集合覆蓋，含接地驗證 `verify_grounded`
- `src/movement_coach/video.py`：影片取樣為 base64 JPEG
- `src/movement_coach/vlm.py`：Ollama 客戶端與三段自由推理 + 一段約束映射
- `src/movement_coach/pipeline.py`：`MovementCoach` 進入點與文字報告
- `src/movement_coach/errors.py`：例外階層
- `src/movement_coach/api.py`：FastAPI 薄 adapter（唯一匯入 web 套件的模組）
- `web/index.html`：靜態前端，無 build step
- `tests/`：105 個測試
- `docs/spec.md`：需求、範圍與成功標準
- `docs/architecture.md`：系統、演算法及驗證設計

## Configuration

- `MOVEMENT_COACH_DATASET`：資料集路徑，預設 `data/exercises.json`。非敏感。
- `OllamaVLM(host=..., model=..., seed=..., temperature=...)`：模型服務位置與參數，預設 `http://localhost:11434` / `qwen2.5vl:7b` / `temperature=0.2`。非敏感。
- 上傳影片暫存於系統暫存目錄，逾 1 小時自動清除，不寫入 repository。

不得把密碼、token、私鑰、個資或正式資料寫入 repository。輸入影片可能包含可識別個人，不得提交。

## Known Limitations

以下限制來自對 `exercises.json` 的實際檢查（見 `docs/architecture.md` 的資料集稽核）：

- 資料庫只有健身動作，**武術、球類、日常動作為 0 筆**。因動作辨識已與資料庫解耦，這不影響診斷流程，但**處方內容仍受限**。
- 資料庫可推薦肌力與柔軟度訓練，**無法推薦技術訓練或動作控制練習**。
- 肌群標註稀疏：平均每個動作只標 2.95 條肌肉，且深度不隨動作複雜度擴展（barbell deadlift 只標 3 條）。無啟動程度權重、無穩定肌分類、無活動度資料。
- 欄位詞彙不一致：`secondary_muscles` 使用 40 個詞，只有 9 個能對回 `target` 的 19 個詞；`hip flexors` 出現 77 次但不存在對應的 `target`，無法直接檢索。
- `abductors` 僅 5 筆、`adductors` 僅 6 筆，髖外展與內收訓練覆蓋不足。
- 成因推論的因果正確性無從驗證，資料集也不含支撐因果所需的資料。輸出定位為訓練建議，非醫療診斷。
- 單機 2D 輸入下，含軀幹旋轉或身體離開拍攝平面的動作診斷不可靠。
- 系統只看影片中均勻取樣的 6 張畫面，取樣點可能錯過關鍵動作；實測拳擊影片時第一階段曾描述成「男子用毛巾擦臉」。動作描述的人工修正步驟因此是必要的，不是選配。
- 輸出品質（診斷是否穩定、建議是否適切）尚未評估。
- 動畫 GIF 為 180×180、約 12 格的 3D 渲染人偶，非真人影像，本專案不使用。
- 媒體檔（圖片與 GIF）版權屬 Gym visual，僅授權轉散布；MIT 只涵蓋程式碼與資料欄位。
