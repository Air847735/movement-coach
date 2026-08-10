# Architecture and Design

本文件回答「如何實作與驗證」，保存目前採用的系統設計、演算法、測試方法與重要取捨。需求與成功標準以 `spec.md` 為準。

## Overview

- System / approach：四階段管線。前三階段由本地 VLM 進行自然語言推理（描述 → 診斷 → 成因），第四階段以確定性演算法從資料庫檢索處方。兩者之間以一層約束映射銜接，確保自由推理的結論能落回可檢索的詞彙空間。本質是 RAG：推理自由，輸出接地。
- 分層：核心邏輯為純 Python 模組，不依賴任何介面技術；FastAPI 為薄 adapter，只做 HTTP 轉接；前端頁面呼叫 API。核心可在無介面環境下獨立使用。
- Primary language / runtime：Python（版本待確認）
- Data / external boundary：本機 VLM 推論服務（待確認）；`exercises.json` 為唯一動作知識來源；輸入影片為本機檔案。

## Repository Map

分層以 import 方向強制：核心層任何模組都不得匯入 `api.py` 或 web 套件，由
`tests/test_pipeline.py::test_core_package_does_not_pull_in_a_web_framework`
在子行程中檢查。FastAPI 與 uvicorn 是 `pyproject.toml` 的選用相依 `[api]`，
只安裝核心相依時整套函式庫仍可運作。

核心層：

- `src/movement_coach/pipeline.py`：`MovementCoach` 進入點；`describe_movement`（階段一）與
  `diagnose`/`prescribe_for`（階段二至四）刻意分開，讓使用者在中間修正動作描述。
- `src/movement_coach/dataset.py`：`load_exercises` 驗證並索引資料集；`Exercise`、`ExerciseDatabase`。
- `src/movement_coach/muscles.py`：`TARGET_MUSCLES`、`ALIASES`、`normalize`、`normalize_all`。
- `src/movement_coach/prescribe.py`：`score`、`covers`、`prescribe`、`verify_grounded`。
- `src/movement_coach/video.py`：`sample_frames`。
- `src/movement_coach/vlm.py`：`OllamaVLM`、`Assessment` 與回覆解析。
- `src/movement_coach/errors.py`：例外階層。

介面層：

- `src/movement_coach/api.py`：FastAPI 應用，唯一匯入 web 套件的模組。
- `web/index.html`：單檔靜態前端，無 build step、無外部資源。

測試：`tests/`，105 個測試，全部以 stub 取代模型服務，不需要 GPU。

## Components and Responsibilities

- **動作描述**：影片 → 自然語言描述（例如「一個負重深蹲」）。**不與資料庫比對，不受 1,324 個標籤限制。** 輸出交由使用者確認或修正。
- **問題評估**：影片 + 確認後的動作描述 → 自然語言問題診斷。
- **成因推論**：診斷 → 自由推論需強化的環節。允許輸出資料庫詞彙以外的概念（如「踝背屈受限」）。
- **約束映射層**：自由推論結果 → 19 個 `target` 詞彙的子集。映射不到者列為「無對應」，不得以近似詞替代。
- **處方檢索**：弱點肌群集合 + 器材限制 → 最小動作組合 + `instruction_steps`。
- **服務層**：暴露分析與處方兩組端點，承接使用者修正動作描述的中斷點。負責影片大小與格式的前置驗證。
- **前端層**：上傳影片、顯示與修正動作描述、顯示診斷與處方。

### 失敗處理原則

- `exercises.json` 的存在性與 schema 在服務啟動時驗證，不延後到檢索階段。
- VLM 呼叫失敗時，回報失敗發生在哪一階段（描述 / 評估 / 成因），不以部分結果冒充完整輸出。
- 前三階段成功但處方檢索失敗時，仍回傳診斷結果並標明處方缺失原因。
- 影片格式與大小在進入推論前驗證，避免耗費 GPU 資源後才失敗。

## Interfaces and Data Flow

1. 前端上傳影片，服務層呼叫核心層的動作描述功能，回傳自然語言描述。
2. 使用者確認或修正描述後送回，服務層以影片與最終描述呼叫評估與推論。
3. 推論結果經約束映射轉為肌群集合，呼叫處方檢索。
4. 服務層回傳診斷、弱點肌群（含無對應項）與處方（含 `instruction_steps`），前端分區顯示。

- Interface：
  - 核心層：`MovementCoach.describe_movement(video)`、`.diagnose(video, description=, equipment=, max_items=)`、
    `.prescribe_for(assessment, ...)`、`.check_ready()`；`format_report(diagnosis)` 產生文字報告。
  - 服務層：`GET /api/health`、`GET /api/equipment`、`POST /api/describe`（回傳 `token` 與描述）、
    `POST /api/diagnose`（以 `token` 續作）、`DELETE /api/upload/{token}`。
- Data model / state：`Assessment`（自由推理結果）→ `Diagnosis`（含 `weak_muscles`、`unmapped_causes`、
  `prescription`）。上傳影片存於系統暫存目錄 `movement-coach-uploads/`，以隨機 token 命名，
  每次新上傳時清除逾 `UPLOAD_TTL_SECONDS`（3600 秒）的舊檔。token 只接受單一檔名，
  含路徑分隔字元者一律拒絕。

## Algorithm Design

### Problem Definition

- Input：弱點肌群集合 `W`（19 個 `target` 詞彙的子集）、可用器材集合 `E`、動作資料庫 `D`（1,324 筆）。
- Output：動作序列 `P ⊆ D`，覆蓋 `W` 中盡可能多的肌群，且 `|P|` 最小。
- Objective：最小化處方動作數，同時最大化弱點覆蓋。

### Assumptions and Invariants

- `target` 詞彙為 19 個封閉集合，可直接比對。
- `secondary_muscles` 詞彙為 40 個，需經正規化表映射後才能比對（只有 9 個原生可對上）。
- 處方中每個動作必須存在於 `D` 中且可由 `id` 驗證。

### 肌群正規化表

本專案唯一需要人工維護的知識資產。用途有兩個：把資料集 `secondary_muscles` 的 40 個詞正規化到 `target` 的 19 個詞；把 VLM 自由推論的輸出映射到同一組詞彙。目前存放於本文件，實作後應移為獨立資料檔並加上覆蓋率測試。

**原生可對上（9 個，不需映射）**：`biceps`、`calves`、`forearms`、`glutes`、`hamstrings`、`lats`、`traps`、`triceps`、`upper back`

**需映射（22 個）**：

| 來源詞 | → `target` | 來源詞 | → `target` |
|---|---|---|---|
| `quadriceps` | `quads` | `abdominals` | `abs` |
| `shoulders` | `delts` | `core` | `abs` |
| `deltoids` | `delts` | `lower abs` | `abs` |
| `rear deltoids` | `delts` | `obliques` | `abs` |
| `chest` | `pectorals` | `latissimus dorsi` | `lats` |
| `upper chest` | `pectorals` | `trapezius` | `traps` |
| `rhomboids` | `upper back` | `back` | `upper back` |
| `lower back` | `spine` | `inner thighs` | `adductors` |
| `groin` | `adductors` | `soleus` | `calves` |
| `brachialis` | `biceps` | `grip muscles` | `forearms` |
| `wrist flexors` | `forearms` | `wrist extensors` | `forearms` |

**無對應（9 個，必須明確標記而非硬湊）**：`hip flexors`(77)、`ankles`(11)、`feet`(8)、`rotator cuff`(6)、`ankle stabilizers`(4)、`wrists`(3)、`sternocleidomastoid`(2)、`hands`(2)、`shins`(1)。括號為在資料集中的出現次數。

**覆蓋率（2026-08-09 實測）**：31/40 個詞可映射；按出現次數計為 2467/2581 = **95.6%**。未覆蓋的 114 次中，`hip flexors` 佔 77 次。

VLM 自由推論的輸出還會產生資料集中不存在的詞（如 `erector spinae`、`踝背屈受限`）。此類詞的映射規則需在實作時擴充本表，映射不到者一律標記為「無對應」。

### Approach

1. 將 `W` 與各動作的 `secondary_muscles` 經正規化表轉為統一詞彙。
2. 計分：`target` 命中得 1.0，`secondary_muscles` 命中得 0.5（權重待調整，見 Known Gaps）。
3. 貪婪集合覆蓋：每輪選出「對剩餘未覆蓋肌群貢獻最多、分數最高」的動作，從剩餘集合中移除其覆蓋項。
4. 直到覆蓋完成、達到動作數上限，或無動作能再貢獻覆蓋時終止。

### Correctness

- 貪婪集合覆蓋為經典近似演算法，近似比為 `H(n)`（調和級數），不保證最小解。對本問題（`|W| ≤ 19`，動作數上限個位數）此近似程度可接受。
- 未證明：計分權重（1.0 / 0.5）是否反映真實訓練效果。此為啟發式，無資料支持。

### Complexity and Practical Limits

- Time：`O(k · |D| · m)`，`k` 為處方動作數上限，`m` 為單一動作的肌肉標註數（平均 2.95）。實測資料規模下可忽略。
- Space：`O(|D|)`，資料集約 17 MB。
- Practical limit：資料集規模固定為 1,324 筆。
- Precision / randomness：檢索層為確定性演算法，相同輸入必得相同處方。VLM 階段具隨機性，實作時應可指定 seed。

### Edge Cases

- `W` 為空集合：不輸出處方。
- `W` 中的肌群無任何動作以其為 `target`（如 `hip flexors`）：退回以 `secondary_muscles` 檢索；仍無結果則列為「無對應動作」。
- 器材過濾後候選集為空：放寬器材限制並明確告知，或回報無法產出處方。
- 動作描述完全不在資料庫涵蓋範圍（如武術動作）：不影響流程，處方仍可產出，僅限肌力與柔軟度訓練。
- 處方可能包含與待改善動作同類型的動作（例如深蹲問題被推薦另一種深蹲）。已知且允許，理由見「不排除同動作模式」設計決策。

## Verification and Experiments

### Strategy

- Unit：正規化表映射、計分函式、覆蓋計算、集合覆蓋終止條件。
- Integration / system：端到端流程；處方動作的 `id` 全數可在 `exercises.json` 中驗證；核心層在不啟動 FastAPI 的情況下可完整呼叫。
- Static checks：待確認。
- Experiment / benchmark：模組尚未實作，不預先設計實驗、指標算法、樣本數與合格門檻。待端到端可執行後，依實際輸出決定是否需要，以及怎麼量。可能的方向記於下方「後續評估方向」。

### Commands

```bash
pytest                                                    # 105 個測試
pytest -m real_db                                         # 只跑依賴真實資料集的斷言
uvicorn movement_coach.api:app --host 127.0.0.1 --port 8000
```

### Data and Environment

- Dataset：[exercises-dataset](https://github.com/hasaneyldrm/exercises-dataset) `data/exercises.json`，1,324 筆。版本固定於 commit `7455efae41b330c265e7cd4b78dfa848e7ce5ebd`（2026-07-16），SHA-256 `656634224b8977b99a6d765470ee123260d4979715eaa4e7c0b7c8bb0d79f93d`，取得指令見 `README.md`。程式碼與資料為 MIT；媒體檔版權屬 Gym visual，僅授權轉散布，本專案不使用媒體檔。資料集不納入版本控制。
- Environment：Ubuntu、Linux 6.17；conda 環境 `movement-coach`（Python 3.12.13）；
  Ollama 於 Docker 容器提供 `qwen2.5vl:7b`；NVIDIA RTX 3070 8GB。
- Baseline：尚未需要。
- Metrics：目前只有一項可自動檢查——處方動作可由 `id` 在 `exercises.json` 中驗證存在。其餘待依實際輸出再定。
- Reproducibility：檢索層為確定性演算法；VLM 階段可透過 `OllamaVLM(seed=...)` 指定 seed，
  但跨模型或伺服器版本的重現性不保證。測試套件不依賴模型服務。

### 後續評估方向

模組可執行後再考慮，目前不設計、不排程、不訂門檻：

- 診斷結論在重複執行下是否穩定。
- 成因推論的輸出有多少比例能映射到可檢索的肌群。
- 與「VLM 直接生成訓練建議、不接地」的輸出比較差異。

### Critical Cases

- [x] 正常案例：四階段完整執行並產出接地處方（`test_prescribe_for_maps_causes_and_grounds_the_result`，另有真實模型實跑）。
- [x] 邊界案例：資料庫外的動作，流程完整產出診斷與處方（`test_movement_outside_the_database_still_produces_a_prescription`；真實模型以拳擊影片實跑）。
- [x] 邊界案例：診斷結論無法映射時明確標記無對應而非硬湊（`test_unmappable_causes_are_reported_verbatim`）。
- [x] 邊界案例：核心層在無 FastAPI 環境下由腳本呼叫，功能完整（`test_core_package_does_not_pull_in_a_web_framework`，於子行程執行）。
- [x] 邊界案例：VLM 服務未啟動、影片解碼失敗、`exercises.json` 缺失，皆有明確回報且無靜默失敗（`test_vlm.py`、`test_video.py`、`test_dataset.py`）。
- [x] 邊界案例：處方含資料庫不存在的 id 時 `verify_grounded` 拒絕（`test_verify_grounded_rejects_an_unknown_id`）。
- [ ] 介面案例：網頁完整操作流程尚未以自動化測試覆蓋；已手動驗證（見下）。

### Verification Status

2026-08-09，conda 環境 `movement-coach`（Python 3.12.13）：

- `pytest`：**passed**，105 passed in 0.37s。涵蓋詞彙正規化、資料集驗證、集合覆蓋、
  影片取樣、VLM 回覆解析與傳輸失敗、pipeline 接地、分層檢查。
- 資料集稽核：**passed**，對 `exercises.json` 1,324 筆實際統計，結果見下節。
- 正規化表覆蓋率：**passed**，31/40 個詞可映射，按出現次數計 95.6%。已由
  `test_secondary_muscle_coverage_over_real_dataset` 設為 ≥95% 的回歸門檻。
- 端到端實跑（真實 `qwen2.5vl:7b`）：**passed**。對一段拳擊影片執行四階段約 10 秒完成，
  產出 2 個處方動作（`1775`、`1685`），編號皆可在資料庫中驗證存在。該動作不在資料集內，
  流程未中止。
- HTTP 介面實跑：**passed**（手動）。`GET /api/health`、`GET /api/equipment`、
  `POST /api/describe`（1.1 MB mp4 上傳）、`POST /api/diagnose`（含使用者修正描述、器材過濾）
  皆回傳預期結果；`GET /` 回傳前端頁面。
- 前端頁面：**not run**（無瀏覽器自動化測試；僅確認 HTTP 200 與內容型別）。
- Format / lint / type check：**not run**（專案尚未設定這些工具）。
- 打包（`python -m build`）：**not run**。

不得把未實際執行的檢查記為通過。

不得把未實際執行的檢查記為通過。

### 資料集稽核結果（2026-08-09 實測）

- 有效欄位僅 4 個：`name`、`equipment`、`target`、`secondary_muscles`。`category` 與 `body_part` 完全相同（1324/1324），`muscle_group` 完全等於 `secondary_muscles[0]`（1324/1324），兩者為冗餘欄位。
- `target` 為 19 個封閉詞彙；`equipment` 28 種；`body_part` 10 種。
- `(body_part, equipment, target)` 只有 181 種組合，平均 7.31 個動作共用一組，僅 59 個動作（4.5%）能由此三欄唯一確定。
- 名稱高度近似（Jaccard ≥ 0.8）的配對有 110 對。
- 肌肉標註平均 2.95 條；`barbell full squat` 5 條、`barbell deadlift` 僅 3 條，深度不隨動作複雜度擴展。
- `secondary_muscles` 使用 40 個詞，僅 9 個能對回 `target`。高頻對不上者：`shoulders`(400)、`quadriceps`(161)、`core`(94)、`chest`(91)、`hip flexors`(77)。
- 武術相關動作 0 筆（`martial`、`punch`、`agility` 皆為 0；`kick` 的 17 筆全為 kickback 類）。
- `abductors` 僅 5 筆、`adductors` 僅 6 筆，髖外展與內收訓練覆蓋不足。
- 動畫 GIF 為 180×180、約 12 格的 3D 渲染人偶。

## Design Decisions and Trade-offs

### 推理與接地分離

- Status：accepted
- Context：VLM 直接生成訓練動作會產生不存在的動作名稱與作法，無法驗證也無法執行。
- Decision：VLM 只負責推理，輸出的訓練動作一律從 `exercises.json` 檢索。
- Alternatives：全部交給 VLM 生成；全部用手寫規則。
- Consequences：處方 100% 可驗證且自帶作法；代價是受限於資料庫內容，無法推薦技術訓練。

### 動作辨識採 top-5 候選 + 使用者確認

- Status：**superseded**（由「動作辨識與資料庫解耦」取代）
- Context：原先假設動作辨識必須對應到資料庫的 1,324 個標籤之一。
- Decision：輸出 top-5 候選供使用者確認，低於信心門檻時輸出 `unknown`。
- Rejected reason：重新檢視後發現辨識結果對應到資料庫沒有實質用途。資料庫的 `instruction_steps` 只描述單點姿勢，作為「正確作法」的參考價值低；而成因推論已決定採自由推理，也用不到資料庫的肌群標註來限縮候選。此設計把一個不存在的約束引入系統，並衍生出 `unknown` 降級路徑等不必要的複雜度。

### 動作辨識與資料庫解耦

- Status：accepted
- Context：專案範圍為通用人體動作，但資料庫只有健身動作（武術 0 筆）。若要求辨識結果必須命中資料庫，非健身動作將無法處理。
- Decision：動作辨識輸出自然語言描述，不與資料庫比對。使用者可直接修正描述文字。資料庫只在最後的處方檢索階段使用。
- Alternatives：維持 top-5 候選比對；限定領域為健身動作。
- Consequences：資料庫缺少某類動作不再影響前三階段，武術等動作可完整診斷並取得肌力訓練處方；介面從固定選單簡化為可編輯文字；`unknown` 降級路徑與信心門檻校準問題一併消除。代價是失去以資料庫標註限縮成因候選的可能性，但該用途在採自由推理後本已不存在。

### 成因推論採自由推理 + 約束映射

- Status：accepted
- Context：資料庫的肌群標註方向是「這個動作練到什麼」，不是「這個人為何做不好」，且缺活動度資料，無法支撐因果推論。
- Decision：允許 VLM 自由推論成因，再以第二次呼叫強制映射到 19 個 `target` 詞彙，映射不到則明確標記。
- Alternatives：自建「動作缺陷 → 成因」知識庫；完全不做成因推論直接給處方。
- Consequences：保留診斷品質與可讀性；代價是因果正確性無從驗證。輸出必須使用「通常和…有關」而非斷定語氣。

### 不綁定動作領域

- Status：accepted
- Context：使用者的需求涵蓋健身以外的動作（武術、踢腿等），但資料庫只有健身動作。
- Decision：輸入不限領域。配合「動作辨識與資料庫解耦」，資料庫涵蓋範圍不再構成流程限制。
- Alternatives：限定健身動作（資料庫即完整標籤空間）。
- Consequences：適用範圍廣；剩餘代價是處方內容仍受資料庫限制，只能提供肌力與柔軟度訓練，無法提供技術訓練。

### 單機 2D 輸入

- Status：accepted
- Context：多機 3D 需要標定與同步，前期成本高。
- Decision：先以單機 2D 影片為輸入。
- Alternatives：直接接 `pose2sim` 多機 3D。
- Consequences：門檻低、可快速驗證架構；代價是含軀幹旋轉或離開拍攝平面的動作，診斷不可靠，需明確標記。

### 不排除同動作模式的動作

- Status：accepted
- Context：原本要求處方不得包含與待改善動作同類型的動作（深蹲有問題不該再推薦深蹲）。但「動作辨識與資料庫解耦」後，系統只有自然語言描述，無從判定使用者的動作對應資料庫哪一筆；且資料集沒有動作模式（squat / hinge / push / pull）欄位，要實作需替 1,324 筆動作另行標註。
- Decision：不做同動作模式排除，允許處方包含同類型動作。
- Alternatives：替 1,324 筆動作補標動作模式（工作量大且無現成資料）；以 `body_part` 粗略排除（精度低，且會誤刪有效處方）。
- Consequences：處方可能出現「深蹲做不好 → 推薦另一種深蹲」的情況，需在輸出或介面上讓使用者自行判斷。移除了一條無法實作也無法驗證的成功標準。若日後補上動作模式標註，可重新評估。

### 核心與介面分離，介面採 FastAPI

- Status：accepted
- Context：本專案同時要作為可操作的工具與可被其他專案（`video-understanding`、`pose2sim`）匯入的模組。若把功能寫在介面框架內，模組用途即消失。
- Decision：核心邏輯實作為純 Python 模組，不匯入任何 web 相關套件；FastAPI 作為薄 adapter，只做請求接收、呼叫核心、序列化回應；前端為呼叫 API 的靜態頁面。
- Alternatives：Gradio（開發最快，但介面框架與邏輯耦合度高，且元件模型限制流程設計）；純 CLI（無法滿足操作介面需求）。
- Consequences：核心可被腳本或其他專案直接使用，介面可替換；代價是前端需自行實作，開發量高於 Gradio。驗證時需明確測試「核心層在無 FastAPI 環境下可完整呼叫」。

### 首次評測結果（2026-08-09）

`scripts/eval_recognition.py`，`34data/workout-vids` 每類抽 5 支共 110 支，
`qwen2.5vl:7b`、seed 42、平均 4.8 秒/支。

動作辨識（唯一有標準答案的階段）：

- 完全命中 36/110 = **32.7%**；含部分命中 56/110 = **50.9%**。
- 全對的類別：`deadlift`、`pull up`、`push-up`（各 5/5），`plank`、`squat`（4/5）。
  都是體態差異大的全身動作。
- 全錯的類別：`lat pulldown`、`tricep pushdown`（各 0/5）。但兩者的描述在機制上是對的
  （「pulls down on a cable machine handle while seated」），只是沒有使用資料集的命名。
  評分規則要求字面吻合，而本系統刻意輸出自由文字，因此此數字低估實際理解程度。
- 真正的混淆：`leg extension` → 一致描述為 leg curl（相反動作）；`hip thrust` → bench press／
  rowing machine；`russian twist` → sit-up。

以下三項為評測發現的實際缺陷，**1 與 2 已修正，3 隨之消失**（見下節）：

1. **約束映射從不拒絕。** 12 份診斷全部是「無對應項目：—」。prompt 要求無法對應者略過，
   但模型仍把「Lumbar spine mobility restrictions」「Posture control challenges」
   「Balance and proprioception」等非肌力項目一律映射到某個肌群。接地率看似 100%，
   實為過度映射，使「明確標記無對應」的保護機制失效。
2. **`levator scapulae` 的字面陷阱。** 12 份診斷中出現 6 次，是第二高頻的對應肌群。
   成因是自由推論常提到 scapular retraction／scapula，而 19 個允許詞中只有
   `levator scapulae` 含 scapula 字樣，模型依字面相似度選它。正確對應應為 `traps` 或
   `upper back`（rhomboids）。
3. **錯誤映射被稀有度規則放大。** `levator scapulae` 在資料集中候選極少，處方檢索的
   「稀有肌群優先」規則因此把它排在最前，導致 `side push neck stretch`（頸部伸展）
   在 12 份處方中出現 6 次，成為臥推與飛鳥問題的首選處方。稀有度規則本身正確，
   但會忠實放大上游的錯誤映射。

### 約束映射改為逐項表態後的複測（2026-08-09）

原本要求模型回傳一份肌肉清單，省略無法對應者。模型改為對每個項目各給一個答案，
`none` 是明列的合法答案，並在指令中點出 `levator scapulae` 是頸部肌肉、
`serratus anterior` 只用於肩胛前引。同 12 支影片、同 seed 複測：

| 指標 | 修正前 | 修正後 |
|---|---|---|
| 有回報「無對應項目」的份數 | 0/12 | **9/12** |
| `levator scapulae` 被選中次數 | 6 | **0** |
| 處方含 `side push neck stretch` | 6 | **0** |
| 能產出處方的份數 | 12/12 | 12/12 |

被拒絕的項目確實都不是肌力問題：`Posture control challenges`、
`Upper arm mobility limitations`、`Lumbar spine mobility`、
`Pelvic tilt control problems`、`Shoulder joint mobility`。
`Scapular retractor muscle weakness (e.g., rhomboids)` 現在對應到 `traps` 而非
`levator scapulae`。拒絕變多並未使任何一份失去處方。

根因記錄：模型把「交一份清單、省略不適用者」理解為逐項翻譯，傾向為每個輸入項目
產生一個輸出，不願留白。讓「無對應」成為必須明講的答案，而非需要主動省略的動作，
即可消除此傾向。

### 骨架比對辨識：已測試並否決（2026-08-10）

動機是三個真實混淆（`leg extension` ↔ leg curl、`hip thrust` ↔ bench press、
`russian twist` ↔ sit-up）全部是**方向**的混淆，而抽取靜態影格無法表達方向。
做法是對資料集全部 GIF 逐幀跑骨架，建立每個動作一個範例的參考庫，再以關節活動幅度
（ROM）指紋做最近鄰比對。

結果，同 110 支影片、同評分規則：

| | top-1 完全命中 | top-5 完全命中 | top-5 含部分命中 |
|---|---|---|---|
| 骨架 ROM 指紋 | 3.6% | **12.7%** | 34.5% |
| VLM（對照） | **32.7%** | — | 50.9%（top-1） |

事前約定的中止條件為「骨架 top-5 打不過 VLM top-1 即停止」。12.7% < 32.7%，觸發，
**此路線終止**。

失敗原因，依重要性排序：

1. **ROM 在定義上丟棄方向。** 它記錄「膝關節活動了 52 度」，不記錄「先伸直再彎曲」。
   走骨架路線的動機正是取得方向資訊，卻選了一個無法表達方向的特徵。
   `lever leg extension` 與 `lever seated leg curl` 的指紋餘弦相似度為 0.645，
   其差異來自雜訊而非方向。
2. **9 個維度無法索引 1,284 個類別。** 這是資訊量問題，換演算法解決不了。
3. **參考 GIF 使用洋蔥皮動畫。** 每幀疊有前一姿勢的淡色殘影，骨架會橫跨兩個身體
   （深蹲最明顯）。以亮度門檻抑制殘影只救回部分影格。
4. **器材差異骨架不可見。** `barbell curl` 與 `dumbbell hammer curl` 相似度 0.913，
   兩者動作相同、只差握法與器材。

參考庫本身仍有訊號：以資料集 `target` 欄位比對「活動幅度最大的關節是否符合該肌群」，
整體一致率 61.0%（隨機基準 31.7%）。但分佈極不平均，且失敗處是特徵未涵蓋所致：
`lats` 94.7%、`upper back` 94.2%、`biceps` 90.1%、`pectorals` 85.6%；
而 `calves` 25.4%（小腿靠踝關節，COCO-17 無腳掌）、`abs` 20.1%、`traps` 20.0%、
`spine` 5.3%（軀幹屈曲與旋轉、聳肩皆未量測）。

大幅度的全身動作仍分得出來：`squat` 5/5、`deadlift` 4/5 完全命中。

**未被測試的假設**：方向資訊需要時間序軌跡才能表達，而軌跡比對需要先切出單一重複
（參考 GIF 是 1 次，查詢影片是 8 次）。本輪未實作，因此「骨架能否解決方向混淆」
這個原始問題**仍未得到回答**——被否決的是 ROM 指紋，不是骨架本身。

`src/movement_coach/pose.py` 與相關腳本保留，因為姿勢評估（角度、活動範圍、
左右對稱）本就需要這些量測，與辨識用途無關。

### 骨架量測作為評估階段的輔助輸入（2026-08-10）

骨架用於辨識已被否決，改用於它原本擅長的事：量測。`metrics.py` 從逐幀關節角度算出
活動範圍、左右差、軀幹傾角與重複次數，以文字併入評估階段的提示。這回答了
`spec.md` 原本列為 Open Question 的「是否需要骨架作為 VLM 的輔助輸入」。

同 12 支影片、同 seed 對照：

| | 未發現問題 | 診斷引用實際角度數字 |
|---|---|---|
| 無骨架量測 | 0/12 | **0/12** |
| 有骨架量測（初版提示） | 8/12 | — |
| 有骨架量測（修正提示） | 6/12 | **5/12** |

主要收穫是診斷從猜測變成可查證。加入量測前，模型對全部 12 支都找得出問題、
且從未引用任何數字；加入後改為引用實測值，例如「軀幹前傾 58°，過度」、
「肩關節活動範圍左 84° 右 79°，超出側平舉的一般範圍」。

「未發現問題」的比例同時大幅上升。無標準答案可判定何者正確，兩種解讀都成立：
先前對每一支影片都挑得出毛病，本身就可疑；但也可能是模型看到數字正常便放過了
畫面上的問題。修正提示時已明確要求量測未涵蓋之處仍須從畫面判斷，比例由 8/12
降至 6/12。

初版提示要求「優先採信這些數字勝過畫面」，這是錯的：單機拍攝下遠側肢體是推測的，
一支胸飛鳥影片量到「左右手肘差 102°」——生理上不可能，是遮擋造成的假象。
提示已改為說明左右差超過約 30° 應視為該側未清楚可見，屬缺漏資料而非缺陷。
此指示未被完全遵守（一支硬舉仍將 36° 的左右膝差當成缺陷回報）。

量測本身的可靠度以 `MIN_CONFIDENCE` 控制。實測 0.3 過鬆（坐姿腿伸展把手肘報為
最大活動關節、臥推報出 93° 髖關節位移）、0.65 過嚴（深蹲只剩單側膝關節），
定為 0.5。

重複次數偵測改用關節角度而非畫面差分。原先直接取自相關窗內最大值，會固定落在
最短的 lag，使 3 秒的彎舉被報成 12 下、靜態棒式被報成 50 下。改為只接受真正的
局部峰值，並在次諧波同樣強時拒絕作答（避免把過快的震盪報成較低的次數）。

## Known Gaps

- 影片只取樣 6 張均勻分布的畫面，關鍵動作可能落在取樣點之外。實測拳擊影片時
  第一階段描述為「男子用毛巾擦臉」，屬取樣落點問題而非模型能力問題。
  動作描述的人工修正步驟因此是必要設計，不是選配。
- 動作辨識完全命中率 32.7%。部分失分源自評分規則要求字面吻合（`lat pulldown` 的
  描述機制正確但未使用該名稱），但也有真實混淆：`leg extension` 一致被描述為
  leg curl、`hip thrust` 被描述為 bench press、`russian twist` 被描述為 sit-up。
- 約束映射的拒絕行為只在 12 支樣本上複測過，尚未確認是否會走向另一個極端（過度拒絕）。
- 診斷與處方的適切性仍無標準答案可評，只能人工判讀。
- 正規化表有 9 個詞無對應 `target`，其中 `hip flexors` 出現 77 次，該肌群無法產出處方。
- 是否需要 2D pose 作為 VLM 輔助輸入未決。
- 前端無自動化測試，僅手動驗證過 HTTP 層。
- 未設定 formatter、linter、type checker；打包從未實際執行。
- 上傳影片的清理僅在新上傳時觸發，長期閒置的服務不會自行回收暫存檔。
- 服務層為單一模組層級的 `MovementCoach` 實例，未考慮多工作行程下的資料集重複載入成本。
