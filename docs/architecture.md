# Architecture and Design

本文件回答「如何實作與驗證」，保存目前採用的系統設計、演算法、測試方法與重要取捨。需求與成功標準以 `spec.md` 為準。

## Overview

- System / approach：四階段管線。前三階段由本地 VLM 進行自然語言推理（描述 → 診斷 → 成因），第四階段以確定性演算法從資料庫檢索處方。兩者之間以一層約束映射銜接，確保自由推理的結論能落回可檢索的詞彙空間。本質是 RAG：推理自由，輸出接地。
- 分層：核心邏輯為純 Python 模組，不依賴任何介面技術；FastAPI 為薄 adapter，只做 HTTP 轉接；前端頁面呼叫 API。核心可在無介面環境下獨立使用。
- Primary language / runtime：Python（版本待確認）
- Data / external boundary：本機 VLM 推論服務（待確認）；`exercises.json` 為唯一動作知識來源；輸入影片為本機檔案。

## Repository Map

原始碼尚未實作，目錄結構待確認。規劃中的分層與責任：

- 核心層：VLM 推理（描述、診斷、成因）、約束映射、處方檢索。純函式，不匯入 web 相關套件。
- 服務層：FastAPI 應用。只負責接收請求、呼叫核心層、序列化回應。不得含業務邏輯。
- 前端層：靜態頁面，呼叫服務層 API。技術待確認（建議無 build step）。
- 測試與實驗位置：待確認。

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

- Interface：核心層為 Python 函式；服務層為 HTTP API（端點設計待確認）；前端為網頁。
- Data model / state：待確認。影片在請求之間的保存方式與清理策略未定。

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

```text
待確認（尚未實作）
```

### Data and Environment

- Dataset：[exercises-dataset](https://github.com/hasaneyldrm/exercises-dataset) `data/exercises.json`，1,324 筆。版本固定於 commit `7455efae41b330c265e7cd4b78dfa848e7ce5ebd`（2026-07-16），SHA-256 `656634224b8977b99a6d765470ee123260d4979715eaa4e7c0b7c8bb0d79f93d`，取得指令見 `README.md`。程式碼與資料為 MIT；媒體檔版權屬 Gym visual，僅授權轉散布，本專案不使用媒體檔。資料集不納入版本控制。
- Environment：待確認。
- Baseline：尚未需要。
- Metrics：目前只有一項可自動檢查——處方動作可由 `id` 在 `exercises.json` 中驗證存在。其餘待模組可執行後再定。
- Reproducibility：待確認。

### 後續評估方向

模組可執行後再考慮，目前不設計、不排程、不訂門檻：

- 診斷結論在重複執行下是否穩定。
- 成因推論的輸出有多少比例能映射到可檢索的肌群。
- 與「VLM 直接生成訓練建議、不接地」的輸出比較差異。

### Critical Cases

- [ ] 正常案例：資料庫內的健身動作（如深蹲），完整跑完四階段。
- [ ] 邊界案例：資料庫外的動作（如跆拳道踢腿），流程完整產出診斷與處方，不因無法比對資料庫而中止。
- [ ] 邊界案例：診斷結論無法映射（如「踝背屈受限」），明確標記無對應而非硬湊。
- [ ] 邊界案例：核心層在無 FastAPI 環境下由腳本呼叫，功能完整。
- [ ] 邊界案例：VLM 服務未啟動、影片解碼失敗、`exercises.json` 缺失，皆有明確回報且無靜默失敗。
- [ ] 介面案例：網頁可完成上傳、修正動作描述、檢視診斷與處方的完整操作。

### Verification Status

- 資料集稽核：**passed**（2026-08-09，對 `exercises.json` 1,324 筆實際統計，結果見下節）。可由本文件記載的數字重新驗證。
- 正規化表覆蓋率：**passed**（2026-08-09，31/40 個詞可映射，按出現次數計 95.6%）。表格內容已完整記載於「肌群正規化表」，可重新驗證。
- 集合覆蓋演算法原型：**初步觀察，非正式驗證**。2026-08-09 於 session 暫存區以 Python 執行，兩組情境皆能以 2–3 個動作覆蓋 4 個弱點肌群。**原型程式碼未保存，此結果目前無法重現**，實作後須以正式測試取代本條記錄。
- 其他所有檢查：**not run**（尚未實作）。

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

## Known Gaps

- 全部程式碼尚未實作。處方檢索層的原型未保存，其結果目前無法重現。
- 集合覆蓋演算法已知兩個缺陷（實測發現，尚未修正）：
  1. `secondary_muscles` 權重 0.5 過鬆，短跑（`wind sprints`）被選為小腿訓練。
  2. 未對類別大小正規化，`abs`（169 個動作）易被過度選中。
- 正規化表有 9 個詞無對應 `target`，其中 `hip flexors` 出現 77 次，該肌群無法產出處方。
- 輸出品質的評估方式未定，待模組可執行後再依實際輸出決定。
- 是否需要 2D pose 作為 VLM 輔助輸入未決。
- API 端點設計、影片在請求之間的保存與清理策略未定。
- 前端技術未定。
- VLM 推論服務、Python 版本、硬體需求皆未確認。
