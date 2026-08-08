# Handoff

- Status: `active`

## Current Goal

完成專案的需求與設計文件，確認方向後開始實作。目前僅有文件，尚無任何原始碼。

## Completed

- 依 `/srv/template/python` 模板建立專案骨架。
- `docs/spec.md`：需求、範圍、輸入輸出、成功標準、Open Questions。
- `docs/architecture.md`：四階段設計、分層架構、集合覆蓋演算法、肌群正規化表、資料集稽核結果、7 項設計決策（含 1 項 superseded）。
- `README.md`：專案定位、資料取得步驟（含固定 commit 與 SHA-256）、已知限制。
- `AGENTS.md`：專案專屬規則 6 條。
- `.gitignore`：Python、外部資料集、影片與輸出。
- 資料集稽核（2026-08-09）：對 `exercises.json` 1,324 筆實際統計，結果記於 `docs/architecture.md`。
- 肌群正規化表（2026-08-09）：40 → 19 詞彙映射，覆蓋率 95.6%，完整表格記於 `docs/architecture.md`。

無 commit 記錄（尚未初始化為 Git repository）。

## In Progress

- 無進行中的實作工作。

## Remaining

- Git 初始化與接上 GitHub remote（使用者表示會自行建立 remote repository，**不得自行執行 `git init` 或建立 remote**）。
- 全部程式碼：核心層（VLM 推理、約束映射、處方檢索）、服務層（FastAPI）、前端層。
- 決定 Python 版本、套件管理方式、VLM 推論服務。
- 決定前端技術（`docs/spec.md` Open Questions 已列，建議無 build step 的靜態頁面）。
- 決定影片輸入規格（格式、長度上限、解析度下限、大小上限）。
- 修正集合覆蓋演算法的兩個已知缺陷（`secondary_muscles` 權重過鬆、未對類別大小正規化）。
- 輸出品質的評估方式待模組可執行後再決定，目前刻意不預先設計實驗與門檻。

## Changed Files and Interfaces

- `README.md`、`AGENTS.md`、`docs/spec.md`、`docs/architecture.md`、`.gitignore`、`HANDOFF.md`：全部由模板填寫完成。
- `CLAUDE.md`：未修改，維持模板內容。
- 尚無任何程式介面。

## Verification Status

- 資料集稽核：passed（2026-08-09，可由 `docs/architecture.md` 記載的數字重新驗證）。
- 正規化表覆蓋率：passed（2026-08-09，31/40 詞、按出現次數 95.6%）。
- 集合覆蓋演算法原型：初步觀察，**原型程式碼未保存，結果無法重現**。實作後須以正式測試取代。
- 其他所有檢查：not run（尚未實作）。

## Important Constraints

- 核心層不得匯入 FastAPI 或任何 web 套件；任何功能都必須能在不啟動網頁服務的情況下由 Python 直接呼叫。
- 動作辨識階段不得與 `exercises.json` 比對，資料庫只在處方檢索階段使用。
- 處方動作必須全部來自 `exercises.json` 且可由 `id` 驗證。
- 成因推論輸出不得使用斷定因果的語氣。
- 不得提交輸入影片、資料集本體或資料集媒體檔。
- 完整規則見 `AGENTS.md` 的 Project-specific Rules。

## Contradictions or Uncertainty

- 無已知矛盾。2026-08-09 的文件稽核發現「排除同動作模式」與「動作辨識與資料庫解耦」互相衝突，已由使用者決定移除該需求，並在 `docs/architecture.md` 記錄為「不排除同動作模式的動作」設計決策。
- 授權條款（LICENSE）尚未處理，使用者表示暫不處理。

## Suggested Next Step

依 `README.md` 的指令下載 `data/exercises.json` 並驗證 SHA-256，接著重建處方檢索層（核心層第一個模組），以 `docs/architecture.md` 的肌群正規化表為基礎，並補上單元測試取代目前無法重現的原型記錄。
