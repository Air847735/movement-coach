# RTMPose 使用說明（rtmlib 路線）

本文件說明本機如何執行 RTMPose、為何不走 mmpose，以及 GPU 設定與實測效能。
與專案需求無關，屬於環境與工具說明。

## 一句話

**模型是同一個 RTMPose，差別只在執行方式**：`mmpose` 是完整的研究訓練框架，
`rtmlib` 是同一批模型的純推論封裝。本專案只需要推論，因此走 rtmlib。

## rtmlib 與 mmpose 的差別

| | mmpose | rtmlib |
|---|---|---|
| 定位 | 研究框架：訓練 + 推論 + 評測 | 只做推論 |
| 相依 | `mmengine`、`mmcv`、`mmdet`、torch | `onnxruntime`、`opencv`、`numpy`、`tqdm` |
| 安裝痛點 | `mmcv` 需與 torch/CUDA 版本嚴格對齊，常需編譯 | 純 pip，無編譯 |
| 模型格式 | PyTorch checkpoint + config 檔 | ONNX，首次使用自動下載 |
| 設定方式 | Python config 檔（繼承、覆寫） | 建構子參數 |
| 能否訓練 | 可以 | **不行** |
| 模型選擇 | 完整 model zoo | RTMPose / RTMO / RTMDet / YOLOX / ViTPose 等推論模型 |
| 權重來源 | OpenMMLab | 同上，官方匯出的 ONNX 版本 |

**權重是同一批**——rtmlib 下載的就是 OpenMMLab 官方匯出的 ONNX
（網址均為 `download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/...`）。
精度沒有差異，差別在於少了 mmcv 那一層相依。

選擇 rtmlib 的理由：本專案不訓練模型，而 `mmcv` 與 torch/CUDA 的版本綁定
在同一台機器上要與其他專案共存時特別麻煩。

## 安裝

`rtmlib` 宣告相依 `opencv-python` 與 `opencv-contrib-python`，會與本專案使用的
`opencv-python-headless` 互相覆蓋（三者都提供 `cv2`）。以 `--no-deps` 安裝並自行補
其餘相依即可避免：

```bash
pip install --no-deps rtmlib
pip install onnxruntime tqdm
```

模型權重快取於 `~/.cache/rtmlib`（約 191 MB），**跨 conda 環境共用**，不會重複下載。

## GPU 設定

`onnxruntime-gpu` 需要 CUDA 12 與 cuDNN 9 的共享函式庫。這些函式庫由 pip 的
`nvidia-*-cu12` 套件提供，安裝在 `site-packages/nvidia/*/lib`，**不在動態載入器的
預設搜尋路徑上**。缺少設定時的症狀是：

```
Failed to load library libonnxruntime_providers_cuda.so with error:
libcublasLt.so.12: cannot open shared object file
```

接著 onnxruntime 會**靜默退回 CPU**——程式不會失敗，只是變慢，很容易沒發現。

`Pose2Sim` 環境已加入啟動腳本自動處理：

- `$CONDA_PREFIX/etc/conda/activate.d/onnxruntime_cuda.sh`
- `$CONDA_PREFIX/etc/conda/deactivate.d/onnxruntime_cuda.sh`

`conda activate Pose2Sim` 後 `CUDAExecutionProvider` 即可用。確認方式：

```python
import onnxruntime as ort
assert "CUDAExecutionProvider" in ort.get_available_providers()
```

注意該環境同時存在兩套 CUDA：torch 使用 CUDA 13（`torch 2.12.1+cu132`），
onnxruntime 使用 CUDA 12。兩套並存沒有問題，啟動腳本只加入 cu12 的路徑。

## 實測效能

RTX 3070 8GB，116 幀 1280×720 真實影片（非合成影像）：

| 模型 | CPU | GPU |
|---|---|---|
| `lightweight`（RTMPose-s + YOLOX-tiny） | **34.8 fps** | 27.4 fps |
| `balanced`（RTMPose-m + YOLOX-m） | 6.2 fps | **12.4 fps** |

小模型在 CPU 上反而較快：單次推論的呼叫開銷蓋過運算本身。模型變大後 GPU 才拉開差距。

**本專案採 `lightweight` + CPU**，理由有三：

1. 這個組合最快（34.8 fps），一段 5 秒影片約 3 秒處理完。
2. 動作辨識只需要關節角度的粗略走勢，不需要 `balanced` 的精度。
3. **GPU 留給 VLM**。8 GB VRAM 在載入 `qwen2.5vl:7b` 後已所剩無幾，
   骨架與 VLM 搶顯存反而整體更慢。

參考點：VLM 處理 24 張影格需 18.5 秒，骨架處理全部 116 幀只要 3.3 秒。

## 用法

```python
import cv2
from rtmlib import Body

model = Body(mode="lightweight", backend="onnxruntime", device="cpu")

keypoints, scores = model(frame)   # frame 為 BGR ndarray
# keypoints: (人數, 17, 2)  COCO-17 座標
# scores:    (人數, 17)     每點信心值
```

COCO-17 關節點索引：

```
 0 鼻   1 左眼  2 右眼  3 左耳  4 右耳
 5 左肩  6 右肩  7 左肘  8 右肘  9 左腕 10 右腕
11 左髖 12 右髖 13 左膝 14 右膝 15 左踝 16 右踝
```

不含腳掌。需要踝背屈或足部角度時改用 `BodyWithFeet`（Halpe-26）。

其他可用類別：`Body`、`BodyWithFeet`、`Wholebody`、`Hand`、`Animal`、
`RTMPose`、`RTMO`、`RTMDet`、`YOLOX`、`ViTPose`、`PoseTracker`、
`RTMPose3d`、`Wholebody3d`、`Custom`。

## 已知問題

- 180×180 的小圖建議先放大再送入（本專案對資料集 GIF 放大 3 倍），偵測率較穩定。
- 關節角度序列有跳動，使用前需平滑處理。
- 單機 2D 的角度隨拍攝視角改變；含軀幹旋轉或身體離開拍攝平面的動作不可靠。
- `onnxruntime` 找不到 CUDA 時會靜默退回 CPU，不會拋錯。效能異常時應先檢查
  `ort.get_available_providers()`。
