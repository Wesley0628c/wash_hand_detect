# Wash Hand Detect — MediaPipe 七步洗手即時辨識系統

本專案使用 **MediaPipe Hand Landmarker + OpenCV + 160維幾何特徵工程 + 15幀時序統計量 (960維) + XGBoost 姿勢分類器 + 洗手狀態機**，實現標準洗手七步（內、外、夾、弓、大、立、腕）之即時偵測、步驟引導、計時評估與視覺化展示系統。

👉 詳細更新歷程與錯誤分析請參閱：[版本更新與修正記錄 (CHANGELOG.md)](./CHANGELOG.md)

---

## 🌟 系統特色

1. **雙手骨架偵測與時序追蹤**：使用 MediaPipe 擷取左右手各 21 個 3D 特徵點，透過上一幀歐氏距離匈牙利匹配防範交叉換位，並包含 4 幀殘影平滑補償。
2. **多維幾何特徵工程**：包含 Wrist Ratio ($R_{wrist}$)、指節到掌心距離 (Knuckle-to-Palm)、對稱交錯深度 (Symmetric Interlace)、指尖聚攏度 (Fingertip Spread) 等 160 維判別特徵。
3. **15 幀時序統計特徵 (960 維)**：在 15 幀滑動窗口內計算均值、標準差、極值與變化差值，捕獲動態洗手搓揉軌跡。
4. **Landmark Dropout 泡沫資料增強**：隨機 20% 節點遮蔽與 20% 單手掉點模擬，使模型在肥皂泡沫與重疊遮擋下具備極高強健度。
5. **雙模/混合分類器 (Hybrid Classifier)**：
   - **XGBoost 姿勢分類器**：5-Fold 交叉驗證 Accuracy 達 94.0%，Macro F1 達 0.93。
   - **幾何互斥約束 (Negative Evidence)**：徹底消除「外 vs 腕」、「弓 vs 外」之互相誤判。
6. **洗手狀態機 (State Machine)**：
   - **教學模式 (Sequence Mode)**：強制依「內 → 外 → 夾 → 弓 → 大 → 立 → 腕」順序進行。
   - **自由模式 (Free Mode)**：任意順序搓洗，直至七步全數完成。
7. **現代化非遮擋 UI / HUD**：寬螢幕自動避讓洗手工作區，支援繁體中文即時狀態、信心度進度條與視覺化影片匯出。

---

## 📂 專案結構

```text
wash_hand_detect/
├── README.md
├── CHANGELOG.md                   # 詳細版本更新、功能修正與錯誤診斷記錄
├── ARCHITECTURE.md                # 系統架構與特徵定義詳細說明
├── IMPLEMENTATION.md
├── requirements.txt
│
├── models/
│   ├── wash_hand_xgb.joblib       # 訓練完成之 XGBoost 分類器模型
│   └── wash_hand_lstm.keras       # LSTM 時序模型
│
├── data/
│   ├── raw/
│   │   ├── wash_7steps_yt.mp4     # 測試集一 (教育宣導雙畫面)
│   │   └── wash_7steps_yt2.mp4    # 測試集二 (水槽俯拍實錄 1080p)
│   ├── annotated_eval_yt.mp4      # 影片一視覺化評估匯出檔
│   └── annotated_eval_yt2.mp4     # 影片二視覺化評估匯出檔
│
├── src/
│   ├── __init__.py
│   ├── camera.py                  # 攝影機 / 測試影像來源封裝
│   ├── hand_detector.py           # MediaPipe 雙手骨架擷取、ROI 上採樣與時序追蹤
│   ├── features.py                # 160 維幾何特徵計算與歸一化
│   ├── temporal_features.py       # 15 幀時序統計量 (960 維) 與 Landmark Dropout 增強
│   ├── rule_classifier.py         # 規則式分類器 (含 Negative Evidence 負向互斥)
│   ├── ml_classifier.py           # XGBoost / Hybrid 機器學習分類器介面
│   ├── train_ml.py                # XGBoost 模型訓練與 5-Fold 交叉驗證
│   ├── evaluate_segments.py       # Ground Truth 段落評估與視覺化影片匯出工具
│   ├── state_machine.py           # 洗手狀態轉移與持續時間計時
│   ├── ui.py                      # 繁體中文非遮擋 HUD 渲染
│   ├── test_video.py              # 離線影片測試工具
│   └── realtime.py                # 即時主程式 Pipeline
│
└── tests/
    └── test_pipeline.py           # 單元與整合測試套件 (10/10 綠燈)
```

---

## 🚀 快速開始

### 1. 安裝環境與套件

```bash
# 建立虛擬環境 (建議 Python 3.10 / 3.11)
python3.11 -m venv venv
source venv/bin/activate

# 安裝相依套件 (含 MediaPipe, OpenCV, XGBoost, LightGBM, Pillow)
pip install -r requirements.txt
```

### 2. 執行影片基準評估與視覺化影片匯出

```bash
# 評估影片 1 並產出帶 HUD 標註影片
PYTHONPATH=. ./venv/bin/python src/evaluate_segments.py \
  --video data/raw/wash_7steps_yt.mp4 \
  --model-type hybrid \
  --output data/annotated_eval_yt.mp4

# 評估影片 2 (水槽俯拍) 並產出帶 HUD 標註影片 (準確率 92%)
PYTHONPATH=. ./venv/bin/python src/evaluate_segments.py \
  --video data/raw/wash_7steps_yt2.mp4 \
  --model-type hybrid \
  --output data/annotated_eval_yt2.mp4
```

### 3. 重新訓練 XGBoost 姿勢分類器

```bash
# 啟動 4 倍 Landmark Dropout 增強並進行 5-Fold 交叉驗證
PYTHONPATH=. ./venv/bin/python src/train_ml.py --aug 4 --output models/wash_hand_xgb.joblib
```

### 4. 啟動即時 WebCam 辨識

```bash
# 啟動即時辨識 (使用 Hybrid XGBoost 分類器)
python -m src.realtime --guide-mode free
```

---

## 🧪 執行單元測試

```bash
PYTHONPATH=. pytest tests/
```

