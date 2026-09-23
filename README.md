# Wash Hand Detect — MediaPipe 七步洗手即時辨識系統

本專案使用 **MediaPipe Hand Landmarker + OpenCV + 特徵萃取 + 規則分類器 (MVP) + LSTM 時序模型 + 洗手狀態機**，實現標準洗手七步（內、外、夾、弓、大、立、腕）之即時偵測、步驟引導與計時評估系統。

---

## 🌟 系統特色

1. **雙手骨架偵測**：使用 MediaPipe 擷取左右手各 21 個 3D 特徵點，嚴格對齊左右手順序並進行座標歸一化 (Wrist 歸零 + 手掌尺度縮放)。
2. **多維特徵工程**：包含掌心中心 (Palm Center)、掌面法向量 (Palm Normal)、手指關節彎曲角度 (Finger Angles)、對應指尖距離與運動速度。
3. **即時雙模分類**：
   - **Rule-based Classifier (MVP)**：無需龐大訓練資料即可精準辨識與即時姿勢反饋。
   - **LSTM / GRU 時序模型**：30 幀滑動窗口時序預測，依受試者 (Person ID) 分割資料預防 Data Leakage。
4. **洗手狀態機 (State Machine)**：
   - **教學模式 (Sequence Mode)**：強制依「內 → 外 → 夾 → 弓 → 大 → 立 → 腕」順序進行。
   - **自由模式 (Free Mode)**：任意順序搓洗，直至七步全數完成。
   - 包含秒數達標計時、短暫飄移容錯與完成結算畫面。
5. **現代化 UI / HUD**：支援繁體中文即時狀態、信心度進度條、七步清單與姿勢指導反饋。

---

## 📂 專案結構

```text
wash_hand_detect/
├── README.md
├── IMPLEMENTATION.md
├── requirements.txt
│
├── models/
│   └── wash_hand_lstm.keras       # 訓練完成之 LSTM 模型權重
│
├── data/
│   ├── raw/
│   ├── processed/                 # 依受試者存放之 .npy 30幀特徵檔案
│   └── labels.csv                 # 錄製樣本索引清單
│
├── src/
│   ├── __init__.py
│   ├── camera.py                  # 攝影機 / 測試影像來源封裝
│   ├── hand_detector.py           # MediaPipe 雙手骨架擷取與繪製
│   ├── features.py                # 幾何特徵計算與歸一化
│   ├── rule_classifier.py         # 規則式分類器與姿勢指引
│   ├── collect_data.py            # 互動式資料收集錄製工具
│   ├── train.py                   # LSTM 時序模型訓練與評估 (F1/Confusion Matrix)
│   ├── state_machine.py           # 洗手狀態轉移與持續時間計時
│   ├── ui.py                      # 繁體中文 HUD 渲染與介面元件
│   └── realtime.py                # 即時主程式 Pipeline
│
└── tests/
    └── test_pipeline.py           # 單元與整合測試套件
```

---

## 🚀 快速開始

### 1. 安裝環境與套件

```bash
# 建立虛擬環境 (建議 Python 3.10 / 3.11)
python3.11 -m venv venv
source venv/bin/activate

# 安裝相依套件
pip install -r requirements.txt
```

### 2. 啟動即時洗手辨識 (預設 Rule-based 模式)

```bash
# 啟動即時辨識 (使用預設攝影機)
python -m src.realtime

# 指定教學循序模式或自由模式
python -m src.realtime --guide-mode sequence --step-duration 2.5
```

**快捷鍵說明**：
- `Q`：離開程式
- `R`：重新開始洗手計時
- `M`：切換 教學模式 / 自由模式
- `C`：切換 分類器 (Rule-based / LSTM)

---

## 📊 資料收集與 LSTM 模型訓練

### 1. 錄製手勢資料

```bash
# 為受試者錄製特定手勢 (預設 10 clips，每 clip 30 幀)
python -m src.collect_data --person person_01 --action inside
python -m src.collect_data --person person_01 --action outside
```

### 2. 訓練 LSTM 模型

```bash
# 依 Person ID 自動切分 Train/Val/Test 並評估 Confusion Matrix 與 F1-score
python -m src.train --epochs 30 --batch-size 16
```

### 3. 以 LSTM 模型運行即時辨識

```bash
python -m src.realtime --mode lstm --model-path models/wash_hand_lstm.keras
```

---

## 🧪 執行測試

```bash
pytest tests/
```
