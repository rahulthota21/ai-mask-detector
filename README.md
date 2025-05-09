# 🛡️ VisionMaskGuard

> A real-time face mask detection app using Python, OpenCV, and a custom-trained MobileNetV2 model.

---

## 📸 Features

- ✅ Detects faces live from webcam
- 🧠 Classifies each face as **Mask 😷** or **No Mask 😐**
- 🟩 Green box for Mask, 🟥 Red box for No Mask
- 📊 Live counter display
- ⚡ Optimized for real-time performance

---

## 🧠 Model Info

- Built using **MobileNetV2**
- Trained on [Face Mask Detection Dataset](https://github.com/chandrikadeb7/Face-Mask-Detection)
- Format: `.h5` (Keras)

---

## 🚀 How to Run

1. **Clone repo**
   ```bash
   git clone https://github.com/rahulthota21/ai-mask-detector.git
   cd ai-mask-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run app**
   ```bash
   python realtime_mask_detector.py
   ```

4. **Press `Q`** to exit.

---

## 🧾 Files

| File | Description |
|------|-------------|
| `realtime_mask_detector.py` | Real-time detection code |
| `mask_detector.h5` | Pretrained mask classification model |
| `MaskDetectorTraining.ipynb` | Notebook used to train the model |
| `requirements.txt` | All required Python packages |

---

## 🧑‍💻 Author

- **Name:** Thota Rahul  
- **GitHub:** [@rahulthota21](https://github.com/rahulthota21)

---

## 📄 License

This project is open-source and free to use under [MIT License](LICENSE).
