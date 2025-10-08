# 🎭 Emotion Detection Using Convolutional Neural Networks (CNN)

A deep learning project that classifies human emotions from facial expressions using **Convolutional Neural Networks (CNN)**. The model is trained on the **[Emotion Detection FER Dataset](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)** from Kaggle, which contains grayscale facial images representing seven emotional categories — *Angry, Disgust, Fear, Happy, Sad, Surprise,* and *Neutral.*
Additionally, a real-time emotion detection system using **OpenCV** enables webcam-based emotion recognition with live predictions and CSV logging.

---

## 🧠 Project Overview

This project consists of two main components:

1. **Model Training (`project.py`)**

   * Builds and trains a deep CNN model for emotion classification.
   * Includes data preprocessing, augmentation, model evaluation, and visualization.

2. **Real-Time Detection (`emotion_webcam.py`)**

   * Loads the trained CNN model to detect faces from a live webcam feed.
   * Predicts emotions in real-time and overlays results on video frames.
   * Logs predictions with timestamps and confidence scores to a CSV file.

---

## 📂 Project Structure

```
Emotion-Detection
│
├── project.py                
├── emotion_webcam.py         
├── emotion_cnn_model.h5      
├── emotion_log.csv           
├── archive (2)/               
│   ├── train/                 
│   └── test/                 
├── README.md                 
└── Requirements.txt
```

---

## 📊 Dataset

The project uses the **[Emotion Detection FER Dataset](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)** available on Kaggle.
This dataset contains **48x48 grayscale facial images** classified into seven emotion categories.

| Emotion  | Description                    |
| -------- | ------------------------------ |
| Angry    | Expression of anger            |
| Disgust  | Expression of disgust          |
| Fear     | Expression of fear             |
| Happy    | Expression of joy or happiness |
| Sad      | Expression of sadness          |
| Surprise | Expression of astonishment     |
| Neutral  | Relaxed or non-expressive face |

> **Note:** The dataset folders (`train` and `test`) are structured such that each emotion is stored in its corresponding subdirectory.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Emotion-Detection-CNN.git
cd Emotion-Detection-CNN
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies

Create a `requirements.txt` file or install manually:

```bash
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn
```

---

## 🧩 Model Architecture

The CNN architecture includes multiple convolutional blocks with **Batch Normalization**, **Dropout**, and **MaxPooling** layers to improve generalization and reduce overfitting.

**Model Summary:**

* 3 Convolutional Blocks: 64, 128, 256 filters
* Fully Connected Dense Layer: 512 neurons
* Dropout for regularization
* Softmax output for 7-class classification
* Optimizer: Adam (lr = 0.0005)
* Loss Function: Categorical Crossentropy

---

## 🚀 Training the Model

Run the training script:

```bash
python project.py
```

The script will:

* Load and preprocess images.
* Train the CNN for 40 epochs.
* Display training curves for accuracy and loss.
* Generate confusion matrix and classification report.
* Save the final model as `emotion_cnn_model.h5`.

---

## 🎥 Real-Time Emotion Detection

After model training, run the webcam detection script:

```bash
python emotion_webcam.py
```

**Features:**

* Detects faces in real-time using Haar Cascade.
* Classifies emotion using the trained CNN model.
* Displays predicted emotion with confidence score on live video.
* Logs timestamped results to `emotion_log.csv`.

Press **`q`** to quit the webcam window.

---

## 📈 Model Evaluation

The model produces:

* **Accuracy & Loss Curves** for both training and validation.
* **Confusion Matrix** visualizing prediction performance.
* **Classification Report** with precision, recall, and F1-score.

Example visualization (from `project.py`):

```python
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
```

---

## 🧾 Output Samples

* **Model file:** `emotion_cnn_model.h5`
* **CSV Log:** `emotion_log.csv`

  ```
  Timestamp,Emotion,Confidence
  2025-10-08 18:45:12,happy,0.9843
  2025-10-08 18:45:13,neutral,0.7512
  ```
* **Video Window:** Displays face bounding boxes and predicted emotion labels.

---

## 🧪 Technologies Used

| Category         | Tools                          |
| ---------------- | ------------------------------ |
| Deep Learning    | TensorFlow / Keras             |
| Image Processing | OpenCV                         |
| Visualization    | Matplotlib, Seaborn            |
| Data Handling    | NumPy, scikit-learn            |
| Model Logging    | CSV, Confusion Matrix, Reports |

---

## 🧠 Future Improvements

* Integrate **transfer learning** (VGG16, ResNet50) for higher accuracy.
* Deploy as a **web app** using Flask or Streamlit.
* Implement real-time dashboard visualization for logged emotions.
* Explore emotion trends over time using CSV logs.

---

## 👩‍💻 Author

📧 Email: rishika1826@gmail.com
🌐 GitHub: [https://github.com/<your-username>](https://github.com/<your-username>)

> A deep learning experiment blending human emotion and machine perception — built with ❤️ and CNNs.

---

### ⭐ If you find this project useful, please consider giving it a star on GitHub!
