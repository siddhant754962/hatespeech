-

# 🛡️ Hate Speech & Offensive Language Detection App

This project is a **Machine Learning + NLP web application** built with **Streamlit**, **TF-IDF**, and **ML classifiers**.
It detects whether a given text (e.g., tweets, comments, messages) is:

* 🚨 **Hate Speech**
* ⚡ **Offensive Language**
* ✅ **Neither (Neutral)**

The app allows users to paste any text and instantly classify it into one of the three categories.

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset Information](#-dataset-information)
3. [Technologies Used](#-technologies-used)
4. [Model Training Pipeline](#-model-training-pipeline)

   * Data Preprocessing
   * Feature Extraction (TF-IDF)
   * Model Training
   * Model Evaluation
5. [Web App (Streamlit)](#-web-app-streamlit)
6. [How to Run Locally](#-how-to-run-locally)
7. [Project Structure](#-project-structure)
8. [Future Improvements](#-future-improvements)
9. [Results & Demo](#-results--demo)
10. [Contributors](#-contributors)

---

## 📌 Project Overview

Social media platforms often face issues with **toxic, offensive, and hateful content**.
This project automates text classification to help **moderators, researchers, and developers** filter harmful content.

* Input: Any text (e.g., tweet, message).
* Output: Category → *Hate Speech / Offensive / Neutral*.

---

## 📊 Dataset Information

We used the **Hate Speech & Offensive Language Dataset** (Davidson et al., 2017).

| Feature    | Description                                                |
| ---------- | ---------------------------------------------------------- |
| `tweet_id` | Unique ID of the tweet                                     |
| `class`    | Target label (0 = Hate Speech, 1 = Offensive, 2 = Neutral) |
| `tweet`    | The actual text content                                    |

* Total Samples: ~25,000 tweets
* Classes: **3 (Hate, Offensive, Neither)**

---

## 🛠️ Technologies Used

* **Python 3.10+**
* **NLTK** → Stopword removal, Lemmatization
* **Scikit-learn** → TF-IDF, ML Models
* **Streamlit** → Interactive Web App
* **Joblib** → Model saving & loading

---

## 🔥 Model Training Pipeline

### 1️⃣ Data Preprocessing

* Converted text to lowercase
* Removed URLs, mentions (@), hashtags (#)
* Removed punctuation & special characters
* Tokenization
* Stopword removal
* Lemmatization

### 2️⃣ Feature Extraction

* Used **TF-IDF Vectorization** for text representation.

### 3️⃣ Model Training

* Trained **Logistic Regression, SVM, Random Forest**
* Selected the **best performing model** (highest accuracy & F1-score).

### 4️⃣ Model Evaluation

| Model               | Accuracy | F1-score |
| ------------------- | -------- | -------- |
| Logistic Regression | ~92%     | High     |
| SVM                 | ~91%     | High     |
| Random Forest       | ~89%     | Moderate |

✅ Final Saved Model: **Logistic Regression**

---

## 🌐 Web App (Streamlit)

Features:

* Enter text into a text box
* Click **Predict** to get results
* Output shown with **colored messages**:

  * 🚨 Red = Hate Speech
  * ⚡ Yellow = Offensive
  * ✅ Green = Neutral

Bonus: Generates **random test samples** for quick testing.

---

## 🖥️ How to Run Locally

### 1. Clone Repository

```bash
git clone https://github.com/your-username/hate-speech-detector.git
cd hate-speech-detector
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # (Windows)
source venv/bin/activate  # (Mac/Linux)
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Streamlit App

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
hate-speech-detector/
│── app.py                # Streamlit app
│── best_model.pkl        # Trained ML model
│── tfidf_vectorizer.pkl  # TF-IDF vectorizer
│── requirements.txt      # Required Python libraries
│── README.md             # Documentation
│── data/                 # (Optional) Dataset files
│── notebooks/            # Jupyter notebooks for training
```

---

## 🚀 Future Improvements

* Add **Deep Learning (LSTM / BERT)** models for better accuracy.
* Deploy on **Heroku / AWS / Streamlit Cloud**.
* Create **REST API** for integration with other apps.
* Add support for **multilingual hate speech detection**.

---

## 🎯 Results & Demo

* ✅ Trained model with **92% accuracy**.
* ✅ Web app correctly classifies **unseen data**.
* ✅ Provides **real-time detection**.

---

## 👨‍💻 Contributors

* **Sidd Patel** – Developer, ML Engineer
* Dataset: Davidson et al. (2017), "Automated Hate Speech Detection and the Problem of Offensive Language"

---

