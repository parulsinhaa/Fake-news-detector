# 📰 Fake News Detection using Machine Learning

This project detects whether a news article is **real or fake** using a machine learning model trained on publicly available datasets. It uses natural language processing techniques to vectorize the article content and classify it using a PassiveAggressiveClassifier.

## 📌 Features
- Classifies news articles as **Fake (0)** or **Real (1)**
- Uses **CountVectorizer** for text processing
- Trained on labeled data from Kaggle
- Accuracy & confusion matrix printed after training
- Custom text prediction supported

## 🧠 Algorithms & Techniques
- **Text Vectorization:** CountVectorizer (bag-of-words)
- **Classifier:** Passive Aggressive Classifier (from `sklearn`)
- **Evaluation Metrics:** Accuracy, Confusion Matrix

## 📂 Dataset
Dataset used:  
- [Fake and Real News Dataset - Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
- Files used: `Fake.csv`, `True.csv`

## 🛠️ Technologies
- Python  
- Pandas  
- Scikit-learn

## 🚀 How to Run
1. Clone this repo:
git clone https://github.com/your-username/fake-news-detector.git

2. Navigate to the folder and install dependencies:
pip install pandas scikit-learn
3. Place `Fake.csv` and `True.csv` in the same directory
4. Run:
python main.py
Accuracy: 0.93
Confusion Matrix:
[[942 35]
[ 46 977]]
Prediction (1 = Real, 0 = Fake): Real
## ✍️ Author
**Parul Sinha**  
LinkedIn: [@parul-sinha](https://www.linkedin.com/in/parul-sinha)  
GitHub: [@parulsinhaa](https://github.com/parulsinhaa)
