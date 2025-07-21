# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 2: Load datasets
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

# Step 3: Label the data
fake_df['label'] = 0  # Fake = 0
real_df['label'] = 1  # Real = 1

# Step 4: Combine and shuffle
data = pd.concat([fake_df[['text', 'label']], real_df[['text', 'label']]], axis=0)
data = data.sample(frac=1).reset_index(drop=True)  # Shuffle

# Step 5: Train-test split
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Vectorize the text
vectorizer = CountVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 7: Train model
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_vec, y_train)

# Step 8: Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 9: Test with your own input
test_text = ["India’s Prime Minister announces a new AI policy for education."]  # Change this
test_vec = vectorizer.transform(test_text)
prediction = model.predict(test_vec)
print("Prediction:", "Real" if prediction[0] == 1 else "Fake")