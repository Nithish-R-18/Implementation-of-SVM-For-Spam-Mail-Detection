# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Read the spam dataset and preprocess the data by selecting message and label columns.

2. Convert text messages into numerical form using TF-IDF Vectorizer.

3. Split the dataset into training and testing data, then train the SVM classifier.

4. Predict the output using the trained model and display the confusion matrix using a heatmap.


## Program:
```python
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: NITHISH R
RegisterNumber: 212225230202
*/

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = {
    'v1': ['ham','ham','spam','ham','ham'],
    'v2': [
        'Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...',
        'Ok lar... Joking wif u oni...',
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
        'U dun say so early hor... U c already then say...',
        'Nah I don\'t think he goes to usf, he lives around here though'
    ]
}

df = pd.DataFrame(data)

df['label'] = df['v1'].map({'ham':0, 'spam':1})

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['v2'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

new_message = ["Congratulations! You have won a free ticket to Bahamas. Call now!"]
new_message_vect = vectorizer.transform(new_message)
prediction = svm_model.predict(new_message_vect)
print(f"Prediction: {'Spam' if prediction[0]==1 else 'Ham'}")

```

## Output:

<img width="576" height="329" alt="ml ex 12" src="https://github.com/user-attachments/assets/38e0318f-c31f-49b7-84d9-60b976eda5af" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
