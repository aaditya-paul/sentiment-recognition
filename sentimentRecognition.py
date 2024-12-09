import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# add more data to the dictionary
data = {
    "text":[
        "I fucking Love my wife.",
        "I hate my job.",
        "I had a breakup recently.",
        "I am so happy i got my promotion.",
        "I am so angry.",
        "I went out for pandel hopping today",
        "I am feeling happy for some reasons",
        "My boss is a jerk.",
        "Tomorrow is the due date for my project.",
        "The festival is almost over",
        "Life has no meaning.",
        "i am in love"
    ],
    "sentiment":[
"positive",
"negative",
"negative",
"positive",
"negative",
"positive",
"positive",
"negative",
"negative",
"negative",
"negative",
"positive"
    ]
}

# create a dataframe
df = pd.DataFrame(data=data)
# print(df)

# Download the stopwords from NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to preprocess text data
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    words = [word for word in text.split() if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

# Apply preprocessing to the text data
df['processed_text'] = df['text'].apply(preprocess_text)

# convert text data to numerical data for input
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(df['processed_text'])
y = df['sentiment']

# Split the data into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = MultinomialNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

# print(f"Accuracy: {accuracy - 100:.2f}%")

def predict_sentiment(text):
    processed_text = preprocess_text(text)
    text_features = vectorizer.transform([processed_text])
    prediction = model.predict(text_features)
    return prediction[0]

# Test prediction function with custom input
sample_text = input("Enter the text: ")
print(f"Sentiment: {predict_sentiment(sample_text)}")