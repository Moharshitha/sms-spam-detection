import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

data = {
    'message': [
        'Win a free lottery now',
        'Call me later',
        'Congratulations you won a prize',
        'Lets meet tomorrow',
        'Free entry in contest',
        'How are you'
    ],
    'label': [1,0,1,0,1,0]
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])

model = MultinomialNB()
model.fit(X, df['label'])

pickle.dump(model, open('model.pkl','wb'))
pickle.dump(vectorizer, open('vectorizer.pkl','wb'))

print("Model created")