import streamlit as st
import pickle

model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

st.title("SMS Spam Detection")

msg = st.text_input("Enter a message")

if st.button("Check"):
    data = vectorizer.transform([msg])
    result = model.predict(data)

    if result[0] == 1:
        st.error("Spam Message")
    else:
        st.success("Not Spam")