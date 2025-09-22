import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detection System")

news_text = st.text_area("Paste news content here:")

if st.button("Check"):
    if news_text.strip() == "":
        st.warning("Please enter some text")
    else:
        # Transform and predict
        x = vectorizer.transform([news_text])
        prediction = model.predict(x)[0]

        if prediction == 0:
            st.error("❌ This looks like FAKE news!")
            st.write("🔍 Mitigation: Please fact-check on trusted sites like Google News or Snopes.")
        else:
            st.success("✅ This looks like REAL news!")
            st.write("ℹ️ Source seems trustworthy.")
