import streamlit as st
from model import predict_spam

st.title("📩 Spam Message Detector")

msg = st.text_area("Enter your message:")

if st.button("Check"):
    if msg.strip() != "":
        result = predict_spam(msg)
        st.success(f"Result: {result}")
    else:
        st.warning("Please enter a message")