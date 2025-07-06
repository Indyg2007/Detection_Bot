# Detection_Sense_Bot.py
import os
import torch
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from logger import log_prediction
from charts import plot_label_confidence
from openai import OpenAI



# Initialize client (you can also use api_key param here if needed)

client = OpenAI(api_key="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")
#client = OpenAI(api_key=os.getenv("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"))

def ask_chatgpt(prompt_text):
    try:
        # Try GPT-4 first
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in detecting sarcasm, irony, and humor in text."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7
        )
        return "🧠 GPT-4: " + response.choices[0].message.content.strip()

    except Exception as e:
        # Fallback to GPT-3.5-turbo if GPT-4 fails (e.g. 404 error or no access)
        print("GPT-4 failed, falling back to GPT-3.5. Reason:", e)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in detecting sarcasm, irony, and humor in text."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7
        )
        return "🤖 GPT-3.5: " + response.choices[0].message.content.strip()

# === Streamlit Setup ===
st.set_page_config(page_title="Detection_Sense_Bot 😏", page_icon="😏", layout="wide")

st.markdown("""
<style>
@keyframes rainbow {
  0%{color:red;}25%{color:orange;}50%{color:green;}
  75%{color:blue;}100%{color:violet;}
}
h1.banner {
    font-size: 4rem;
    text-align: center;
    animation: rainbow 4s infinite;
    font-weight: bold;
    margin-bottom: 5px;
}
.main > div.block-container {
    max-width: 700px;
    padding: 2rem 3rem;
    background-color: white;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}
.result {
    font-size: 2rem;
    font-weight: 700;
    margin-top: 25px;
    padding: 15px 20px;
    border-radius: 20px;
    color: #FF5722;
    background-color: #ffece6;
    border: 2px solid #ff7043;
}
.confidence {
    font-weight: 500;
    font-size: 1.1rem;
    color: #444;
    margin-bottom: 20px;
}
</style>
<h1 class="banner">🎭 Detection Sense — Humor, Irony & Sarcasm 😏</h1>
""", unsafe_allow_html=True)

# === Load DistilBERT Model ===
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained('checkpoint-best-multiclass')
    tokenizer = DistilBertTokenizerFast.from_pretrained('tokenizer')
    return tokenizer, model

tokenizer, model = load_model()

# === Confetti fallback ===
def trigger_confetti_gif():
    st.image("confetti-gif-2.gif", use_container_width=True)

# === Prediction Function ===
def predict_with_model(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)[0].numpy()
    labels = list(model.config.id2label.values())
    return list(zip(labels, probs))

# === UI Logic ===
st.title("Let's try and detect Humor, Irony or Sarcasm using Detection Sense 😏")

user_input = st.text_input("Please enter your sentence here:")

method = st.radio("Choose detection method:", ["Use My Model", "Use ChatGPT", "Use Both (Hybrid)"])

if user_input:
    if method in ["Use My Model", "Use Both (Hybrid)"]:
        model_predictions = predict_with_model(user_input)
        log_prediction(user_input, model_predictions)
        plot_label_confidence(model_predictions)

        positive = [(label, prob) for label, prob in model_predictions if not label.startswith("not_") and prob >= 0.5]

        if positive:
            st.markdown(f'<div class="result">🔥 Detected (Model):</div>', unsafe_allow_html=True)
            for label, prob in positive:
                st.markdown(f"<div class='confidence'>✔ {label.title()} — {prob*100:.1f}%</div>", unsafe_allow_html=True)

            labels_detected = [label for label, _ in positive]
            if "sarcasm" in labels_detected:
                st.balloons()
            if "humor" in labels_detected:
                st.snow()
            if "irony" in labels_detected:
                st.toast("Irony detected 🌀", icon="🌀")
                trigger_confetti_gif()
        else:
            st.markdown('<p style="color:#999;">Model did not detect humor, sarcasm or irony.</p>', unsafe_allow_html=True)

    if method in ["Use ChatGPT", "Use Both (Hybrid)"]:
        st.markdown("### 🤖 ChatGPT says:")
        gpt_response = ask_chatgpt(
            f"Classify the following text into sarcasm, irony, or humor (or none): \"{user_input}\". Just give the label and reasoning.")
        st.success(gpt_response)

else:
    st.markdown('<p style="color:#999;">Start by typing something above.</p>', unsafe_allow_html=True)



