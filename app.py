# === Imports ===
import gradio as gr
import joblib
import numpy as np
import torch
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === Load Models & Tokenizers ===
# --- Real/Fake Model
label_model = load_model("final_news_label_model.h5")
label_tokenizer = joblib.load("tokenizer.pkl")
label_encoder = joblib.load("source_label_encoder.pkl")

# --- Subject Classification Model
subject_model = load_model("cnn_model.h5")
subject_tokenizer = joblib.load("cnn_tokenizer.pkl")
subject_encoder = joblib.load("label_encoder.pkl")

# --- LLM (FLAN-T5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)

# === Inference Functions ===

def predict_fake_real(text):
    seq = label_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=300)
    pred = label_model.predict(padded)[0][0]
    return "FAKE" if pred > 0.5 else "REAL"

def predict_subject(text):
    seq = subject_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=300)
    pred = np.argmax(subject_model.predict(padded), axis=1)
    return subject_encoder.inverse_transform(pred)[0]

def build_prompt(text, label, subject):
    return (
        f"Article:\n{text}\n\n"
        f"Explain factually why this article might be considered {label.lower()} in the context of {subject.lower()} news."
    )

def generate_explanation(prompt):
    input_ids = llm_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output = llm_model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )
    return llm_tokenizer.decode(output[0], skip_special_tokens=True).strip()

def full_news_insight(text):
    label = predict_fake_real(text)
    subject = predict_subject(text)
    prompt = build_prompt(text, label, subject)
    explanation = generate_explanation(prompt)
    return label, subject, explanation

# === Gradio Interface ===
iface = gr.Interface(
    fn=full_news_insight,
    inputs=gr.Textbox(lines=10, placeholder="Paste a news article here...", label="📝 News Article"),
    outputs=[
        gr.Textbox(label="📰 Prediction (Real or Fake)"),
        gr.Textbox(label="📂 Subject Type"),
        gr.Textbox(label="💡 LLM Explanation")
    ],
    title="🧠 Fake News Detector with LLM Insight",
    description="Enter a news article. The system predicts if it's fake or real, classifies its type, and generates a short explanation using FLAN-T5."
)

iface.launch()
