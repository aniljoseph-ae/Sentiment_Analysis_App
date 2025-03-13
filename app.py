import os
import asyncio
import streamlit as st
import re
import nltk
import spacy
import torch
from nltk.corpus import stopwords
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import subprocess
import sys
from dotenv import load_dotenv

# Load API Key
load_dotenv()
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


os.environ["STREAMLIT_SERVER_PORT"] = "8501"

# Download and load stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Function to ensure the model is downloaded
def download_spacy_model():
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en-core-web-sm"])
        spacy.load("en_core_web_sm")

# Call the function at the start of the app
download_spacy_model()

# Load fine-tuned BERT model and tokenizer from Hugging Face Hub
model_name = "aniljoseph/subtheme_sentiment_BERT_finetuned"  # Replace with your actual model name on Hugging Face
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    st.write("✅ Model Loaded Successfully from Hugging Face!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# Define unwanted entity types
UNWANTED_ENTITIES = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}
UNWANTED_TERMS = {"days", "months", "years", "time", "weeks"}


def predict_sentiment(text):
    """Predict sentiment using the fine-tuned BERT model."""
    if not text.strip():
        return "neutral"  # Default to neutral if empty

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Convert predicted class to label
    labels = ["negative", "positive"]  # Assuming your model follows this order
    return labels[predicted_class]


def clean_text(text):
    """Preprocess text: lowercase, remove special characters, and stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


def extract_subthemes(text):
    """Extract meaningful noun phrases while filtering out unimportant ones."""
    doc = nlp(text)
    named_entities = {ent.text.strip(): ent.label_ for ent in doc.ents if ent.label_ not in UNWANTED_ENTITIES}
    noun_phrases = {chunk.text.strip(): "PRODUCT" for chunk in doc.noun_chunks}
    subthemes = {**named_entities, **noun_phrases}
    filtered_subthemes = {
        k: v for k, v in subthemes.items()
        if len(k.split()) > 1 and not any(term in k.lower() for term in UNWANTED_TERMS)
    }
    return filtered_subthemes


def analyze_review_detailed(review):
    """Analyze subthemes, extract relevant sentences, and determine sentiment."""
    doc = nlp(review)
    subthemes = extract_subthemes(review)
    sentences = list(doc.sents)

    if not subthemes:
        return {"No Subthemes Found": {"sentence": review, "subtheme_category": "N/A", "sentiment": "neutral"}}

    analysis_result = {}
    for subtheme, category in subthemes.items():
        relevant_sentence = next((sent.text for sent in sentences if subtheme.lower() in sent.text.lower()), "Not Found")
        sentiment = predict_sentiment(relevant_sentence) if relevant_sentence != "Not Found" else "neutral"
        formatted_subtheme = subtheme.title().replace("The ", "").strip()

        if formatted_subtheme in analysis_result:
            analysis_result[formatted_subtheme]["subtheme_category"] = category
        else:
            analysis_result[formatted_subtheme] = {
                "sentence": relevant_sentence,
                "subtheme_category": category,
                "sentiment": sentiment
            }
    return analysis_result

# --------------- Streamlit App UI ----------------
st.title("Subtheme Sentiment Analysis")
st.write("Analyze subthemes and their sentiment in user reviews.")

user_input = st.text_area("✍️ Please enter your review:")
if st.button("🔍 Analyze"):
    if user_input:
        result = analyze_review_detailed(user_input)
        st.subheader("📌 Results: Subtheme Sentiment Analysis")
        for subtheme, details in result.items():
            st.markdown(f"**🔹 Subtheme:** {subtheme} ({details['subtheme_category']})")
            st.markdown(f"**➤ Sentence:** {details['sentence']}")
            st.markdown(f"**➤ Sentiment:** {details['sentiment']}")
            st.write("---")
        st.write("### Debugging Information")
        st.write("Extracted Subthemes:", result)
    else:
        st.warning("⚠️ Please enter a review for analysis.")
