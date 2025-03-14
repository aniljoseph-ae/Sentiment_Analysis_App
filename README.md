# Subtheme Sentiment Analysis App

## Overview
The **Subtheme Sentiment Analysis App** is a Streamlit-based web application that analyzes user reviews by identifying subthemes and determining their sentiment using a **fine-tuned BERT model**. The model has been trained on the **Amazon Polarity dataset** and is hosted on **Hugging Face Hub** to optimize space and computational cost.

## Features
- **Text Cleaning and Preprocessing**: Tokenization, stopword removal, and special character filtering.
- **Subtheme Extraction**: Identifies named entities and noun phrases using **spaCy's Named Entity Recognition (NER)** and chunking.
- **Sentiment Prediction**: Uses a **fine-tuned BERT model** to classify subtheme sentiments into **positive** or **negative**.
- **Streamlit Web Interface**: A user-friendly UI for easy interaction and real-time sentiment analysis.

## Model
The fine-tuned **BERT model** is stored on Hugging Face Hub under the repository:
```
aniljoseph/subtheme_sentiment_BERT_finetuned
```

## Installation and Setup
### 1. Clone the Repository
```sh
git clone https://github.com/aniljoseph-ae/Sentiment_Analysis_App.git
cd Sentiment_Analysis_App
```

### 2. Create a Virtual Environment (Optional but Recommended)
```sh
python -m venv venv
source venv/bin/activate  # For MacOS/Linux
venv\Scripts\activate    # For Windows
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Set up Environment Variables
Create a `.env` file in the project root and add the following:
```sh
HUGGINGFACE_ACCESS_TOKEN=your_huggingface_api_key
```

### 5. Run the Application
```sh
streamlit run app.py
```

## Application Workflow
### 1. **Text Preprocessing**
- Converts input text to lowercase.
- Removes special characters and stopwords.
- Tokenizes text into words and sentences.

### 2. **Subtheme Extraction**
- Identifies **named entities** (e.g., products, organizations) using **spaCy's Named Entity Recognition (NER)**.
- Extracts **noun phrases** using a **chunk parser**.
- Filters out unimportant entities like dates, numbers, and general time expressions.

### 3. **Sentiment Analysis with BERT**
- The preprocessed subthemes are classified as **positive** or **negative**.
- The model predicts sentiment using the fine-tuned **BERT model from Hugging Face**.

### 4. **Displaying Results**
- Extracted subthemes are displayed along with their **category, relevant sentence, and sentiment**.
- Sentiments are color-coded for easy visualization.

## Code Structure
```
Sentiment_Analysis_App/
│-- models/
│   ├── sentiment_model.py         # Model loading and inference
│-- app.py                         # Main Streamlit app
│-- requirements.txt               # Required dependencies
│-- .env.example                   # Sample environment file
│-- README.md                      # Documentation
```

## Example Usage
### Input:
```
The battery life of this phone is terrible, but the camera quality is amazing!
```
### Output:
| Subtheme       | Category  | Sentence                                  | Sentiment |
|---------------|-----------|-------------------------------------------|-----------|
| Battery Life  | PRODUCT   | The battery life of this phone is terrible | Negative  |
| Camera Quality | PRODUCT  | The camera quality is amazing!             | Positive  |

## Deployment
The app is deployed on **Streamlit Cloud**:
[Streamlit App](https://your-streamlit-app-link)

## Contributing
Feel free to fork the repository and contribute. Submit pull requests for improvements and bug fixes.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---
Developed by **Anil Joseph**

