import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🐦", layout="wide")

@st.cache_resource
def load_model():
    device = torch.device('cpu')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.load_state_dict(torch.load('best_roberta_final.pt', map_location=device))
    model.eval()
    return model, tokenizer, device

try:
    model, tokenizer, device = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Model loading failed: {e}")
    model_loaded = False

def predict_sentiment(text):
    if not model_loaded:
        return None, None
    
    encoding = tokenizer.encode_plus(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device)
        )
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment, confidence

st.title("Twitter Sentiment Analysis")
st.markdown("RoBERTa-based classifier with 85.53% test accuracy")

with st.sidebar:
    st.header("Model Information")
    st.write("Architecture: RoBERTa-base")
    st.write("Dataset: Sentiment140")
    st.write("Training samples: 100,000")
    st.write("Test accuracy: 85.53%")
    st.write("Precision: 85.60%")
    st.write("Recall: 85.53%")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Analysis"])

with tab1:
    st.subheader("Analyze Tweet")
    
    text_input = st.text_area(
        "Enter text:",
        height=100,
        placeholder="Type your tweet here"
    )
    
    if st.button("Analyze"):
        if text_input:
            with st.spinner("Processing..."):
                sentiment, confidence = predict_sentiment(text_input)
                
                if sentiment:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Sentiment", sentiment)
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    st.progress(confidence)
        else:
            st.warning("Please enter text")

with tab2:
    st.subheader("Batch Analysis")
    
    uploaded_file = st.file_uploader("Upload CSV with 'text' column", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if 'text' in df.columns:
            st.write(f"Loaded {len(df)} rows")
            
            if st.button("Analyze All"):
                results = []
                progress_bar = st.progress(0)
                
                for idx, row in df.iterrows():
                    sent, conf = predict_sentiment(str(row['text']))
                    results.append({
                        'text': row['text'],
                        'sentiment': sent,
                        'confidence': conf
                    })
                    progress_bar.progress((idx + 1) / len(df))
                
                results_df = pd.DataFrame(results)
                
                col1, col2 = st.columns(2)
                with col1:
                    positive = (results_df['sentiment'] == 'Positive').sum()
                    st.metric("Positive", positive)
                
                with col2:
                    negative = (results_df['sentiment'] == 'Negative').sum()
                    st.metric("Negative", negative)
                
                fig = px.pie(
                    results_df,
                    names='sentiment',
                    title='Sentiment Distribution'
                )
                st.plotly_chart(fig)
                
                st.dataframe(results_df)
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results",
                    csv,
                    "results.csv",
                    "text/csv"
                )
        else:
            st.error("CSV must contain 'text' column")
            