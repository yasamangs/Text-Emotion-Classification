import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.preprocessing import LabelEncoder


@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        "yasamangs/FineTuneBert")
    return tokenizer, model
    

# Encode labels
label_encoder = LabelEncoder()
tokenizer, model = get_model()

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

d = {
    0: 'anger', 
    1: 'fear', 
    2: 'joy', 
    3: 'love', 
    4: 'sad', 
    5: 'suprise'
}

if user_input and button:
    test_sample = tokenizer([user_input], padding=True,
                            truncation=True, max_length=64, return_tensors='pt')
    # test_sample
    output = model(**test_sample)
    st.write("Logits: ", output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)

    st.write("Prediction: ", d[y_pred[0]])
