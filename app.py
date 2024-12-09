import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    if not text.strip():
        return "Please provide valid input."
    
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence matches max_sequence_len-1
    
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted)
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "Word not found in vocabulary."

# Streamlit app
st.title("Next Word Prediction With LSTM")

# Input field
input_text = st.text_area("Enter a sequence of words:", "To be or not to", height=100)

# Predict button
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] - 1  # Subtract 1 because we're predicting one step ahead
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.subheader(f"Predicted word: {next_word}")

# Display model information
st.subheader("Model Information:")
st.write(f"Input shape: {model.input_shape}")
st.write(f"Output shape: {model.output_shape}")
st.write(f"Number of layers: {len(model.layers)}")
