# Next Word Prediction Using LSTM

## Project Description
This project develops a deep learning model to predict the next word in a given sequence of words. The model leverages Long Short-Term Memory (LSTM) networks, which excel at sequence prediction tasks. 

The project demonstrates the end-to-end process of building, training, and deploying a natural language processing application.

---

## Project Overview

### 1. Data Collection
The dataset is derived from the text of Shakespeare's *Hamlet*. This rich and complex text presents a challenging and meaningful context for the model to learn and predict.

### 2. Data Preprocessing
- The text is tokenized and converted into sequences of numerical data.
- Sequences are padded to ensure uniform input lengths.
- The data is split into training and testing sets for model validation.

### 3. Model Building
- An **LSTM-based model** is designed with:
  - An **embedding layer** to convert words into dense vector representations.
  - Two **LSTM layers** to capture the sequential dependencies of the text.
  - A **dense output layer** with a softmax activation function for predicting the probabilities of the next word.

### 4. Model Training
- The model is trained on the prepared sequences.
- **Early stopping** is implemented to monitor validation loss and halt training when no improvement is observed, mitigating overfitting.

### 5. Model Evaluation
- The trained model is tested on example sentences to evaluate its performance in accurately predicting the next word.

### 6. Deployment
- A **Streamlit web application** is developed, allowing users to input a sequence of words and get real-time predictions for the next word.

---

## Features
- **Interactive Prediction**: Users can type a sequence and get the next word prediction instantly.
- **Efficient Training**: Implements early stopping for optimized training.
- **Customizable Data**: Designed to work with different datasets for tailored predictions.

---

## How to Run the Project
### Prerequisites
Ensure the following are installed:
- Python 3.x
- TensorFlow/Keras
- Streamlit
- NumPy
- NLTK

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/hoepression/LSTM--Text-Predictor-.git
   cd LSTM--Text-Predictor-
