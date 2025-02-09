import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
import numpy as np
0
def generate_text_gpt2(prompt, max_length=100):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def build_lstm_model(vocab_size, embedding_dim=64, lstm_units=128):
    model = Sequential([
        Embedding(vocab_size, embedding_dim),
        LSTM(lstm_units, return_sequences=True),
        LSTM(lstm_units),
        Dense(vocab_size, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    user_prompt = "Artificial intelligence is transforming the world by"
    generated_text = generate_text_gpt2(user_prompt)
    print("GPT-2 Generated Text:\n", generated_text)
    

