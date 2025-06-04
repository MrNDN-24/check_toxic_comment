from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from classify_model import Classify
import torch
import numpy as np
import os
import gdown
from underthesea import word_tokenize

app = Flask(__name__)

# Tải mô hình và tokenizer khi khởi động ứng dụng
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    model = Classify(number_of_category=2)
    if not os.path.exists("save_weights.pt"):
        file_id = "1JeQ100QELbCCjCozF5SsHT1ca08Vvfuw"
        url = f"https://drive.google.com/uc?id={file_id}"
        print("Downloading save_weights.pt from Google Drive...")
        gdown.download(url, "save_weights.pt", quiet=False)
        if not os.path.exists("save_weights.pt"):
            raise FileNotFoundError("Failed to download save_weights.pt")
    model.load_state_dict(torch.load("save_weights.pt", map_location=torch.device('cpu')))
    model.eval()

def predict_toxic(sentence):
    sentence = word_tokenize(sentence, format="text")
    sequence = tokenizer.encode(sentence)
    if len(sequence) < 20:
        sequence = [0] * (20 - len(sequence)) + sequence
    else:
        sequence = sequence[:20]
    padded = torch.tensor([sequence])
    with torch.no_grad():
        preds = model(padded)
    preds = np.argmax(preds.cpu().numpy(), axis=1)
    return preds[0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sentence = data.get('sentence', '')
    if not sentence:
        return jsonify({'error': 'No sentence provided'}), 400
    result = predict_toxic(sentence)
    return jsonify({'toxic': int(result)})

if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
