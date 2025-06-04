from flask import Flask, request, jsonify
from classify_model import model, tokenizer
import torch
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gdown
from underthesea import word_tokenize


if not os.path.exists("save_weights.pt"):
    file_id = "1JeQ100QELbCCjCozF5SsHT1ca08Vvfuw"  
    url = f"https://drive.google.com/uc?id={file_id}"
    print("Downloading save_weights.pt from Google Drive...")
    gdown.download(url, "save_weights.pt", quiet=False)
    if not os.path.exists("save_weights.pt"):
        raise FileNotFoundError("Failed to download save_weights.pt")

model.load_state_dict(torch.load("save_weights.pt"))
model.eval()

app = Flask(__name__)

def predict_toxic(sentence):
    # tokens = rdrsegmenter.tokenize(sentence)
    # statement = ""
    # for token in tokens:
    #     statement += " ".join(token)
    # sentence = statement
    sentence = word_tokenize(sentence, format="text")
    sequence = tokenizer.encode(sentence)
    while len(sequence) == 20:
        sequence.insert(0, 0)
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
    print("Dự đoán:", result, type(result))
    return jsonify({'toxic': int(result)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    # app.run(host="0.0.0.0", port=5001)