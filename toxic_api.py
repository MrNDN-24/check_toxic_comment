from flask import Flask, request, jsonify
from classify_model import model, tokenizer, rdrsegmenter
import torch
import numpy as np

model.load_state_dict(torch.load("save_weights.pt"))
model.eval()

app = Flask(__name__)

def predict_toxic(sentence):
    tokens = rdrsegmenter.tokenize(sentence)
    statement = ""
    for token in tokens:
        statement += " ".join(token)
    sentence = statement
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
    app.run(host="0.0.0.0", port=5001)
