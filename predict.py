import torch
import numpy as np
from classify_model import model, tokenizer, rdrsegmenter

model.load_state_dict(torch.load("save_weights.pt"))
model.eval()

def result(sentence):
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

print("🎉 Mô hình đã sẵn sàng! Nhập câu để kiểm tra toxic (0: bình thường, 1: độc hại)")
while True:
    sentence = input("Nhập câu bình luận (gõ 'exit' để thoát): ")
    if sentence.lower() == "exit":
        break
    print("➡️ Kết quả dự đoán:", result(sentence))
