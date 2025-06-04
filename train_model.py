from classify_model import run, model, X_test, y_test
from sklearn.metrics import classification_report
import torch
import numpy as np

# Huấn luyện mô hình
run(250)

# Load mô hình tốt nhất
model.load_state_dict(torch.load("save_weights.pt"))
model.eval()

# Đánh giá trên tập test
with torch.no_grad():
    preds = model(X_test)
    preds = preds.detach().cpu().numpy()
preds = np.argmax(preds, axis=1)

print(classification_report(y_test, preds))
