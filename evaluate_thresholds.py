import pandas as pd
import numpy as np

gt = pd.read_csv("dataset.csv", sep=";")
pred = pd.read_csv("app/output.csv", sep=";")

df = gt.merge(pred, on="file_name", suffixes=("_true","_pred"))

def accuracy_at(thr):
    correct = ((df["Is_fake_pred"] >= thr) == (df["Is_fake_true"] == 1)).sum()
    return correct / len(df)

ths = np.linspace(0.4, 0.8, 41)
accs = [(t, accuracy_at(t)) for t in ths]
best_thr, best_acc = max(accs, key=lambda x: x[1])
print(f"Best accuracy {best_acc:.3%} at threshold {best_thr:.2f}")
