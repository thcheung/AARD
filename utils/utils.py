import re
import torch
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        # t = '' if t.startswith('#') and len(t) > 1 else t
        t = '' if t.startswith('@') and len(t) > 1 else t
        t = '' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def print_metrics(y_true, y_pred):
    print(f"Confusion matrix: \n {confusion_matrix(y_true, y_pred)}")
    print(f"F1 Score (Micro): {f1_score(y_true, y_pred, average='micro')}")
    print(f"F1 Score (Macro): {f1_score(y_true, y_pred, average='macro')}")
    print(
        f"F1 Score (Weighted): {f1_score(y_true, y_pred, average='weighted')}")
    print(f"Accuracy): {accuracy_score(y_true, y_pred)}")


def labels_to_weights(labels):
    num = max(labels) + 1
    counts = [labels.count(i) for i in range(0, num)]
    total = sum(counts)
    counts = [total/count for count in counts]
    return torch.tensor(counts, dtype=torch.float)

def labels_to_weights2(labels, labels2):
    num = max(labels) + 1
    counts = [labels.count(i)+1 for i in range(0, num)]
    total = sum(counts)
    counts2 = [labels2.count(i)+1 for i in range(0, num)]
    total2 = sum(counts2)
    counts = [counts2[idx]/counts[idx] for idx , count in enumerate(counts)]
    return torch.tensor(counts, dtype=torch.float)

def image_transforms():
    transforms = []
    transforms.append(T.Resize(384))
    transforms.append(T.CenterCrop(384))
    return T.Compose(transforms)

def label_to_value(label):
    VALUES = [0.0,-1.0, 1.0]
    value = VALUES[int(label)]
    value = np.asarray([value])
    value = torch.tensor(value, dtype=torch.float)
    return value
    
def cosine_distance(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)