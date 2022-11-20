import torch.nn as nn
from sklearn.metrics import f1_score as f1
import torch.nn.functional as nnf

#loss
def loss(outputs, targets):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    return loss

#metric
def f1_score(outputs, targets):
    outputs = outputs.cpu()
    targets = targets.cpu()
    prob = nnf.softmax(outputs, dim=1)
    top_p, top_class = prob.topk(1, dim=1)

    f1_score = f1(targets.data, top_class, average='weighted')

    return f1_score