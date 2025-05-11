import numpy as np
import os
import json
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def denormalize(tensor, mean, std):
    
    tensor = tensor * torch.tensor(std).view(1, 3, 1, 1) + torch.tensor(mean).view(1, 3, 1, 1)
    return tensor


def get_metrics(results,threshold=12):
    
    res = np.array(results).squeeze(1)
    res = np.where(np.round(res) < threshold, 1, 0)
    y_pred = res[:, 0]
    y_true = res[:, 1]

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    cm_sum = np.sum(cm)
    cm_percent = cm / cm_sum * 100  # convert to percentage

    # Plot
    fig, ax = plt.subplots()
    im = ax.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)

    # Titles and labels
    class_names = ['Positive', 'Negative']
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix (%)'
    )

    # Rotate the tick labels and set alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell with percentage
    fmt = '.1f'
    thresh = cm_percent.max() / 2.
    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            ax.text(j, i, format(cm_percent[i, j], fmt) + '%',
                    ha="center", va="center",
                    color="white" if cm_percent[i, j] > thresh else "black")

    fig.tight_layout()
    plt.colorbar(im)
    plt.show()
    # 2. Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)  # sensitivity = recall
    f1 = f1_score(y_true, y_pred)

    # Calculate specificity manually
    TN = cm[1, 1]
    FP = cm[1, 0]
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    # Print metrics
    print(f"Accuracy: {(accuracy * 100):.3f}%")
    print(f"Precision: {precision:.3f}")
    print(f"Sensitivity (Recall): {recall:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"F1 Score: {f1:.3f}")