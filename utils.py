import numpy as np
import os
import json
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score,mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random

def denormalize(tensor, mean, std):
    
    tensor = tensor * torch.tensor(std).view(1, 3, 1, 1) + torch.tensor(mean).view(1, 3, 1, 1)
    return tensor

def extract_hla(image):
    image = np.array(image)  # PIL → numpy (H, W, 3)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    h = hsv[:, :, 0:1]    # Hue channel
    l = lab[:, :, 0:1]    # Lightness
    a = lab[:, :, 1:2]    # A (red-green)
    
    hla = np.concatenate([h, l, a], axis=2)  # Shape: (H, W, 3)
    hla = Image.fromarray(hla.squeeze().astype(np.uint8)) if hla.shape[2] == 1 else Image.fromarray(hla.astype(np.uint8))
    return hla

def get_metrics(results,threshold=12):
    
    #res = np.array(results).squeeze(1)
    res = results
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
    print(f"MAE: {np.mean(np.abs(results[:,0]-results[:,1])):.3f} g/dL")
    


def plot_image_pairs(data, n_rows=3, n_cols=3):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()

    # Randomly select N samples (N = n_rows * n_cols)
    sample_data = random.sample(data, min(len(data), n_rows * n_cols))

    for idx, ((img1, img2), result, actual) in enumerate(sample_data):
        # Resize to same height if needed
        if img1.size[1] != img2.size[1]:
            height = min(img1.size[1], img2.size[1])
            img1 = img1.resize((img1.width, height))
            img2 = img2.resize((img2.width, height))

        # Stitch horizontally
        total_width = img1.width + img2.width
        stitched = Image.new("RGB", (total_width, img1.height))
        stitched.paste(img1, (0, 0))
        stitched.paste(img2, (img1.width, 0))

        # Plot
        threshold = 12
        Res = f"Predicted- {'Positive' if result[0] < threshold else 'Negative'}\nActual- {'Positive' if actual[0] < threshold else 'Negative'}"
        ax = axes[idx]
        ax.imshow(stitched)
        ax.set_title(Res, fontsize=10)
        ax.axis('off')

    # Hide unused subplots (in case len(data) < n_rows * n_cols)
    for ax in axes[len(sample_data):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()