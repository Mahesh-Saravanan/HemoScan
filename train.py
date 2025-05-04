import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os, sys
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dataloader import CustomDataloader
from config import Config
from model import ModelV1
from PIL import Image
import json
import matplotlib.pyplot as plt


def train(model = ModelV1(),dataloader = None,config = Config()):
   
    device = config.device
    batch_size = config.batch_size
    data_dir = config.data_dir
    transform = config.transform
    shuffle = config.shuffle
    num_epochs = config.num_epochs
    model = model.to(device)

        
    
    criterion = config.criterion
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)


    if dataloader is None:
        dataset = CustomDataloader(data_dir, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    

    loss_images_dir = config.log_path
    os.makedirs(loss_images_dir, exist_ok=True)
    loss_image_path = os.path.join(loss_images_dir, 'loss_plot.png')


    all_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        running_loss = 0.0

        total_batches = len(dataloader)
        for batch_idx, (img_pf, img_pj, targets) in enumerate(dataloader):
            img_pf = img_pf.to(device)
            img_pj = img_pj.to(device)
            targets = targets.to(device)

            outputs = model(img_pf, img_pj)
            targets = targets.unsqueeze(1)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_losses.append(loss.item())

            sys.stdout.write(f"\r[Train] Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{total_batches}], Loss: {loss.item():.4f}")
            sys.stdout.flush()
        
        
        all_losses.extend(epoch_losses)
        plt.figure(figsize=(10, 6))

        
        plt.plot(all_losses, label='Loss per Batch (all epochs)', color='blue')

        
        start_idx = len(all_losses) - len(epoch_losses)
        plt.plot(range(start_idx, len(all_losses)), epoch_losses, label=f"Epoch {epoch+1}", color='red', linewidth=2)

        plt.title(f"Loss per Batch (Epoch {epoch+1})")
        plt.xlabel('Batch Index')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.savefig(loss_image_path)
        plt.close()

        avg_loss = running_loss / total_batches
        print(f"  ------> Epoch [{epoch+1}/{num_epochs}] Completed. Avg Loss: {avg_loss:.4f}")

        if (epoch + 1) % config.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(config.model_save_path, f"model_epoch_{epoch+1}.pth"))
            
def regression_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE (%)': mape
    }

def validate(model, dataloader, config = Config()):

    model.eval()
    device = config.device
    criterion = config.criterion

    total_loss = 0.0
    total_batches = len(dataloader)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (img_pf, img_pj, targets) in enumerate(dataloader):
            img_pf = img_pf.to(device)
            img_pj = img_pj.to(device)
            targets = targets.to(device)

            outputs = model(img_pf, img_pj)
            targets = targets.unsqueeze(1)

            loss = criterion(outputs, targets)

            total_loss += loss.item()

            all_preds.append(outputs)
            all_targets.append(targets)

            
            sys.stdout.write(
                f"\r[Validation] Batch [{batch_idx+1}/{total_batches}] - Batch Loss: {loss.item():.4f}"
            )
            sys.stdout.flush()

    avg_loss = total_loss / total_batches
    print(f"\n[Validation] Completed. Avg Loss: {avg_loss:.4f}")

    
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = regression_metrics(targets, preds)

    print("\n[Validation Metrics]")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return avg_loss, metrics

def inference(model, path, mean = 12, std = 8, config = Config()):
    model.eval()

    path_pf = os.path.join(path, "PF.JPG")
    path_pb = os.path.join(path, "PB.JPG")

    device = config.device
    transform = config.transform

    results = []

    with torch.no_grad():
        
        img_pf = Image.open(path_pf).convert("RGB")
        img_pb = Image.open(path_pb).convert("RGB")

        if transform:
            img_pf = transform(img_pf)
            img_pb = transform(img_pb)

        img_pf = img_pf.unsqueeze(0).to(device)
        img_pb = img_pb.unsqueeze(0).to(device)

        output = model(img_pf, img_pb)
        output = output.cpu().numpy()[0]
        output = output * std + mean
        output = round(float(output),2)

        json_files = [file for file in os.listdir(path) if file.endswith(".json")]

        if json_files:
            json_path = os.path.join(path, json_files[0])
            with open(json_path, 'r') as f:
                data = json.load(f)
                actual_data = float(data['hb'])

        print(f"Predicted HB: {output:.2f}       | Actual HB: {actual_data:.2f}")
        