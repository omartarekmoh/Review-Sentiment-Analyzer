from torch.utils.data import DataLoader
import torch
import re

def generate_batches(dataset, batch_size=32, shuffle=True, drop_last=True, device="cpu"):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    for data_dict in data_loader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = tensor.to(device)
        yield out_data_dict
        

def compute_accuracy(predictions, targets):
    predicted_labels = torch.round(torch.sigmoid(predictions))
    
    targets = targets.float()
    
    correct_predictions = (predicted_labels == targets).float().sum()
    accuracy = correct_predictions / targets.size(0)
    
    return accuracy.item()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.!?,])", r" \1 ", text)
    text = re.sub(r"[^A-Za-z?,.!]+", r" ", text)
    return text