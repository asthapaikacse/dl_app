"""
Data preparation with class balancing techniques
NO synthetic data - only valid sampling strategies
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import math

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class StrokeDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class BalancedBatchSampler(Sampler):
    """
    Method 2: Balanced Batch Sampling
    Ensures each batch has equal Normal and Stroke samples
    NO synthetic data - just smart sampling
    """
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        
        self.normal_indices = np.where(self.labels == 0)[0]
        self.stroke_indices = np.where(self.labels == 1)[0]
        
        # Number of batches per epoch
        self.num_batches = max(len(self.normal_indices), len(self.stroke_indices)) * 2 // batch_size
        self.length = self.num_batches * batch_size
        
    def __iter__(self):
        # Shuffle indices
        normal_perm = np.random.permutation(self.normal_indices)
        stroke_perm = np.random.permutation(self.stroke_indices)
        
        # Cycle through minority class if needed
        normal_cycle = np.tile(normal_perm, math.ceil(self.length / 2 / len(normal_perm)))
        stroke_cycle = np.tile(stroke_perm, math.ceil(self.length / 2 / len(stroke_perm)))
        
        # Interleave: Normal, Stroke, Normal, Stroke...
        balanced_indices = []
        for i in range(self.length // 2):
            balanced_indices.append(normal_cycle[i])
            balanced_indices.append(stroke_cycle[i])
            
        return iter(balanced_indices)
    
    def __len__(self):
        return self.length

def get_data_paths(normal_dir, stroke_dir):
    normal_files = sorted([os.path.join(normal_dir, f) for f in os.listdir(normal_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    stroke_files = sorted([os.path.join(stroke_dir, f) for f in os.listdir(stroke_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    files = normal_files + stroke_files
    labels = [0] * len(normal_files) + [1] * len(stroke_files)
    
    print(f"\nDataset Statistics:")
    print(f"  Normal images: {len(normal_files)} ({len(normal_files)/len(files)*100:.1f}%)")
    print(f"  Stroke images: {len(stroke_files)} ({len(stroke_files)/len(files)*100:.1f}%)")
    print(f"  Imbalance ratio: 1:{len(normal_files)/len(stroke_files):.2f}")
    
    return files, labels

def stratified_split(files, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        files, labels, test_size=(val_ratio + test_ratio), 
        random_state=seed, stratify=labels
    )
    
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=(1 - val_ratio_adjusted),
        random_state=seed, stratify=temp_labels
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_files)} (N:{train_labels.count(0)}, S:{train_labels.count(1)})")
    print(f"  Val:   {len(val_files)} (N:{val_labels.count(0)}, S:{val_labels.count(1)})")
    print(f"  Test:  {len(test_files)} (N:{test_labels.count(0)}, S:{test_labels.count(1)})")
    
    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)

def get_transforms(img_size=224, is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_dataloaders(normal_dir, stroke_dir, batch_size=16, img_size=224, 
                       balance_method='balanced_batch'):
    """
    Create dataloaders with class balancing
    
    balance_method: 'none', 'weighted_sampler', 'balanced_batch'
    """
    
    files, labels = get_data_paths(normal_dir, stroke_dir)
    
    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = \
        stratified_split(files, labels)
    
    train_dataset = StrokeDataset(train_files, train_labels, 
                                  transform=get_transforms(img_size, is_training=True))
    val_dataset = StrokeDataset(val_files, val_labels,
                                transform=get_transforms(img_size, is_training=False))
    test_dataset = StrokeDataset(test_files, test_labels,
                                 transform=get_transforms(img_size, is_training=False))
    
    # Balanced training sampling
    if balance_method == 'balanced_batch':
        # Method 2: Balanced batches (equal N and S in each batch)
        train_sampler = BalancedBatchSampler(train_labels, batch_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                  sampler=train_sampler, num_workers=0, pin_memory=True)
    elif balance_method == 'weighted_sampler':
        # Alternative: Weighted random sampling
        class_counts = [train_labels.count(0), train_labels.count(1)]
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                  num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader, (train_files, val_files, test_files)