"""
Improved model architectures for high accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTModel

class StrokeCNN(nn.Module):
    """
    Deeper CNN with residual connections
    """
    def __init__(self, dropout=0.3):
        super(StrokeCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout/2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout/2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout/2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class StrokeViT(nn.Module):
    """
    Fine-tuned ViT with custom classifier
    """
    def __init__(self, model_name='google/vit-base-patch16-224-in21k', num_labels=2, 
                 unfreeze_layers=6, dropout=0.1):
        super(StrokeViT, self).__init__()
        
        self.vit = ViTModel.from_pretrained(model_name, add_pooling_layer=True)
        self.dropout = nn.Dropout(dropout)
        
        # Custom deep classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )
        
        # Freeze all first
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # Unfreeze last N layers
        if unfreeze_layers > 0:
            for layer in self.vit.encoder.layer[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            # Unfreeze pooler
            for param in self.vit.pooler.parameters():
                param.requires_grad = True
                
    def forward(self, x):
        outputs = self.vit(x)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

class ImprovedStrokeViT(nn.Module):
    """
    Novel architecture: ViT with attention pooling and multi-head classifier
    """
    def __init__(self, model_name='google/vit-base-patch16-224-in21k', num_labels=2,
                 unfreeze_layers=6, dropout=0.1):
        super(ImprovedStrokeViT, self).__init__()
        
        self.vit = ViTModel.from_pretrained(model_name, add_pooling_layer=False)
        self.dropout = nn.Dropout(dropout)
        
        hidden_size = self.vit.config.hidden_size
        
        # NOVELTY 1: Attention-based pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # NOVELTY 2: Multi-scale feature extraction
        self.patch_processor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # NOVELTY 3: Deep classifier with skip connections
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_labels)
        )
        
        # Auxiliary classifier for deep supervision
        self.aux_classifier = nn.Linear(hidden_size, num_labels)
        
        # Freeze strategy
        for param in self.vit.parameters():
            param.requires_grad = False
            
        if unfreeze_layers > 0:
            for layer in self.vit.encoder.layer[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
    
    def forward(self, x, return_features=False):
        outputs = self.vit(x, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Global average pooling of all tokens except CLS
        patch_tokens = hidden_states[:, 1:]  # [batch, seq_len-1, hidden]
        cls_token = hidden_states[:, 0]  # [batch, hidden]
        
        # Attention pooling
        attn_weights = F.softmax(self.attention_pool(patch_tokens), dim=1)  # [batch, seq_len-1, 1]
        attended_features = (patch_tokens * attn_weights).sum(dim=1)  # [batch, hidden]
        
        # Process patches
        processed_patches = self.patch_processor(attended_features)
        
        # Combine CLS and attended features
        combined = torch.cat([cls_token, processed_patches], dim=1)
        
        # Main prediction
        main_logits = self.classifier(combined)
        
        # Auxiliary prediction
        aux_logits = self.aux_classifier(cls_token)
        
        if return_features:
            return main_logits, aux_logits, attn_weights
        return main_logits, aux_logits

def get_model(model_type='cnn', **kwargs):
    if model_type == 'cnn':
        return StrokeCNN(dropout=kwargs.get('dropout', 0.3))
    elif model_type == 'vit':
        return StrokeViT(
            model_name=kwargs.get('model_name', 'google/vit-base-patch16-224-in21k'),
            num_labels=kwargs.get('num_labels', 2),
            unfreeze_layers=kwargs.get('unfreeze_layers', 6),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif model_type == 'improved_vit':
        return ImprovedStrokeViT(
            model_name=kwargs.get('model_name', 'google/vit-base-patch16-224-in21k'),
            num_labels=kwargs.get('num_labels', 2),
            unfreeze_layers=kwargs.get('unfreeze_layers', 6),
            dropout=kwargs.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")