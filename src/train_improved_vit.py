"""
Novel Improved ViT with attention mechanisms (Target: >95%)
Balanced training without synthetic data
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from config import IMPROVED_VIT_CONFIG, DEVICE, OUTPUT_DIR
from src.models import ImprovedStrokeViT
from src.utils import (EarlyStopping, MetricsTracker, compute_metrics, 
                      CostSensitiveLoss, find_optimal_threshold)
from src.data_preparation import create_dataloaders

def train_improved_vit(normal_dir, stroke_dir, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(OUTPUT_DIR, 'models')
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print("Training BALANCED Improved ViT (Target: >95% Accuracy)")
    print("NOVELTY: Attention Pooling + Multi-scale + Cost-Sensitive")
    print("=" * 70)
    
    train_loader, val_loader, test_loader, file_splits = create_dataloaders(
        normal_dir, stroke_dir,
        batch_size=IMPROVED_VIT_CONFIG['batch_size'],
        img_size=224,
        balance_method='balanced_batch'
    )
    
    model = ImprovedStrokeViT(
        model_name='google/vit-base-patch16-224-in21k',
        num_labels=2,
        unfreeze_layers=IMPROVED_VIT_CONFIG['unfreeze_layers'],
        dropout=IMPROVED_VIT_CONFIG['dropout']
    ).to(DEVICE)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    # Cost-sensitive for main, standard for auxiliary
    criterion_main = CostSensitiveLoss(
        class_weights=IMPROVED_VIT_CONFIG['class_weights'],
        fn_cost=IMPROVED_VIT_CONFIG['false_negative_cost']
    )
    criterion_aux = nn.CrossEntropyLoss(
        weight=torch.tensor(IMPROVED_VIT_CONFIG['class_weights']).to(DEVICE),
        label_smoothing=0.1
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=IMPROVED_VIT_CONFIG['lr'],
                           weight_decay=IMPROVED_VIT_CONFIG['weight_decay'])
    
    def lr_lambda(epoch):
        if epoch < IMPROVED_VIT_CONFIG['warmup_epochs']:
            return (epoch + 1) / IMPROVED_VIT_CONFIG['warmup_epochs']
        else:
            progress = (epoch - IMPROVED_VIT_CONFIG['warmup_epochs']) / \
                      (IMPROVED_VIT_CONFIG['epochs'] - IMPROVED_VIT_CONFIG['warmup_epochs'])
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    early_stopping = EarlyStopping(patience=IMPROVED_VIT_CONFIG['early_stopping_patience'], mode='max')
    metrics = MetricsTracker('Improved_ViT_Balanced')
    scaler = GradScaler()
    
    best_val_f1 = 0.0
    best_model_path = os.path.join(save_dir, 'improved_vit_best.pth')
    optimal_threshold = 0.5
    
    for epoch in range(IMPROVED_VIT_CONFIG['epochs']):
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with autocast():
                main_out, aux_out = model(images)
                
                loss_main = criterion_main(main_out, labels)
                loss_aux = criterion_aux(aux_out, labels)
                loss = loss_main + 0.4 * loss_aux  # Higher weight to auxiliary
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
            _, predicted = torch.max(main_out, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
        train_loss = np.mean(train_losses)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_losses = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                with autocast():
                    main_out, _ = model(images)
                    loss = criterion_main(main_out, labels)
                
                val_losses.append(loss.item())
                probs = torch.softmax(main_out, dim=1)[:, 1]
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        optimal_threshold = find_optimal_threshold(all_labels, all_probs, target_recall=0.95)
        all_preds = (all_probs >= optimal_threshold).astype(int)
        
        val_loss = np.mean(val_losses)
        val_metrics = compute_metrics(all_labels, all_preds, all_probs)
        val_metrics['threshold'] = optimal_threshold
        
        metrics.update(
            train_loss=train_loss, train_acc=train_acc,
            val_loss=val_loss, val_acc=val_metrics['accuracy'],
            val_precision=val_metrics['precision'],
            val_recall=val_metrics['recall'],
            val_f1=val_metrics['f1_score'],
            val_specificity=val_metrics['specificity']
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:02d}/{IMPROVED_VIT_CONFIG['epochs']}] LR:{current_lr:.2e} | "
              f"Train:{train_acc:.4f} | Val Acc:{val_metrics['accuracy']:.4f} "
              f"F1:{val_metrics['f1_score']:.4f} Rec:{val_metrics['recall']:.4f} "
              f"Spec:{val_metrics['specificity']:.4f} Thr:{optimal_threshold:.3f}")
        
        scheduler.step()
        
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metrics['f1_score'],
                'val_metrics': val_metrics,
                'optimal_threshold': optimal_threshold,
                'config': IMPROVED_VIT_CONFIG,
            }, best_model_path)
            print(f"  -> Saved best (Val F1:{val_metrics['f1_score']:.4f})")
            metrics.best_metrics = val_metrics
            metrics.optimal_threshold = optimal_threshold
        
        early_stopping(val_metrics['f1_score'])
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"\nCompleted! Best Val F1:{best_val_f1:.4f}, Threshold:{metrics.optimal_threshold:.4f}")
    
    curves_path = os.path.join(OUTPUT_DIR, 'figures', 'improved_vit_training_curves.png')
    metrics.plot_training_history(curves_path)
    
    return model, metrics, best_model_path, file_splits

def evaluate_improved_vit(model_path, data_loader, file_paths=None, dataset_name="Test"):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    config = checkpoint.get('config', IMPROVED_VIT_CONFIG)
    optimal_threshold = checkpoint.get('optimal_threshold', 0.5)
    
    model = ImprovedStrokeViT(
        num_labels=2,
        unfreeze_layers=config.get('unfreeze_layers', 10),
        dropout=config.get('dropout', 0.15)
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    all_attentions = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            main_out, _, attn_weights = model(images, return_features=True)
            
            probs = torch.softmax(main_out, dim=1)[:, 1]
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if attn_weights is not None:
                all_attentions.append(attn_weights.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= optimal_threshold).astype(int)
    
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    
    print(f"\n{'='*70}")
    print(f"IMPROVED ViT {dataset_name} Results (Threshold={optimal_threshold:.3f})")
    print(f"{'='*70}")
    print(f"Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Sensitivity: {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"F1-Score:    {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:     {metrics.get('roc_auc', 0):.4f}")
    print(f"{'='*70}")
    
    # SAVE PREDICTIONS FOR EVERY IMAGE
    from src.utils import save_predictions_to_csv, plot_confusion_matrix, plot_roc_curve
    
    if file_paths is not None:
        pred_path = os.path.join(OUTPUT_DIR, 'results', f'improved_vit_{dataset_name.lower()}_predictions.csv')
        save_predictions_to_csv(file_paths, all_labels, all_preds, all_probs, pred_path, optimal_threshold)
        print(f"Saved predictions for {len(file_paths)} images to CSV")
    
    # Save attention weights
    if all_attentions:
        attn_array = np.concatenate(all_attentions, axis=0)
        attn_path = os.path.join(OUTPUT_DIR, 'results', f'{dataset_name.lower()}_attention_maps.npy')
        np.save(attn_path, attn_array)
        print(f"Saved attention maps: {attn_path}")
    
    cm_path = os.path.join(OUTPUT_DIR, 'figures', f'improved_vit_{dataset_name.lower()}_confusion_matrix.png')
    plot_confusion_matrix(all_labels, all_preds, cm_path)
    
    roc_path = os.path.join(OUTPUT_DIR, 'figures', f'improved_vit_{dataset_name.lower()}_roc.png')
    plot_roc_curve(all_labels, all_probs, roc_path, f'Improved ViT {dataset_name}')
    
    return metrics, all_labels, all_preds, all_probs

if __name__ == "__main__":
    from config import NORMAL_DIR, STROKE_DIR
    
    model, metrics, model_path, file_splits = train_improved_vit(NORMAL_DIR, STROKE_DIR)
    
    _, val_loader, test_loader, _ = create_dataloaders(
        NORMAL_DIR, STROKE_DIR, batch_size=IMPROVED_VIT_CONFIG['batch_size']
    )
    
    evaluate_improved_vit(model_path, val_loader, file_splits[1], "Validation")
    evaluate_improved_vit(model_path, test_loader, file_splits[2], "Test")