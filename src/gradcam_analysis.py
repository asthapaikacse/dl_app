"""
NOVELTY: Gradient-weighted Class Activation Mapping (Grad-CAM)
For visualizing model decisions on CT scans
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from config import DEVICE, OUTPUT_DIR
from src.models import StrokeCNN, StrokeViT, ImprovedStrokeViT

class GradCAM:
    """
    Grad-CAM for visualizing where the model looks
    Crucial for medical AI interpretability
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Weight the channels by gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # Global average pooling
        cam = (weights * activations).sum(dim=0)  # Weighted combination
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy(), target_class

def apply_colormap_on_image(org_img, activation, colormap=cv2.COLORMAP_JET):
    """Apply heatmap on image"""
    activation = cv2.resize(activation, (org_img.size[0], org_img.size[1]))
    activation = np.uint8(255 * activation)
    activation = cv2.applyColorMap(activation, colormap)
    activation = cv2.cvtColor(activation, cv2.COLOR_BGR2RGB)
    
    org_img = np.array(org_img)
    overlayed_img = org_img * 0.5 + activation * 0.5
    overlayed_img = overlayed_img.astype(np.uint8)
    
    return overlayed_img

def visualize_cnn_gradcam(model_path, image_path, save_path):
    """Generate Grad-CAM for CNN model"""
    
    # Load model
    model = StrokeCNN().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Target layer: last conv layer
    target_layer = model.conv3[0]  # Last conv layer
    
    grad_cam = GradCAM(model, target_layer)
    
    # Load and preprocess image
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    org_img = Image.open(image_path).convert('RGB')
    input_tensor = transform(org_img).unsqueeze(0).to(DEVICE)
    
    # Generate CAM
    cam, pred_class = grad_cam.generate_cam(input_tensor)
    
    # Visualize
    overlayed = apply_colormap_on_image(org_img, cam)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(org_img)
    axes[0].set_title('Original CT Scan')
    axes[0].axis('off')
    
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlayed)
    axes[2].set_title(f'Overlay (Pred: {"Stroke" if pred_class == 1 else "Normal"})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return pred_class

def visualize_vit_attention(model_path, image_path, save_path):
    """
    NOVELTY: Visualize attention maps from ViT
    Shows which patches the model focuses on
    """
    
    # Load model
    model = ImprovedStrokeViT(num_labels=2).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load image
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    org_img = Image.open(image_path).convert('RGB')
    input_tensor = transform(org_img).unsqueeze(0).to(DEVICE)
    
    # Get attention weights
    with torch.no_grad():
        outputs, _, attention_weights = model(input_tensor, return_attention=True)
        _, pred_class = torch.max(outputs, 1)
    
    # Reshape attention to spatial grid (14x14 for 224/16)
    if attention_weights is not None:
        attn_map = attention_weights[0, :, 0].reshape(14, 14).cpu().numpy()
        attn_map = cv2.resize(attn_map, (224, 224))
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(org_img)
        axes[0].set_title('Original CT Scan')
        axes[0].axis('off')
        
        axes[1].imshow(attn_map, cmap='hot')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        
        # Overlay
        attn_colored = cv2.applyColorMap(np.uint8(255 * attn_map / attn_map.max()), 
                                         cv2.COLORMAP_HOT)
        attn_colored = cv2.cvtColor(attn_colored, cv2.COLOR_BGR2RGB)
        overlayed = np.array(org_img.resize((224, 224))) * 0.6 + attn_colored * 0.4
        
        axes[2].imshow(overlayed.astype(np.uint8))
        axes[2].set_title(f'Overlay (Pred: {"Stroke" if pred_class.item() == 1 else "Normal"})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return pred_class.item()

def generate_explainability_report(model_paths, sample_images, output_dir):
    """
    Generate comprehensive explainability report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report = []
    
    for i, img_path in enumerate(sample_images):
        print(f"Processing sample {i+1}/{len(sample_images)}: {img_path}")
        
        # CNN Grad-CAM
        try:
            cnn_pred = visualize_cnn_gradcam(
                model_paths['cnn'], img_path,
                os.path.join(output_dir, f'sample_{i+1}_cnn_gradcam.png')
            )
            report.append(f"Sample {i+1} (CNN): Predicted {'Stroke' if cnn_pred == 1 else 'Normal'}")
        except Exception as e:
            print(f"CNN Grad-CAM failed: {e}")
        
        # ViT Attention
        try:
            vit_pred = visualize_vit_attention(
                model_paths['improved_vit'], img_path,
                os.path.join(output_dir, f'sample_{i+1}_vit_attention.png')
            )
            report.append(f"Sample {i+1} (Improved ViT): Predicted {'Stroke' if vit_pred == 1 else 'Normal'}")
        except Exception as e:
            print(f"ViT attention failed: {e}")
    
    # Save report
    with open(os.path.join(output_dir, 'explainability_report.txt'), 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nExplainability report saved to: {output_dir}")

if __name__ == "__main__":
    from config import NORMAL_DIR, STROKE_DIR
    
    model_paths = {
        'cnn': os.path.join(OUTPUT_DIR, 'models', 'cnn_best.pth'),
        'improved_vit': os.path.join(OUTPUT_DIR, 'models', 'improved_vit_best.pth')
    }
    
    # Get sample images
    normal_samples = [os.path.join(NORMAL_DIR, f) for f in os.listdir(NORMAL_DIR)[:3]]
    stroke_samples = [os.path.join(STROKE_DIR, f) for f in os.listdir(STROKE_DIR)[:3]]
    sample_images = normal_samples + stroke_samples
    
    output_dir = os.path.join(OUTPUT_DIR, 'figures', 'explainability')
    generate_explainability_report(model_paths, sample_images, output_dir)