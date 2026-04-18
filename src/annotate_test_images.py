import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tqdm import tqdm

from config import DEVICE, OUTPUT_DIR, IMPROVED_VIT_CONFIG
from src.models import ImprovedStrokeViT

def load_trained_model(model_path):
    """Load the trained Improved ViT model"""
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    config = checkpoint.get('config', IMPROVED_VIT_CONFIG)
    
    model = ImprovedStrokeViT(
        num_labels=2,
        unfreeze_layers=config.get('unfreeze_layers', 10),
        dropout=config.get('dropout', 0.15)
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    optimal_threshold = checkpoint.get('optimal_threshold', 0.5)
    print(f"Model loaded. Optimal threshold: {optimal_threshold:.3f}")
    
    return model, optimal_threshold

def get_test_transform():
    """Same transform as used during testing"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def annotate_image(image_path, prediction, probability, output_path):
    """
    Write prediction on image with dark text
    Green border for correct, Red border for incorrect (if we know truth)
    """
    # Open original image
    img = Image.open(image_path).convert('RGB')
    
    # Create draw object
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default
    try:
        # Try system fonts
        font_large = ImageFont.truetype("arial.ttf", 36)
        font_small = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    # Text to write
    label_text = "STROKE" if prediction == 1 else "NORMAL"
    prob_text = f"{probability:.1%}"
    
    # Colors - DARK for visibility
    text_color = (0, 0, 0)  # Black
    stroke_color = (255, 255, 255)  # White outline for contrast
    
    # Position: Top-left corner with padding
    x, y = 20, 20
    
    # Draw text with outline for better visibility
    # Outline
    for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2), (0, -2), (0, 2), (-2, 0), (2, 0)]:
        draw.text((x+dx, y+dy), label_text, font=font_large, fill=stroke_color)
    # Main text
    draw.text((x, y), label_text, font=font_large, fill=text_color)
    
    # Draw probability below
    y2 = y + 45
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        draw.text((x+dx, y2+dy), prob_text, font=font_small, fill=stroke_color)
    draw.text((x, y2), prob_text, font=font_small, fill=text_color)
    
    # Add colored corner indicator
    # Green = Normal, Red = Stroke
    indicator_color = (0, 200, 0) if prediction == 0 else (200, 0, 0)
    draw.rectangle([(0, 0), (15, 15)], fill=indicator_color)
    
    # Save annotated image
    img.save(output_path, quality=95)
    
    return output_path

def annotate_all_test_images(model_path, test_files, test_labels=None, output_dir=None):
    """
    Run model on all test images and write predictions on them
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, 'annotated_test_images')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model, threshold = load_trained_model(model_path)
    transform = get_test_transform()
    
    print(f"\nAnnotating {len(test_files)} test images...")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    results = []
    correct_count = 0
    
    for idx, img_path in enumerate(tqdm(test_files, desc="Processing")):
        try:
            # Load and transform image
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            # Predict
            with torch.no_grad():
                outputs, _ = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                stroke_prob = probs[0, 1].item()
                
                # Apply threshold
                prediction = 1 if stroke_prob >= threshold else 0
            
            # Check correctness if labels provided
            if test_labels is not None:
                actual = test_labels[idx]
                correct = (prediction == actual)
                if correct:
                    correct_count += 1
                status = "✓" if correct else "✗"
            else:
                actual = None
                correct = None
                status = "?"
            
            # Create output filename
            base_name = os.path.basename(img_path)
            name, ext = os.path.splitext(base_name)
            out_name = f"{name}_PRED_{'STROKE' if prediction == 1 else 'NORMAL'}{ext}"
            out_path = os.path.join(output_dir, out_name)
            
            # Annotate image
            annotate_image(img_path, prediction, stroke_prob, out_path)
            
            # Store result
            results.append({
                'original_path': img_path,
                'annotated_path': out_path,
                'actual_label': 'STROKE' if actual == 1 else 'NORMAL' if actual == 0 else 'UNKNOWN',
                'predicted_label': 'STROKE' if prediction == 1 else 'NORMAL',
                'stroke_probability': stroke_prob,
                'correct': correct,
                'filename': out_name
            })
            
            # Print some examples
            if idx < 5 or idx % 50 == 0:
                true_label = f" (True: {'STROKE' if actual == 1 else 'NORMAL'})" if actual is not None else ""
                print(f"{status} {base_name} -> {'STROKE' if prediction == 1 else 'NORMAL'} ({stroke_prob:.1%}){true_label}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 70)
    print("ANNOTATION COMPLETE")
    print("=" * 70)
    print(f"Total images processed: {len(results)}")
    if test_labels is not None:
        accuracy = correct_count / len(results) * 100
        print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{len(results)})")
    print(f"Annotated images saved to: {output_dir}")
    
    # Save CSV with all predictions
    import pandas as pd
    csv_path = os.path.join(output_dir, 'annotation_results.csv')
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Results CSV: {csv_path}")
    
    # Create summary by actual class
    if test_labels is not None:
        print("\nBreakdown by actual class:")
        normal_results = [r for r in results if r['actual_label'] == 'NORMAL']
        stroke_results = [r for r in results if r['actual_label'] == 'STROKE']
        
        if normal_results:
            normal_correct = sum(1 for r in normal_results if r['correct'])
            print(f"  NORMAL:  {normal_correct}/{len(normal_results)} correct ({100*normal_correct/len(normal_results):.1f}%)")
        
        if stroke_results:
            stroke_correct = sum(1 for r in stroke_results if r['correct'])
            print(f"  STROKE:  {stroke_correct}/{len(stroke_results)} correct ({100*stroke_correct/len(stroke_results):.1f}%)")
    
    return results

def main():
    """Main function to annotate all test images"""
    
    # Path to your trained model
    model_path = os.path.join(OUTPUT_DIR, 'models', 'improved_vit_best.pth')
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please check the path or train the model first.")
        return
    
    # Get test files from the saved split info
    split_file = os.path.join(OUTPUT_DIR, 'results', 'dataset_splits.json')
    
    if os.path.exists(split_file):
        import json
        with open(split_file, 'r') as f:
            splits = json.load(f)
        test_files = splits['test']
        
        # Extract labels from paths
        test_labels = [1 if 'stroke' in f.lower() else 0 for f in test_files]
    else:
        # Fallback: scan test directory manually
        print("Warning: split file not found, scanning directories...")
        from config import NORMAL_DIR, STROKE_DIR
        
        # You'll need to specify which files are test files
        # For now, let's use the last 15% of each class as test set
        normal_files = sorted([os.path.join(NORMAL_DIR, f) for f in os.listdir(NORMAL_DIR) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        stroke_files = sorted([os.path.join(STROKE_DIR, f) for f in os.listdir(STROKE_DIR) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Take last 15% as test (same as stratified split)
        n_normal_test = int(len(normal_files) * 0.15)
        n_stroke_test = int(len(stroke_files) * 0.15)
        
        test_files = normal_files[-n_normal_test:] + stroke_files[-n_stroke_test:]
        test_labels = [0] * n_normal_test + [1] * n_stroke_test
    
    # Run annotation
    results = annotate_all_test_images(model_path, test_files, test_labels)
    
    print("\n" + "=" * 70)
    print("DONE! Check the annotated images in:")
    print(os.path.join(OUTPUT_DIR, 'annotated_test_images'))
    print("=" * 70)

if __name__ == "__main__":
    main()