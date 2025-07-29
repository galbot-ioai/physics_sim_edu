import argparse
import os
from ultralytics import YOLO
import cv2
import numpy as np

def parse_args():
    """Parse command line arguments."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description="YOLO segmentation prediction with visualization")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=os.path.join(script_dir, "best.pt"),
        help="Path to the trained YOLO segmentation model"
    )
    parser.add_argument(
        "--img_path", 
        type=str, 
        default=os.path.join(script_dir, "test_image.png"),
        help="Path to the input image for prediction"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=os.path.join(script_dir, "output_with_masks.png"),
        help="Path to save the output image"
    )
    parser.add_argument(
        "--show_classes", 
        action="store_true",
        help="Display all available classes in the model"
    )
    return parser.parse_args()

def display_model_classes(model):
    """Display all available classes in the model."""
    if hasattr(model, 'names') and model.names:
        print(f"\n📋 Model contains {len(model.names)} classes:")
        print("=" * 50)
        for class_id, class_name in model.names.items():
            print(f"Class {class_id:2d}: {class_name}")
        print("=" * 50)
    else:
        print("⚠️  No class information available in the model")

def draw_predictions(image, result):
    """Draw bounding boxes, masks, and labels on the image."""
    output_img = image.copy()
    
    if result.boxes is None:
        print("No objects detected.")
        return output_img
    
    # Get prediction data
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = result.boxes.cls.cpu().numpy()  # Class indices
    
    print(f"Detected {len(boxes)} objects")
    
    # Draw masks if available
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        orig_height, orig_width = image.shape[:2]
        
        for i, mask in enumerate(masks):
            # Resize mask to original image dimensions
            mask_resized = cv2.resize(
                mask.astype(np.float32),
                (orig_width, orig_height),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Create colored mask overlay
            color = np.random.randint(0, 255, 3).tolist()
            colored_overlay = np.zeros_like(output_img)
            colored_overlay[mask_resized > 0.5] = color
            
            # Blend mask with image
            output_img = cv2.addWeighted(output_img, 1, colored_overlay, 0.3, 0)
    
    # Draw bounding boxes and labels
    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
        x1, y1, x2, y2 = box.astype(int)
        class_name = result.names[int(class_id)]
        
        # Generate consistent color for each class
        np.random.seed(int(class_id))
        color = np.random.randint(0, 255, 3).tolist()
        
        # Draw thick bounding box
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, thickness=3)
        
        # Prepare label text
        label = f"{class_name}: {conf:.2f}"
        
        # Calculate label background size
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw label background
        cv2.rectangle(
            output_img, 
            (x1, y1 - label_height - baseline - 5), 
            (x1 + label_width, y1), 
            color, 
            cv2.FILLED
        )
        
        # Draw label text
        cv2.putText(
            output_img, 
            label, 
            (x1, y1 - baseline - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )
        
        print(f"Object {i+1}: {class_name} (confidence: {conf:.2f})")
    
    return output_img

def main():
    """Main function to run YOLO segmentation prediction with visualization."""
    args = parse_args()
    
    # Load model and run prediction
    model = YOLO(args.model_path)
    print(f"Loaded {model.task} model from: {model.ckpt_path}")
    
    # Always display available classes
    display_model_classes(model)
    
    results = model(args.img_path)
    
    for result in results:
        # Get original image
        orig_img = result.orig_img.copy()
        print(f"\n🖼️  Image size: {orig_img.shape[1]}x{orig_img.shape[0]}")
        
        # Display detected classes summary
        if result.boxes is not None:
            detected_classes = {}
            for class_id in result.boxes.cls.cpu().numpy():
                class_name = result.names[int(class_id)]
                detected_classes[class_name] = detected_classes.get(class_name, 0) + 1
            
            print(f"\n🎯 Detected classes in this image:")
            for class_name, count in detected_classes.items():
                print(f"   • {class_name}: {count} instance(s)")
        
        # Draw predictions on image
        output_img = draw_predictions(orig_img, result)
        
        # Save result
        cv2.imwrite(args.output_path, output_img)
        print(f"\n💾 Saved visualization to: {args.output_path}")

if __name__ == "__main__":
    main()
