import argparse
import torch
from ultralytics import YOLO

def parse_args():
    """Parse command line arguments for YOLO training."""
    parser = argparse.ArgumentParser(description="YOLO segmentation model training")
    parser.add_argument(
        "--model", 
        type=str, 
        default="yolo11n-seg.pt",
        help="Path to the pre-trained YOLO model or model name"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/datasets/yolo_seg.yaml",
        help="Path to the dataset configuration YAML file"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=200,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=640,
        help="Image size for training"
    )
    parser.add_argument(
        "--batch", 
        type=int, 
        default=-1,
        help="Batch size (-1 for auto batch size)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use for training (auto, cpu, cuda, or specific GPU like 0,1,2,3)"
    )
    return parser.parse_args()

def main():
    """Main function to run YOLO segmentation training."""
    args = parse_args()
    
    # Set device based on argument or auto-detect
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device
    
    print(f"Using device: {device}")

    # Load a pre-trained YOLO model
    model = YOLO(args.model)

    # Start training with specified parameters
    model.train(
        data=args.data, 
        epochs=args.epochs, 
        imgsz=args.imgsz, 
        batch=args.batch, 
        device=device
    )

if __name__ == "__main__":
    main()
