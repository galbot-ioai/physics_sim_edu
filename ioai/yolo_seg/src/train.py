import os
import argparse
from pprint import pprint
import torch
from ultralytics import YOLO


def main(args):
    # Get parent directory (ioai/yolo_seg/) for default paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pre-trained YOLO model
    model_path = args.model_path
    model = YOLO(model_path)

    # Set data path based on dataset type
    if args.dataset == 'sim':
        data_path = os.path.join(parent_dir, 'data/sim_datasets/yolo_seg.yaml')
        print(f"Using simulated dataset: {data_path}")
    elif args.dataset == 'real':
        data_path = os.path.join(parent_dir, 'data/real_datasets/yolo_seg.yaml')
        print(f"Using real dataset: {data_path}")
    elif args.dataset == 'co':
        data_path = os.path.join(parent_dir, 'data/co_datasets/yolo_seg.yaml')
        print(f"Using cotrain dataset: {data_path}")
    else:
        raise ValueError("Invalid dataset type. Choose from 'sim', 'real', or 'co'.")

    # Training parameters
    epochs = args.epochs
    imgsz = args.imgsz
    batch = args.batch

    # Train the model
    model.train(data=data_path, epochs=epochs, imgsz=imgsz, batch=batch, device=device)


if __name__ == "__main__":
    # Get parent directory for default paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    parser = argparse.ArgumentParser(description="Train YOLO model for segmentation")
    parser.add_argument('--dataset', type=str, default='real', 
                       help='Dataset type (sim, real, or co)')
    parser.add_argument('--model_path', type=str, 
                       default=os.path.join(parent_dir, 'ckpts/cotrain_all_class_0731_1.pt'), 
                       help='Path to the pre-trained model')
    parser.add_argument('--epochs', type=int, default=200, 
                       help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=1280, 
                       help='Input image size')
    parser.add_argument('--batch', type=int, default=-1, 
                       help='Batch size (-1 for auto)')
    args = parser.parse_args()
    
    main(args)