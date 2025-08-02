from pprint import pprint
import torch
from ultralytics import YOLO
import argparse

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # model = YOLO('yolo11n-seg.pt')  # Load a pre-trained YOLO model
    model = YOLO('ckpts/cotrain_all_class_0730.pt')  # Load a pre-trained YOLO model

    if args.dataset == 'sim':
        data_path = 'data/sim_datasets/yolo_seg.yaml'
        pprint(object=f"Using simulated dataset: {data_path}")
    elif args.dataset == 'real':
        data_path = 'data/real_datasets/yolo_seg.yaml'
        pprint(object=f"Using real dataset: {data_path}")
    elif args.dataset == 'co':
        data_path = 'data/co_datasets/yolo_seg.yaml'
        pprint(object=f"Using cotrain dataset: {data_path}")
    else:
        raise ValueError("Invalid dataset type. Choose from 'sim', 'real', or 'co'.")
    epochs = 200
    imgsz = 1280
    batch = -1

    model.train(data=data_path, epochs=epochs, imgsz=imgsz, batch=batch, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model for segmentation")
    parser.add_argument('--dataset', type=str, default='real', help='dataset type (real or sim)')
    args = parser.parse_args()
    
    main(args)