from pprint import pprint
from ultralytics import YOLO
import cv2
import numpy as np


MODEL_PATH = "ckpts/cotrain_all_class_0731_1.pt"
# IMG_PATH = "data/real_datasets/images/test/000003_pic_10.png"
IMG_PATH = "test_data/rgb_image.png"
    
# Load custom segmentation model
model = YOLO(MODEL_PATH)
print(f"Model type: {model.task}")
print(f"Model file: {model.ckpt_path}\n")

# Perform prediction
results = model(IMG_PATH)

pprint(results)  # Print results for debugging

for result in results:
    # Get original image dimensions
    orig_img = result.orig_img.copy()
    orig_height, orig_width = orig_img.shape[:2]
    print(f"Original image size: {orig_width}x{orig_height}")

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()  # Convert to numpy array
        class_ids = result.boxes.cls.cpu().numpy()
        print(f"Detected {len(masks)} segmentation masks")
        print(f"Mask dimensions before resize: {masks[0].shape}")

        # Create output image
        output_img = orig_img.copy()

        for i, mask in enumerate(masks):
            class_id = int(class_ids[i])
            class_name = result.names[class_id]
            print(f"Mask {i + 1}: Class {class_id} ({class_name})")

            # Resize mask to original image dimensions
            mask_resized = cv2.resize(
                mask.astype(np.float32),
                (orig_width, orig_height),
                interpolation=cv2.INTER_LINEAR,
            )

            # Create color overlay
            color = [int(x) for x in np.random.randint(0, 255, 3)]
            colored_overlay = np.zeros_like(output_img)
            colored_overlay[mask_resized > 0.5] = color  # Apply threshold

            # Blend with original image
            output_img = cv2.addWeighted(output_img, 1, colored_overlay, 0.5, 0)

        # Save result
        cv2.imwrite("output_with_masks.png", output_img)
        print("Saved visualization to 'output_with_masks.png'")
    else:
        print(
            "No masks found. This might be a detection model or no objects were detected."
        )

    # Check for bounding boxes (available in both detection and segmentation)
    if result.boxes is not None:
        print("Bounding boxes detected:")
        boxes = result.boxes.xyxy  # bounding boxes in xyxy format
        confidences = result.boxes.conf  # confidence scores
        classes = result.boxes.cls  # class indices
        print(f"Number of detections: {len(boxes)}")
        print(f"Classes detected: {classes}")
        print(f"Confidences: {confidences}")
    else:
        print("No objects detected.")
