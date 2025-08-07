from ultralytics import YOLO
import numpy as np
import cv2

class YoloSeg():
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.to("cuda") 
        _ = self.model.predict(np.zeros((640, 640, 3)))


    def segment_image(self, img_path):
        results = self.model(img_path)
        return results
    
    def get_best_mask(self, results, class_name: str):
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                best_mask = None
                best_score = 0

                for i, mask in enumerate(masks):
                    if result.names[class_ids[i]] == class_name:
                        score = result.boxes.conf[i]
                        if score > best_score:
                            best_score = score
                            best_mask = mask

                if best_mask is not None:
                    return best_mask

        return None

if __name__ == "__main__":
    
    img_path = "test_data/rgb_image.png"
    model_path = "ckpts/cotrain_all_class_0731_1.pt"

    seg = YoloSeg(model_path)
    results = seg.segment_image(img_path)
    
    for result in results:
        if result.masks is not None:
            print(f"Detected {len(result.masks)} masks in the image.")
        else:
            print("No masks detected.")
            
    best_mask = seg.get_best_mask(results, "bin")
    if best_mask is not None:
        print("Best mask found for class 'bin'.")
        cv2.imwrite("best_mask_bin.png", best_mask * 255)  # Save the mask as an image
    else:
        print("No mask found for class 'bin'.")