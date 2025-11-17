import os
import cv2
import numpy as np
from ultralytics import YOLO

# ----- CONFIG -----
images_dir = "D:/Projects/AIProjects/DermaIQ/Dataset/woundData/data_wound_seg/images/test"
labels_dir = "D:/Projects/AIProjects/DermaIQ/Dataset/woundData/data_wound_seg/labels/test"
output_dir = "D:/Projects/AIProjects/DermaIQ/Dataset/woundData/results"
model_path = "../runs/segment/yolo11n/weights/best.pt"

os.makedirs(output_dir, exist_ok=True)

# Load YOLOv11 model
model = YOLO(model_path)

def load_yolo_mask(image_path, label_path):
    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = list(map(float, line.split()))
                xy = parts[1:]
                points = np.array(xy).reshape(-1, 2)
                points[:, 0] *= W
                points[:, 1] *= H
                points = points.astype(int)
                cv2.fillPoly(mask, [points], 255)
    return img, mask

def save_side_by_side(img, mask, output_path):
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((img, mask3))
    cv2.imwrite(output_path, combined)

def save_overlay(img, mask, output_path, alpha=0.5):
    color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img, 1.0, color_mask, alpha, 0)
    cv2.imwrite(output_path, blended)

def save_prediction_with_score_above_mask(img_path, output_path):
    results = model(img_path)
    img = cv2.imread(img_path)

    for r in results:
        masks = r.masks.data.cpu().numpy()
        final_mask = (masks.max(axis=0) * 255).astype(np.uint8)

        # Convert to color mask
        color_mask = cv2.applyColorMap(final_mask, cv2.COLORMAP_JET)
        color_mask = cv2.resize(color_mask, (img.shape[1], img.shape[0]))

        # Blend mask onto image
        blended = cv2.addWeighted(img, 1.0, color_mask, 0.5, 0)

        # Compute mean confidence (precision proxy)
        confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else [0]
        mean_conf = float(np.mean(confs))

        # Find bounding box of mask
        coords = cv2.findNonZero(final_mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            text_x = x
            text_y = max(y - 10, 0)  # 10 pixels above the top
        else:
            # If mask is empty, put text top-left
            text_x, text_y = 10, 30

        # Put text above the segmented area
        cv2.putText(blended, f"wound: {mean_conf:.2f}",
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Save final image
        cv2.imwrite(output_path, blended)

# ----- PROCESS ALL IMAGES -----
for file_name in os.listdir(images_dir):
    if not file_name.endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(images_dir, file_name)
    label_path = os.path.join(labels_dir, os.path.splitext(file_name)[0] + ".txt")

    img, gt_mask = load_yolo_mask(image_path, label_path)

    # Save original + GT mask side by side
    # save_side_by_side(img, gt_mask, os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_gt_side.png"))

    # Save GT mask overlay
    save_overlay(img, gt_mask, os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_gt_overlay.png"))

    # Save predicted mask (optional)
    save_prediction_with_score_above_mask(image_path, os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_pred_mask.png"))

print("All images processed and saved in:", output_dir)