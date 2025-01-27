import os
import cv2
import csv
import numpy as np

"""
This script is used to generate submission.csv for hw2.

Note: Predicted masks must be grayscale(1-channel).
"""

pred_mask_dir = '/home/ouvic/ML/ML_HW2/final_predict/vote_012_1122_x5_3' # Path relative to ur peredicted masks
save_csv_path = '/home/ouvic/ML/ML_HW2/csv/submission_1122_3.csv' # Path relative to save submission.csv

def rle_encoding(mask: np.ndarray, class_id: int) -> str:
    binary_mask = (mask == class_id).astype(np.uint8)
    pixels = binary_mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs) if runs.size > 1 else 'none'
def generate_submission_csv():
    all_masks = []
    for filename in os.listdir(pred_mask_dir):
        if filename.endswith('.png'):
            all_masks.append(os.path.join(pred_mask_dir, filename))
    
    all_masks = sorted([os.path.join(pred_mask_dir, filename) for filename in os.listdir(pred_mask_dir) if filename.endswith('.png')],
                   key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    with open(save_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['img_id'] + [f'class_{i}' for i in range(13)]
        writer.writerow(header)
        for mask_path in all_masks:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_name = mask_path.split('/')[-1]
            rle_encodings = {}
            for class_id in range(13):
                rle_encoded = rle_encoding(mask, class_id)
                rle_encodings[class_id] = rle_encoded
            rows = [mask_name] + [rle_encodings[class_id] for class_id in range(13)]
            writer.writerow(rows)

generate_submission_csv()