# Convert OpenImages Woman Class to YOLO Format
# Run this in Jupyter Notebook (Local Machine)

import os
import shutil
from pathlib import Path
from tqdm import tqdm

print("=" * 70)
print("OpenImages to YOLO Format Converter")
print("=" * 70)

# Define paths
openimages_base = "./../OID"
yolo_base = "./../woman_detection_yolo"

def create_yolo_structure(base_path):
    dirs = ['train', 'val', 'test']
    
    # Create (images, labels) for each (train, val, test) directory 
    for dir_name in dirs:
        images_path = Path(base_path) / dir_name / 'images'
        labels_path = Path(base_path) / dir_name / 'labels'
        
        images_path.mkdir(parents=True, exist_ok=True)
        labels_path.mkdir(parents=True, exist_ok=True)

# create_yolo_structure(yolo_base)

def copy_images_to_yolo_directory(yolo_base, openimages_base):
    openimages_subset = ['train', 'validation', 'test']
    yolo_base_subset = ['train', 'val', 'test']
    
    map_subset = dict(zip(openimages_subset, yolo_base_subset))
    for item in tqdm(openimages_subset):
        tqdm.write(f"Copying images from {item} to YOLO {map_subset[item]}")
        src_dir = Path(openimages_base) / 'Dataset' / item / 'Woman'
        images_list = list(src_dir.glob('*.jpg'))
        shutil_dest = Path(yolo_base) / map_subset[item] / 'images'
        shutil_dest.mkdir(parents=True, exist_ok=True)
        
        for img_path in images_list:
            shutil.copy(img_path, shutil_dest / img_path.name)
        
            
# copy_images_to_yolo_directory(yolo_base, openimages_base)