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


def convert_bbox_to_yolo(xmin, ymin, xmax, ymax):
    x_center = round((xmax + xmin) / 2.0, 6)
    y_center = round((ymax + ymin) / 2.0, 6)
    width    = round(xmax - xmin, 6) 
    height   = round(ymax - ymin, 6)
    return [x_center, y_center, width, height]


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


# Convert OpenImages labels to YOLO format and copy to YOLO directory
def convert_labels_to_yolo_format(yolo_base, openimages_base):
    openimages_subset = ['train', 'validation', 'test']
    yolo_base_subset = ['train', 'val', 'test']
    
    map_subset = dict(zip(openimages_subset, yolo_base_subset))
    
    for item in openimages_subset:
        src_dir = Path(openimages_base) / 'Dataset' / item / 'Woman' / 'Label'
        dist_dir = Path(yolo_base) / map_subset[item] / 'labels'
        dist_dir.mkdir(parents=True, exist_ok=True)
        
        labels_list = list(src_dir.glob('*.txt'))
        
        images_list = [item.name for item in (Path(yolo_base) / map_subset[item] / 'images').glob('*.jpg')]
        
        for label_path in tqdm(labels_list):
            tqdm.write(f"Converting label: {label_path.name}")
            image_name = label_path.stem + '.jpg'
            if image_name not in images_list:
                continue
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            yolo_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = 0
                xmin, ymin, xmax, ymax = map(float, parts[1:])
                yolo_format = convert_bbox_to_yolo(xmin, ymin, xmax, ymax)
                yolo_line = f"{class_id} " + " ".join(map(str, yolo_format))
                yolo_lines.append(yolo_line)
            
            with open(dist_dir / label_path.name, 'w') as f:
                f.write("\n".join(yolo_lines))
#convert_labels_to_yolo_format(yolo_base, openimages_base)
            
                
def print_yolo_directory_structure():  
    print("\n" + "=" * 60)
    print("📂 YOLO DIRECTORY STRUCTURE")
    print("=" * 60)
    yolo_dir = Path("./../woman_detection_yolo")
    for split in ['train', 'val', 'test']:
        split_path = yolo_dir / split
        if split_path.exists():
            print(f"\n{split}/")
            
            # Images folder
            images_folder = split_path / "images"
            if images_folder.exists():
                jpg_count = len(list(images_folder.glob("*.jpg")))
                print(f"  └── images/ ({jpg_count} images)")
            
            # Labels folder
            labels_folder = split_path / "labels"
            if labels_folder.exists():
                txt_count = len(list(labels_folder.glob("*.txt")))
                print(f"  └── labels/ ({txt_count} txt files)")
            
print_yolo_directory_structure()                
            
                    
                
            

