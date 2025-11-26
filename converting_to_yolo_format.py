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
openimages_base = "./OID/Dataset"
yolo_base = "./openimages_woman/woman_detection_yolo"

# Create YOLO dataset structure
print("\nCreating YOLO dataset structure...")
for split in ['train', 'val', 'test']:
    os.makedirs(f"{yolo_base}/{split}/images", exist_ok=True)
    os.makedirs(f"{yolo_base}/{split}/labels", exist_ok=True)
print("Directories created!")

# Step 2: Conversion function
def convert_openimages_to_yolo(xmin, ymin, xmax, ymax):
    """
    Convert OpenImages format to YOLO format
    OpenImages: xmin, ymin, xmax, ymax (normalized 0-1)
    YOLO: x_center, y_center, width, height (normalized 0-1)
    """
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    return x_center, y_center, width, height

# Step 3: Process each split
def process_split(split_name, yolo_split_name):
    """
    Process train/validation/test split
    split_name: 'train', 'validation', 'test' (OpenImages naming)
    yolo_split_name: 'train', 'val', 'test' (YOLO naming)
    """
    print(f"\n{'=' * 70}")
    print(f"Processing {split_name.upper()} → {yolo_split_name.upper()}")
    print(f"{'=' * 70}")
    
    # Source paths
    images_dir = Path(openimages_base) / split_name / "Woman"
    labels_dir = images_dir / "Label"
    
    # Destination paths
    yolo_images_dir = Path(yolo_base) / yolo_split_name / "images"
    yolo_labels_dir = Path(yolo_base) / yolo_split_name / "labels"
    
    # Check if source exists
    if not images_dir.exists():
        print(f"❌ {split_name} directory not found. Skipping...")
        return 0, 0
    
    # Get all images
    image_files = list(images_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} images")
    
    converted_count = 0
    skipped_count = 0
    
    # Process each image
    for img_path in tqdm(image_files, desc=f"Converting {split_name}"):
        try:
            # Read OpenImages label file
            label_file = labels_dir / f"{img_path.stem}.txt"
            
            if not label_file.exists():
                skipped_count += 1
                continue
            
            # Read bounding boxes
            yolo_annotations = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # OpenImages format: class_name xmin ymin xmax ymax
                        class_name = parts[0]
                        xmin, ymin, xmax, ymax = map(float, parts[1:5])
                        
                        # Convert to YOLO format
                        x_center, y_center, width, height = convert_openimages_to_yolo(
                            xmin, ymin, xmax, ymax
                        )
                        
                        # YOLO format: class_id x_center y_center width height
                        # Woman class = 0 (single class)
                        yolo_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        yolo_annotations.append(yolo_line)
            
            # Only copy if we have valid annotations
            if yolo_annotations:
                # Copy image
                dest_img = yolo_images_dir / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # Write YOLO label
                dest_label = yolo_labels_dir / f"{img_path.stem}.txt"
                with open(dest_label, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                converted_count += 1
            else:
                skipped_count += 1
        
        except Exception as e:
            print(f"\n❌ Error processing {img_path.name}: {str(e)}")
            skipped_count += 1
    
    print(f"✅ Converted: {converted_count} images")
    print(f"⚠️ Skipped: {skipped_count} images")
    return converted_count, skipped_count

# Step 4: Process all splits
print("\n" + "=" * 70)
print("Starting Conversion")
print("=" * 70)

total_converted = 0
total_skipped = 0

# Map OpenImages split names to YOLO split names
split_mapping = {
    'train': 'train',
    'validation': 'val',
    'test': 'test'
}

for oi_split, yolo_split in split_mapping.items():
    converted, skipped = process_split(oi_split, yolo_split)
    total_converted += converted
    total_skipped += skipped

# Step 5: Verify dataset
print("\n" + "=" * 70)
print("📊 DATASET SUMMARY")
print("=" * 70)

for split in ['train', 'val', 'test']:
    images_path = Path(yolo_base) / split / "images"
    labels_path = Path(yolo_base) / split / "labels"
    
    if images_path.exists():
        img_count = len(list(images_path.glob("*.jpg")))
        label_count = len(list(labels_path.glob("*.txt")))
        print(f"{split.upper():>5}: {img_count:>6} images, {label_count:>6} labels")

print(f"\n✅ Total Converted: {total_converted}")
print(f"⚠️ Total Skipped: {total_skipped}")

# Step 6: Sample validation
print("\n" + "=" * 70)
print("🔍 SAMPLE VALIDATION")
print("=" * 70)

sample_label = list(Path(yolo_base).rglob("labels/*.txt"))[0]
print(f"\nSample label file: {sample_label.name}")
print("Content (first 3 lines):")
with open(sample_label, 'r') as f:
    for i, line in enumerate(f):
        if i < 3:
            print(f"  {line.strip()}")
        else:
            break

print("\n" + "=" * 70)
print("🎉 CONVERSION COMPLETE!")
print("=" * 70)
print(f"📁 YOLO Dataset location: {yolo_base}")
print("\n💡 Next step: Create dataset.yaml manually or upload to Colab!")