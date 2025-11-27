# YOLOv8n Woman Detection Training Script
# Run this locally on your machine

# ============================================================
# STEP 1: Install Ultralytics YOLOv8 (run once in terminal)
# pip install ultralytics
# ============================================================

import os
from ultralytics import YOLO
import torch
from pathlib import Path

print("=" * 70)
print("YOLOv8n Woman Detection - Training Setup")
print("=" * 70)

# ============================================================
# STEP 2: Check GPU Availability
# ============================================================
print(f"\n🔧 GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("   ⚠️  WARNING: No GPU detected! Training will be VERY slow on CPU.")
    print("   Consider using Google Colab Pro with GPU instead.")

# ============================================================
# STEP 3: Setup Paths
# ============================================================
dataset_path = "/home/khaled-ekramy/GraduationProject/woman_detection_yolo"
dataset_yaml = f"{dataset_path}/dataset.yaml"

# Verify dataset exists
if not os.path.exists(dataset_yaml):
    print(f"\n❌ ERROR: dataset.yaml not found at {dataset_yaml}")
    print(f"Please ensure your dataset is at: {dataset_path}")
    exit(1)
else:
    print(f"\n✅ Dataset found: {dataset_yaml}")

# Create output directory for training results
output_dir = Path(dataset_path) / "runs" / "detect" / "woman_yolov8n"
output_dir.mkdir(parents=True, exist_ok=True)
print(f"✅ Output directory: {output_dir}")

# ============================================================
# STEP 4: Load Pretrained YOLOv8n Model
# ============================================================
print("\n" + "=" * 70)
print("Loading Pretrained YOLOv8n Model")
print("=" * 70)

# Load YOLOv8n with COCO pretrained weights
model = YOLO('yolov8n.pt')  # Automatically downloads if not present
print("✅ YOLOv8n pretrained model loaded!")

# ============================================================
# STEP 5: Training Configuration
# ============================================================
print("\n" + "=" * 70)
print("Training Configuration")
print("=" * 70)

train_config = {
    'data': dataset_yaml,           # Path to dataset.yaml
    'epochs': 100,                   # Number of training epochs
    'imgsz': 640,                    # Image size (416x416 for speed)
    'batch': 5,                     # Batch size (adjust based on GPU memory)
    'device': 0,                     # GPU device (0 for first GPU, 'cpu' for CPU)
    'workers': 2,                    # Number of dataloader workers
    'project': str(output_dir.parent.parent),  # Project directory
    'name': 'woman_yolov8n',        # Experiment name
    'exist_ok': True,                # Overwrite existing
    'pretrained': True,              # Use pretrained weights
    'optimizer': 'AdamW',            # Optimizer
    'lr0': 0.001,                    # Initial learning rate
    'momentum': 0.937,               # Momentum
    'weight_decay': 0.0005,          # Weight decay
    'warmup_epochs': 3,              # Warmup epochs
    'patience': 50,                  # Early stopping patience
    'save': True,                    # Save checkpoints
    'save_period': 5,               # Save checkpoint every N epochs
    'cache': False,                  # Cache images (set True if you have enough RAM)
    'val': True,                     # Validate during training
    'plots': True,                   # Generate training plots
    'verbose': True                  # Verbose output
}

# Print configuration
print("\nTraining Parameters:")
for key, value in train_config.items():
    print(f"  {key}: {value}")

# ============================================================
# STEP 6: Start Training
# ============================================================
print("\n" + "=" * 70)
print("🚀 STARTING TRAINING")
print("=" * 70)

print("\n💡 Training will save checkpoints every 10 epochs.")
print("💡 Press Ctrl+C to stop training (last checkpoint will be saved).\n")

try:
    # Train the model
    results = model.train(**train_config)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    
except KeyboardInterrupt:
    print("\n" + "=" * 70)
    print("⚠️  Training interrupted by user")
    print("=" * 70)
    
except Exception as e:
    print("\n" + "=" * 70)
    print(f"❌ ERROR during training: {str(e)}")
    print("=" * 70)
    import traceback
    traceback.print_exc()

# ============================================================
# STEP 7: Find Best Model
# ============================================================
print("\n" + "=" * 70)
print("📊 Training Results")
print("=" * 70)

best_model_path = output_dir / "weights" / "best.pt"
last_model_path = output_dir / "weights" / "last.pt"

if best_model_path.exists():
    print(f"\n✅ Best model saved at: {best_model_path}")
    print(f"✅ Last model saved at: {last_model_path}")
    print(f"\n📈 Training plots saved at: {output_dir}")
    print(f"   - results.png: Training metrics")
    print(f"   - confusion_matrix.png: Confusion matrix")
    print(f"   - PR_curve.png: Precision-Recall curve")
else:
    print("\n⚠️  Model weights not found. Check training logs above.")

# ============================================================
# STEP 8: Validate Best Model
# ============================================================
print("\n" + "=" * 70)
print("🔍 Validating Best Model")
print("=" * 70)

if best_model_path.exists():
    best_model = YOLO(str(best_model_path))
    
    # Run validation
    metrics = best_model.val(data=dataset_yaml, imgsz=416)
    
    print("\n📊 Validation Metrics:")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")

# ============================================================
# STEP 9: Export for Deployment
# ============================================================
print("\n" + "=" * 70)
print("📦 Next Steps for Deployment")
print("=" * 70)
print("\nTo optimize for Nvidia T1000 deployment:")
print(f"1. Export to TensorRT: model.export(format='engine', device=0)")
print(f"2. Export to ONNX: model.export(format='onnx')")
print(f"3. Test inference speed on your T1000")
print(f"\nBest model location: {best_model_path}")

print("\n" + "=" * 70)
print("🎉 SCRIPT COMPLETE!")
print("=" * 70)
