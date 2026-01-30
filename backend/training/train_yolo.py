from ultralytics import YOLO
import os
import sys
import glob
import torch
import torch.nn as nn
from functools import partial

# Workaround for PyTorch 2.6+ security change
# Whitelisting is failing recursively or context is not propagating to ultralytics internal calls.
# We will monkeypatch torch.load to force weights_only=False globally for this script.
# Use this only because we trust the model source (official Ultralytics/local).

_original_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    # Force weights_only=False to bypass the security check completely
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = safe_torch_load

def run_training_pipeline(dataset_location: str, epochs: int = 50, imgsz: int = 640):
    """
    Full pipeline: Train -> Validate -> Test for YOLOv11
    """
    print(f"Starting Training Pipeline with dataset at: {dataset_location}")
    
    # 1. Initialize Model
    # Using YOLOv11 Nano model for speed, change to 'yolo11s.pt' or 'yolo11m.pt' for better accuracy
    model = YOLO("./backend/training/yolo11s.pt")
    
    # The dataset location from Roboflow usually contains a data.yaml file
    data_yaml = os.path.join(dataset_location, "data.yaml")
    
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found in {dataset_location}")

    # 2. Train
    print("\n=== STEP 1: TRAINING ===")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project="mcc_construction",
        name="yolo11_columns",
        exist_ok=True,
        plots=True
    )
    print(f"Training finished. Best model saved at: {results.save_dir}")

    # 3. Validate
    print("\n=== STEP 2: VALIDATION ===")
    # Validate the best model on the validation set
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")

    # 4. Test (Inference on Test Set)
    print("\n=== STEP 3: TESTING ===")
    # Roboflow datasets usually have train/valid/test folders
    test_images_path = os.path.join(dataset_location, "test", "images")
    
    if os.path.exists(test_images_path):
        print(f"Running inference on test images at: {test_images_path}")
        test_results = model.predict(
            source=test_images_path,
            conf=0.25,
            save=True,
            project="mcc_construction",
            name="test_inference",
            exist_ok=True
        )
        print(f"Test inference saved to mcc_construction/test_inference")
    else:
        print("Test set not found, skipping testing phase.")

    # 5. Export
    print("\n=== STEP 4: EXPORT ===")
    # Fix for ONNX opset version warning (set to 12 or 17 which are stable)
    export_path = model.export(format="onnx", opset=12)
    print(f"Model exported to: {export_path}")

if __name__ == "__main__":
    # If run directly, assume dataset is in the current directory or passed as arg
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # Default behavior: look for the most recent roboflow download
        # This is a heuristic; usually you'd pass the path explicitly
        dirs = glob.glob("./columns-fncne-*")
        if dirs:
            dataset_path = dirs[0]
        else:
            print("Please provide path to dataset: python train_yolo.py <path_to_dataset>")
            sys.exit(1)
            
    run_training_pipeline(dataset_path)
