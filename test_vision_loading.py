
import os
import sys

# Add backend to path so we can import modules
sys.path.append(os.path.abspath("backend"))

print("--- Testing VisionModel Import Logic ---")

try:
    from processing_unit.vision_model import VisionReasoner
    print("[PASS] VisionReasoner class imported successfully.")
except ImportError as e:
    print(f"[FAIL] Could not import VisionReasoner: {e}")
    sys.exit(1)

# Check the flags set in vision_model.py
import processing_unit.vision_model as vm
print(f"HAS_QWEN_CLASS: {vm.HAS_QWEN_CLASS}")
print(f"HAS_QWEN_UTILS: {vm.HAS_QWEN_UTILS}")
print(f"HAS_QWEN: {vm.HAS_QWEN}")

if vm.HAS_QWEN:
    print("[PASS] Qwen dependencies detected (either direct class or AutoModel fallback).")
else:
    print("[FAIL] Qwen dependencies NOT detected.")

# Mock SystemManager for initialization
class MockSystem:
    def __init__(self):
        self.status = type('obj', (object,), {'value': 'idle'})
        self.config = {}
        self.logs = []
    def log(self, msg, level="INFO"):
        print(f"[MockLog] {msg}")

print("\n--- Testing Model Loading (Dry Run) ---")
# Set a dummy path to trigger the logic, or use real path if exists
test_model_path = os.environ.get("QWEN_MODEL_PATH", "../models/Qwen2.5-VL-3B-Instruct")
print(f"Target Model Path: {test_model_path}")

if os.path.exists(test_model_path):
    print("Model path exists. Attempting instantiation...")
    try:
        # Pass the MockSystem to avoid triggering real SystemManager (which loads YOLO)
        reasoner = VisionReasoner(system=MockSystem())
        if reasoner.is_model_loaded:
            print(f"[SUCCESS] Model loaded successfully using {'Qwen2_5_VLForConditionalGeneration' if vm.HAS_QWEN_CLASS else 'AutoModelForCausalLM'}.")
        else:
            print("[FAIL] Model failed to load (check logs above).")
    except Exception as e:
        print(f"[CRITICAL] Crash during instantiation: {e}")
else:
    print(f"[SKIP] Model path {test_model_path} does not exist. Skipping load test.")

print("\n--- Testing YOLO Object Detector Loading ---")
try:
    from processing_unit.object_detection import ObjectDetector
    
    # Check for custom model path environment variable
    yolo_path = os.environ.get("YOLO_MODEL_PATH")
    
    if not yolo_path:
        # Logic from system_manager.py
        yolo26_paths = ["yolo26n.pt", "../yolo26n.pt", "backend/yolo26n.pt"]
        for p in yolo26_paths:
            if os.path.exists(p):
                yolo_path = p
                break
    
    if not yolo_path:
        yolo_path = "yolo11n.pt" # Fallback
        
    print(f"Target YOLO Path: {yolo_path}")
    
    detector = ObjectDetector(model_path=yolo_path)
    print(f"[SUCCESS] YOLO model loaded from {yolo_path}")
    
    # Optional: Verify it can run a dummy inference
    # import numpy as np
    # dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    # detector.predict(dummy_img)
    # print("[SUCCESS] YOLO dummy inference passed")

except Exception as e:
    print(f"[FAIL] YOLO model failed to load: {e}")
