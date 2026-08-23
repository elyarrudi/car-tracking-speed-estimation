from pathlib import Path
from ultralytics import YOLO
from pathlib import Path
PROJECT_DIR=Path(__file__).resolve().parent
YOLO_PT = PROJECT_DIR / "yolov8m.pt"
YOLO_ONNX = PROJECT_DIR / "yolov8m.onnx"

REID_PT = PROJECT_DIR / "osnet_x0_25_msmt17.pt"
REID_ONNX = PROJECT_DIR / "osnet_x0_25_msmt17.onnx"

if not YOLO_PT.exists():
    raise FileNotFoundError(f"YOLO model not found: {YOLO_PT}")

if not REID_PT.exists():
    raise FileNotFoundError(f"ReID model not found: {REID_PT}")

if not YOLO_ONNX.exists():
    print("Exporting YOLO to ONNX...")
    YOLO(str(YOLO_PT)).export(
        format="onnx",
        imgsz=640,
        opset=17,
        simplify=False,
        dynamic=False,
    )
    print(f"YOLO ONNX saved: {YOLO_ONNX}")
else:
    print(f"YOLO ONNX already exists: {YOLO_ONNX}")

if not REID_ONNX.exists():
    print("Exporting ReID to ONNX...")
    from boxmot.native._common import export_reid_to_onnx
    export_reid_to_onnx(REID_PT)
    print(f"ReID ONNX saved: {REID_ONNX}")
else:
    print(f"ReID ONNX already exists: {REID_ONNX}")

print("\nExport completed.")
print(f"YOLO ONNX: {YOLO_ONNX}")
print(f"ReID ONNX: {REID_ONNX}")