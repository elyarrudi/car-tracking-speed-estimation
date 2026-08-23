import os
os.environ["BOXMOT_VERBOSE"] = "0"

import logging
logging.getLogger("boxmot").setLevel(logging.ERROR)

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from boxmot.trackers.botsort.botsort import BotSort
from boxmot.reid.core import ReID

if not torch.cuda.is_available():
    raise RuntimeError("GPU is not available.")

PROJECT_DIR = Path("/app")

VIDEO_LOCAL = Path(os.getenv("VIDEO_PATH", "/app/input/video.mp4"))
OUTPUT_LOCAL = Path(os.getenv("OUTPUT_PATH", "/app/output/result.mp4"))
REID_PT = PROJECT_DIR / "osnet_x0_25_msmt17.pt"
YOLO_MODEL = PROJECT_DIR / "yolov8m.pt"

USE_REID = os.getenv("USE_REID", "true").lower() in ("true", "1", "yes")

line1_x = int(os.getenv("LINE1_X", "170"))
line1_y1 = int(os.getenv("LINE1_Y1", "250"))
line1_y2 = int(os.getenv("LINE1_Y2", "400"))

line2_x = int(os.getenv("LINE2_X", "795"))
line2_y1 = int(os.getenv("LINE2_Y1", "250"))
line2_y2 = int(os.getenv("LINE2_Y2", "400"))

line_out = int(os.getenv("LINE_OUT", "900"))
REAL_DISTANCE_METERS = float(os.getenv("REAL_DISTANCE_METERS", "13"))

if not VIDEO_LOCAL.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_LOCAL}")

if not REID_PT.exists():
    raise FileNotFoundError(f"ReID model not found: {REID_PT}")

if not YOLO_MODEL.exists():
    raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL}")

OUTPUT_LOCAL.parent.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print("CONFIGURATION")
print("=" * 50)
print(f"Input video: {VIDEO_LOCAL}")
print(f"Output video: {OUTPUT_LOCAL}")
print(f"Use ReID: {USE_REID}")
print(f"Line 1: x={line1_x}, y1={line1_y1}, y2={line1_y2}")
print(f"Line 2: x={line2_x}, y1={line2_y1}, y2={line2_y2}")
print(f"Output line: x={line_out}")
print(f"Real distance: {REAL_DISTANCE_METERS} m")
print("=" * 50)

device = torch.device("cuda:0")

print("GPU:", torch.cuda.get_device_name(0))

model = YOLO(str(YOLO_MODEL))
model.to(device)

print("YOLO running with PyTorch FP32")

tracker_kwargs = dict(
    track_high_thresh=0.3,
    track_low_thresh=0.1,
    new_track_thresh=0.3,
    track_buffer=30,
    frame_rate=30,
    with_reid=USE_REID,
    use_cmc=False,
)

if USE_REID:
    reid = ReID(weights=REID_PT, device=device, half=False)
    tracker_kwargs["reid_model"] = reid.model
    print("ReID running with PyTorch FP32")

tracker = BotSort(**tracker_kwargs)

print(f"BoTSORT ready | with_reid={tracker.with_reid}")

cap = cv2.VideoCapture(str(VIDEO_LOCAL))

if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {VIDEO_LOCAL}")

REAL_FPS = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height}, FPS={REAL_FPS}, frames={total_frames}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(OUTPUT_LOCAL), fourcc, REAL_FPS, (width, height))

if not out.isOpened():
    raise RuntimeError(f"Failed to create output video: {OUTPUT_LOCAL}")

print("Starting PyTorch processing...")

frame_count = 0
total_cars = 0
vehicle_classes = [2, 3, 5, 7]
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5

GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
RED_COLOR = (0, 0, 255)

processing_times = []
enter_frame1 = {}
enter_frame2 = {}
track_speeds = {}
track_times = {}
trajectories = {}
names = model.names

while True:
    start_time = time.perf_counter()
    ret, frame = cap.read()

    if not ret:
        print("Video processing finished.")
        break

    frame_count += 1
    video_time = frame_count / REAL_FPS

    cv2.line(frame, (line1_x, line1_y1), (line1_x, line1_y2), BLUE_COLOR, 3)
    cv2.line(frame, (line2_x, line2_y1), (line2_x, line2_y2), BLUE_COLOR, 3)

    results = model(
        frame,
        device=0,
        verbose=False,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=640,
    )

    detections = []

    for box in results[0].boxes:
        cls = int(box.cls[0])

        if cls in vehicle_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append([x1, y1, x2, y2, float(box.conf[0]), cls])

    car_count = 0

    if detections:
        detections_np = np.asarray(detections, dtype=np.float32)

        try:
            tracked_results = tracker.update(detections_np, frame)

            if tracked_results is None or len(tracked_results) == 0:
                tracked_results = []

            car_count = len(tracked_results)
            total_cars = max(total_cars, car_count)

            for track in tracked_results:
                x1, y1, x2, y2 = map(int, track[:4])
                track_id = int(track[4])
                cls = int(track[6])
                name = names.get(cls, str(cls)) if isinstance(names, dict) else names[cls]
                color = GREEN_COLOR

                if y2 >= line1_y1 and y1 <= line1_y2:
                    x_new = int((x1 + x2) / 2)
                    y_new = int((y1 + y2) / 2)
                    trajectories.setdefault(track_id, []).append((x_new, y_new))
                    pts = trajectories[track_id][-30:]
                    trajectories[track_id] = pts

                    for j in range(1, len(pts)):
                        cv2.line(frame, pts[j - 1], pts[j], BLUE_COLOR, 2)

                if x2 >= line1_x and x1 <= line2_x and y2 >= line1_y1 and y1 <= line1_y2:
                    color = BLUE_COLOR
                    enter_frame1.setdefault(track_id, video_time)

                if track_id in enter_frame1 and x1 > line2_x:
                    if track_id not in enter_frame2:
                        enter_frame2[track_id] = video_time
                    color = RED_COLOR

                f1 = enter_frame1.get(track_id)
                f2 = enter_frame2.get(track_id)

                if f1 is not None and f2 is not None:
                    time_diff = f2 - f1

                    if time_diff > 0:
                        speed_kmh = (REAL_DISTANCE_METERS / time_diff) * 3.6
                        track_speeds[track_id] = speed_kmh

                        if track_id not in track_times:
                            track_times[track_id] = speed_kmh
                            print(f"ID {track_id}: {speed_kmh:.1f} km/h ({time_diff:.2f}s)")

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                if track_id in track_speeds:
                    label = f"{name} ID:{track_id} [{track_speeds[track_id]:.0f} km/h]"
                elif track_id in enter_frame1:
                    label = f"{name} ID:{track_id} [->]"
                else:
                    label = f"{name} ID:{track_id}"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if track_id in track_speeds and x2 > line_out:
                    enter_frame1.pop(track_id, None)
                    enter_frame2.pop(track_id, None)
                    track_speeds.pop(track_id, None)
                    track_times.pop(track_id, None)
                    trajectories.pop(track_id, None)

        except Exception as e:
            print(f"Tracker error: {e}")

    cv2.putText(frame, f"Cars: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN_COLOR, 2)
    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Distance: {REAL_DISTANCE_METERS}m", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE_COLOR, 1)
    cv2.putText(frame, f"Time: {video_time:.1f}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN_COLOR, 1)
    cv2.putText(frame, "PyTorch-FP32", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN_COLOR, 1)

    out.write(frame)

    elapsed = (time.perf_counter() - start_time) * 1000
    processing_times.append(elapsed)

    if frame_count % 50 == 0:
        print(f"Frame {frame_count}/{total_frames} | {elapsed:.1f} ms | {1000 / max(elapsed, 0.001):.2f} FPS")

cap.release()
out.release()

print("\n" + "=" * 50)
print("BENCHMARK - PyTorch + ReID FP32")
print("=" * 50)
print(f"Frames: {frame_count}")
print(f"Maximum vehicles: {total_cars}")

if processing_times:
    times = np.array(processing_times)
    avg_ms = np.mean(times)
    median_ms = np.median(times)
    p95_ms = np.percentile(times, 95)
    total_time = np.sum(times) / 1000

    print(f"Total processing time: {total_time:.2f} sec")
    print(f"Average latency: {avg_ms:.2f} ms/frame")
    print(f"Median latency: {median_ms:.2f} ms/frame")
    print(f"P95 latency: {p95_ms:.2f} ms/frame")
    print(f"Average FPS: {1000 / avg_ms:.2f}")
    print(f"Median FPS: {1000 / median_ms:.2f}")

print(f"Output: {OUTPUT_LOCAL}")
print("=" * 50)
