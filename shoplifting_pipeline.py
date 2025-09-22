"""
shoplifting_pipeline.py

Features:
- YOLOv8 detection (persons + objects)
- DeepSORT tracking (persons and objects separately)
- Detects concealment: product disappears while person remains in frame
- Suspicious persons get red bounding box + "Suspicious" label
- Annotated output.mp4 with GUI (header, footer, bounding boxes)
- Suspicious clips saved into ./suspicious/ with GUI overlay

- Usage:
    python shoplifting_pipeline.py --source demo1.mp4 --yolo_model yolov8n.pt --output processed_output2.mp4
"""


import os
import argparse
from collections import deque, defaultdict
from datetime import datetime

# --- Patch cv2.imshow for headless environments ---
import cv2
if not hasattr(cv2, "imshow"):
    cv2.imshow = lambda *a, **k: None

import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ----------------------------
# Config
# ----------------------------
SUSPICIOUS_FOLDER = "suspicious"
os.makedirs(SUSPICIOUS_FOLDER, exist_ok=True)

CLIP_DURATION_FRAMES = 120
CONCEALMENT_FRAME_WINDOW = 30
ALERT_COOLDOWN_SECONDS = 20

PERSON_CLASS_ID = 0
PRODUCT_CLASS_IDS = None  # None = treat all non-person as products

# ----------------------------
# Utility functions
# ----------------------------
def iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    a1 = max(1, (b1[2]-b1[0])*(b1[3]-b1[1]))
    a2 = max(1, (b2[2]-b2[0])*(b2[3]-b2[1]))
    return inter / union if (union := a1 + a2 - inter) > 0 else 0

def save_clip(frames, out_path, fps=20):
    if not frames: return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
    for f in frames: out.write(f)
    out.release()

def draw_transparent_rect(img, top_left, bottom_right, color, alpha=0.4):
    """Draw semi-transparent rectangle"""
    overlay = img.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

# ----------------------------
# Main pipeline
# ----------------------------
def run_pipeline(source, yolo_model_path, device='cpu', output_path="output.mp4"):
    print("Loading YOLO:", yolo_model_path)
    yolo = YOLO(yolo_model_path)
    DET_CONF = 0.35

    person_tracker = DeepSort(max_age=30, n_init=3)
    object_tracker = DeepSort(max_age=30, n_init=3)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video {source}: {fw}x{fh} @ {fps:.2f} FPS")

    writer = None
    frame_idx = 0
    global_buffer = deque(maxlen=CLIP_DURATION_FRAMES*2)

    # per-track state
    person_frame_buffer = defaultdict(lambda: deque(maxlen=CLIP_DURATION_FRAMES))
    person_last_alert = defaultdict(lambda: datetime.fromtimestamp(0))
    person_is_suspicious = set()
    object_last_seen = {}
    object_assigned_person = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            global_buffer.append(frame.copy())

            # --- YOLO detection ---
            results = yolo.predict(frame, imgsz=640, conf=DET_CONF, verbose=False)
            detections = []
            if len(results) > 0:
                res = results[0]
                boxes = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                for box, score, cls in zip(boxes, scores, cls_ids):
                    detections.append({"bbox": box.tolist(), "score": float(score), "class_id": int(cls)})

            # split detections
            person_dets, object_dets = [], []
            for det in detections:
                if det["class_id"] == PERSON_CLASS_ID:
                    person_dets.append(det)
                else:
                    if PRODUCT_CLASS_IDS is None or det["class_id"] in PRODUCT_CLASS_IDS:
                        object_dets.append(det)

            # --- DeepSORT update ---
            def dets_to_ds(dets, label):
                return [([int(x1), int(y1), int(x2-x1), int(y2-y1)], d["score"], label)
                        for d in dets for x1,y1,x2,y2 in [d["bbox"]]]
            person_tracks = person_tracker.update_tracks(dets_to_ds(person_dets,"person"), frame=frame)
            object_tracks = object_tracker.update_tracks(dets_to_ds(object_dets,"object"), frame=frame)

            current_person_boxes, current_object_boxes = {}, {}
            for tr in person_tracks:
                if not tr.is_confirmed(): continue
                pid = tr.track_id
                x,y,w,h = map(int, tr.to_tlwh()); bbox=(x,y,x+w,y+h)
                current_person_boxes[pid]=bbox
                person_frame_buffer[pid].append(frame.copy())
            for tr in object_tracks:
                if not tr.is_confirmed(): continue
                oid = tr.track_id
                x,y,w,h = map(int,tr.to_tlwh()); bbox=(x,y,x+w,y+h)
                current_object_boxes[oid]=bbox
                object_last_seen[oid]=frame_idx

            # --- associate objects to persons by IoU ---
            for oid, obox in current_object_boxes.items():
                best_pid = None
                best_iou = 0.0
                for pid, pbox in current_person_boxes.items():
                    val = iou(obox, pbox)
                    if val > best_iou:
                        best_iou = val
                        best_pid = pid
                if best_pid and best_iou > 0.2:
                    object_assigned_person[oid] = (best_pid, frame_idx)

            # --- detect concealment ---
            for oid,last_seen in list(object_last_seen.items()):
                if oid not in current_object_boxes:
                    if frame_idx-last_seen<=CONCEALMENT_FRAME_WINDOW:
                        assoc=object_assigned_person.get(oid)
                        if assoc:
                            pid,assoc_frame=assoc
                            if pid in current_person_boxes and frame_idx-assoc_frame<=CONCEALMENT_FRAME_WINDOW:
                                now=datetime.now()
                                if (now-person_last_alert[pid]).total_seconds()>=ALERT_COOLDOWN_SECONDS:
                                    person_last_alert[pid]=now
                                    person_is_suspicious.add(pid)   # <- permanent suspicion
                                    clip=list(person_frame_buffer[pid]) or list(global_buffer)
                                    timestamp=now.strftime("%Y%m%d_%H%M%S")
                                    out_path=f"{SUSPICIOUS_FOLDER}/suspicious_pid{pid}_obj{oid}_{timestamp}.mp4"
                                    save_clip([f.copy() for f in clip],out_path,fps=int(fps))
                                    print(f"[ALERT] Person {pid} concealment -> saved {out_path}")
                        object_last_seen.pop(oid,None)
                        object_assigned_person.pop(oid,None)

            # --- GUI overlay ---
            annotated=frame.copy()
            header_h,footer_h=40,30

            # semi-transparent header
            annotated = draw_transparent_rect(annotated,(0,0),(fw,header_h),(0,0,0),alpha=0.5)
            cv2.putText(annotated,"ShopEyeQ - Smart Shoplifting Detection",(10,25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

            status_text="Status: Normal"; status_color=(0,200,0)
            for pid,bbox in current_person_boxes.items():
                x1,y1,x2,y2=bbox
                if pid in person_is_suspicious:
                    color=(0,0,255); label=f"P:{pid} Suspicious"
                    status_text="ALERT: Suspicious Activity!!!"; status_color=(0,0,255)
                else:
                    color=(0,255,0); label=f"P:{pid}"
                cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
                cv2.putText(annotated,label,(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
            for oid,bbox in current_object_boxes.items():
                x1,y1,x2,y2=bbox
                cv2.rectangle(annotated,(x1,y1),(x2,y2),(255, 144, 30), 2)
                cv2.putText(annotated,f"O:{oid}",(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 144, 30),1)

            # semi-transparent footer
            annotated = draw_transparent_rect(annotated,(0,fh-footer_h),(fw,fh),(0,0,0),alpha=0.5)
            cv2.putText(annotated,status_text,(10,fh-8),cv2.FONT_HERSHEY_SIMPLEX,0.7,status_color,2)

            # init writer
            if writer is None:
                fourcc=cv2.VideoWriter_fourcc(*"mp4v")
                writer=cv2.VideoWriter(output_path,fourcc,float(fps),(fw,fh))
            writer.write(annotated)

    finally:
        cap.release()
        if writer: writer.release()
        print("Finished. Output video:",output_path)

# ----------------------------
# CLI
# ----------------------------
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,help="video file or camera index")
    p.add_argument("--yolo_model",default="yolov8n.pt",help="YOLOv8 model path")
    p.add_argument("--device",default="cpu",help="cpu or cuda")
    p.add_argument("--output",default="output.mp4",help="annotated output file")
    a=p.parse_args()
    run_pipeline(a.source,a.yolo_model,device=a.device,output_path=a.output)
