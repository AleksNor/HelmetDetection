import time
import cv2
from ultralytics import YOLO
from pathlib import Path
from typing import Optional

from app.config import settings
from app.tracker import Tracker, iou

class VideoProcessor:
    def __init__(self, source: str, logger=None):
        self.source = source
        self.logger = logger
        self.cap = self._open_capture(source)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # модель
        if not Path(settings.MODEL_PATH).exists():
            raise RuntimeError(f"Model file not found: {settings.MODEL_PATH}")
        self.model = YOLO(settings.MODEL_PATH).to(settings.DEVICE)

        self.tracker = Tracker()
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.skip = max(int(round(fps / settings.PROCESS_FPS)), 1)
        self.frame_idx = 0
        self.last_frame = None

        settings.OUTPUT_DIR.mkdir(exist_ok=True)
        print(f'DEVICE {self.model.device}')

    def _open_capture(self, source: str) -> Optional[cv2.VideoCapture]:
        # пробуем разные способы открытия
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return cap
        try:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if cap.isOpened():
                return cap
        except Exception:
            pass
        return None

    def read(self):
        # чтение кадра с автоматическим переподключением
        if not self.cap or not self.cap.isOpened():
            self.cap = self._open_capture(self.source)
            time.sleep(0.2)
            if not self.cap:
                return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
            return None

        self.frame_idx += 1
        if self.frame_idx % self.skip != 0 and self.last_frame is not None:
            return self.last_frame

        now = time.time()
        annotated = frame.copy()

        # YOLO
        result = self.model(frame, verbose=False)[0]

        persons, heads, helmets = [], [], []
        for box in result.boxes:
            label = self.model.names[int(box.cls)]
            conf = float(box.conf)
            if label == "person" and conf < settings.CONF_PERSON:
                continue
            if label == "head" and conf < settings.CONF_HEAD:
                continue
            if label == "helmet" and conf < settings.CONF_HELMET:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = (x1, y1, x2, y2)
            if label == "person":
                persons.append(bbox)
            elif label == "head":
                heads.append(bbox)
            elif label == "helmet":
                helmets.append(bbox)

        tracks = self.tracker.update(persons)

        for tid, track in tracks.items():
            pb = track["bbox"]
            has_head = any(iou(pb, h) > 0.15 for h in heads)
            has_helmet = any(iou(pb, h) > 0.15 for h in helmets)
            color = (0, 0, 255) if has_head and not has_helmet else (255, 255, 0)

            cv2.rectangle(annotated, pb[:2], pb[2:], color, 2)
            cv2.putText(annotated, f"person #{tid}", (pb[0], pb[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            for h in heads:
                if iou(pb, h) > 0.15:
                    cv2.rectangle(annotated, h[:2], h[2:], (0,255,255), 2)
            for h in helmets:
                if iou(pb, h) > 0.15:
                    cv2.rectangle(annotated, h[:2], h[2:], (0,255,0), 2)

            # логика нарушения
            if has_head and not has_helmet:
                if track["nohelmet_since"] is None:
                    track["nohelmet_since"] = now
                duration = now - track["nohelmet_since"]
                cv2.putText(annotated, f"NO HELMET {duration:.1f}s",
                            (pb[0], pb[3] + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                if duration >= settings.VIOLATION_SECONDS and now - track["last_violation"] >= settings.COOLDOWN_SECONDS:
                    filename = settings.OUTPUT_DIR / f"violation_{tid}_{int(now)}.jpg"
                    cv2.imwrite(str(filename), annotated)
                    track["last_violation"] = now
                    track["nohelmet_since"] = None
            else:
                track["nohelmet_since"] = None

        self.last_frame = annotated
        return annotated

    def release(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass