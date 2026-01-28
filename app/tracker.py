import time

def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (areaA + areaB - inter)

class Tracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}

    def update(self, persons):
        now = time.time()
        updated = {}


        for tid, track in self.tracks.items():
            best_box = None
            best_iou = 0
            for p in persons:
                v = iou(track["bbox"], p)
                if v > best_iou:
                    best_iou = v
                    best_box = p
            if best_iou > 0.3:
                updated[tid] = {
                    **track,
                    "bbox": best_box,
                    "last_seen": now
                }

        for p in persons:
            if not any(iou(p, t["bbox"]) > 0.3 for t in updated.values()):
                updated[self.next_id] = {
                    "bbox": p,
                    "nohelmet_since": None,
                    "last_violation": 0,
                    "last_seen": now,
                }
                self.next_id += 1

        # удаляем устаревшие
        self.tracks = {tid: t for tid, t in updated.items() if now - t["last_seen"] < 3.0}
        return self.tracks
