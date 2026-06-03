from deep_sort_realtime.deepsort_tracker import DeepSort

from config.configurations import (MAX_AGE,N_INIT,MAX_COSINE_DISTANCE)

class FaceTracker:

    def __init__(self):

        self.tracker = DeepSort(
            max_age=MAX_AGE,
            n_init=N_INIT,
            max_cosine_distance=MAX_COSINE_DISTANCE,
            nn_budget=100
        )

    def update(self, detections, frame):

        tracks_input = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            tracks_input.append(([x1, y1, w, h], 1.0, 'face'))

        tracks = self.tracker.update_tracks(tracks_input,frame=frame)

        results = []

        for track in tracks:

            if not track.is_confirmed():
                continue

            l, t, r, b = map(int, track.to_ltrb())

            matched_det = None
            best_iou = 0

            for det in detections:

                dx1, dy1, dx2, dy2 = det["bbox"]
                
                iou = self.compute_iou((l, t, r, b), (dx1, dy1, dx2, dy2))

                if iou > best_iou:
                    best_iou = iou
                    matched_det = det

            if matched_det is None:
                continue

            results.append({
                "track_id": track.track_id,
                "bbox": (l, t, r, b),
                "recognition_bbox": matched_det["bbox"],
                "landmarks": matched_det["landmarks"]
            })

        return results

    def compute_iou(self, boxA, boxB):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0,yB - yA)

        boxAArea = ((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = ((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        return interArea / (boxAArea + boxBArea - interArea + 1e-6)
