import cv2
import os
from datetime import datetime

from core.frame_manager import FrameManager
from core.detector import PersonDetector
from core.face_detector import FaceDetector
from core.aligner import FaceAligner
from core.embedding import FaceEmbedder
from core.matcher import FaceMatcher
from core.tracker import FaceTracker
from database.database import FaceDatabase
from utils.drawing import draw_tracks
from core.landmark106 import Landmark106
from core.pose_estimator import PoseEstimator

from config.configurations import (
    AUTO_UPDATE_DATABASE,
    FRAME_SKIP,
    MAX_YAW,
    MAX_PITCH,
    MAX_ROLL,
    POSE_ESTIMATION,
    DETECTION_INTERVAL
)

def save_enrollment_image(
    face_img,
    person_name,
    sample_no
):

    folder = os.path.join(
        "face_logs",
        "enrollments",
        person_name
    )

    os.makedirs(
        folder,
        exist_ok=True
    )

    filename = os.path.join(
        folder,
        f"enroll_{sample_no:02d}.jpg"
    )

    cv2.imwrite(
        filename,
        face_img
    )

def save_detection_frame(
    frame,
    identity
):

    folder = os.path.join(
        "face_logs",
        "detections",
        identity
    )

    os.makedirs(
        folder,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = os.path.join(
        folder,
        f"{timestamp}.jpg"
    )

    cv2.imwrite(
        filename,
        frame
    )

def main():

    frame_manager = FrameManager()
    person_detector = PersonDetector()
    face_detector = FaceDetector()
    aligner = FaceAligner()
    landmark106 = Landmark106()
    pose_estimator = PoseEstimator()
    embedder = FaceEmbedder()
    matcher = FaceMatcher()
    tracker = FaceTracker()
    database = FaceDatabase()
    cached_tracks = []
    prev_detections = []
    locked_matches = {}

    enroll_mode = False
    enroll_name = ""
    enroll_embeddings = []
    ENROLL_SAMPLES = 20

    while True:
        frame = frame_manager.get_frame()

        if frame is None:
            break

        #if frame_manager.frame_count % max(1, FRAME_SKIP) == 0:
        if frame_manager.frame_count % DETECTION_INTERVAL == 0:
            persons = person_detector.detect(frame)
            detections = face_detector.detect(frame, persons)
            prev_detections = detections

        else:   
            detections = prev_detections

        tracks = tracker.update(detections,frame)

        final_tracks = []
        active_track_ids = set()

        for t in tracks:
            track_id = t["track_id"]
            active_track_ids.add(track_id)

            if track_id in locked_matches:
                t["identity"], t["score"] = locked_matches[track_id]
                final_tracks.append(t)
                continue

            x1, y1, x2, y2 = t.get("recognition_bbox", t["bbox"])

            landmarks106 = landmark106.get_landmarks(frame, (x1, y1, x2, y2))

            if landmarks106 is None:
                continue

            t["landmarks106"] = landmarks106

            frame_h, frame_w = frame.shape[:2]
            x1 = max(0, min(frame_w, x1))
            y1 = max(0, min(frame_h, y1))
            x2 = max(0, min(frame_w, x2))
            y2 = max(0, min(frame_h, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = frame[y1:y2, x1:x2]

            if landmarks106 is None:
                continue

            # Convert to local face coordinates

            local_landmarks106 = [(px - x1, py - y1) for px, py in landmarks106]


            aligned_face = aligner.align(face_crop, local_landmarks106)

            if aligned_face is None:
                continue
            
            if POSE_ESTIMATION:
                pose = pose_estimator.estimate_pose(aligned_face)

                #print("[DEBUG] Pose:", pose)

                if pose is not None:

                    yaw = pose["yaw"]
                    pitch = pose["pitch"]
                    roll = pose["roll"]

                    #print(f"[DEBUG] "f"Yaw={yaw:.2f} "f"Pitch={pitch:.2f} "f"Roll={roll:.2f}")

            embedding = embedder.get_embedding(aligned_face)

            if embedding is None:
                continue

            if enroll_mode:
                enroll_embeddings.append(embedding)
                save_enrollment_image(aligned_face, enroll_name, len(enroll_embeddings))

                print(f"[INFO] Captured " f"{len(enroll_embeddings)} / " f"{ENROLL_SAMPLES}")

                if len(enroll_embeddings) >= ENROLL_SAMPLES:
                    database.add_person(enroll_name)

                    for emb in enroll_embeddings:
                        database.add_embedding(enroll_name,emb)

                    database.save()

                    print(f"[INFO] Enrollment complete " f"for {enroll_name}")

                    enroll_mode = False
                    enroll_embeddings = []

                continue
            best_name, best_score, second_best = (matcher.match(embedding, database))
            identity, score, allow_update = (matcher.decide(best_name,best_score,second_best))

            if allow_update and identity != "Unknown":
                updated = database.add_embedding_to_existing_person(identity, embedding)
                if updated:
                    if AUTO_UPDATE_DATABASE :
                        database.save()
                        print(f"[INFO] Added new scan to {identity}'s database record")

            t["identity"] = identity
            t["score"] = score

            if identity != "Unknown":
                save_detection_frame( frame, identity)
                locked_matches[track_id] = (identity, score)

            final_tracks.append(t)

        locked_matches = {
            track_id: match
            for track_id, match in locked_matches.items()
            if track_id in active_track_ids
        }

        draw_tracks(frame, final_tracks)

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('e'):

            enroll_name = input(
                "Enter person name: "
            )

            enroll_mode = True
            enroll_embeddings = []

            print(f"[INFO] Enrollment started "f"for {enroll_name}")

        if key == 27:
            break

    frame_manager.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()