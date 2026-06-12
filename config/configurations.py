from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

USE_YOLO = False

FRAME_SKIP = 0

CAMERA_SOURCE = 0

FRAME_SIZE = (640, 480)

YOLO_MODEL = str(BASE_DIR / "models" / "best.pt")
RETINAFACE_MODEL = str(BASE_DIR / "models" / "retinaface" / "detector.onnx")
ARCFACE_MODEL = str(BASE_DIR / "models" / "arcface" / "arcface.onnx")
LANDMARK106_MODEL = str(BASE_DIR / "models" / "retinaface" / "2d106det.onnx")

POSE_ESTIMATION = False
POSE_MODEL = str(BASE_DIR / "models" / "retinaface" / "1k3d68.onnx")

DATABASE_PATH = str(BASE_DIR / "database" / "faces.pkl")

CTX_ID = 0  # GPU = 0, CPU = -1

FACE_CONF_THRESHOLD = 0.8
FACE_MARGIN = 20

MAX_AGE = 50
N_INIT = 2
MAX_COSINE_DISTANCE = 0.3

MATCH_THRESHOLD = 0.75
UPDATE_THRESHOLD = 0.85
AUTO_UPDATE_DATABASE = False

MAX_YAW = 45
MAX_PITCH = 35
MAX_ROLL = 35