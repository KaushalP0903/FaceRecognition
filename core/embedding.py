import numpy as np
from insightface.model_zoo import get_model
from config.configurations import ARCFACE_MODEL, CTX_ID

class FaceEmbedder:

    def __init__(self):
        self.model = get_model(ARCFACE_MODEL)

        if self.model is None:
            raise RuntimeError("Failed to load ArcFace model")

        self.model.prepare(ctx_id=CTX_ID)

    def get_embedding(self, aligned_face):
        if aligned_face is None:
            return None

        emb = self.model.get_feat(aligned_face)
        
        # (1,512) → (512,)
        emb = emb.flatten()

        return self.normalize(emb)

    def normalize(self, emb):

        norm = np.linalg.norm(emb)

        if norm == 0:
            return emb

        return emb / norm
