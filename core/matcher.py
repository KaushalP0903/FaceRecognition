import numpy as np
from config.configurations import (MATCH_THRESHOLD,UPDATE_THRESHOLD)

class FaceMatcher:

    def __init__(self):
        self.threshold = MATCH_THRESHOLD
        self.update_threshold = UPDATE_THRESHOLD

    def cosine_similarity(self, a, b):
        return float(np.dot(a, b))

    def match(self, embedding, database):

        best_name = "Unknown"
        best_score = -1
        second_best = -1

        for name, embeddings in database.data.items():
            if not embeddings:
                continue

            identity_score = -1
            for db_emb in embeddings:

                score = self.cosine_similarity(embedding, db_emb)

                if score > identity_score:
                    identity_score = score

            if identity_score > best_score:

                second_best = best_score
                best_score = identity_score
                best_name = name

            elif identity_score > second_best:

                second_best = identity_score

        return best_name, best_score, second_best

    def decide(
        self,
        best_name,
        best_score,
        second_best
    ):

        if best_score < self.threshold:
            return "Unknown", best_score, False

        if (best_score - second_best) < 0.05:
            return "Unknown", best_score, False

        allow_update = (best_score >= self.update_threshold)
        score_percent = best_score*100
        print(f"[INFO] {best_name} detected with {score_percent:.2f}% face match!")

        return (best_name,best_score,allow_update)
