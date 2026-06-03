import pickle
import os
from config.configurations import DATABASE_PATH

class FaceDatabase:

    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path
        self.data = {}
        self.load()

    def add_person(self, name):
        if name not in self.data:
            self.data[name] = []

    def add_embedding(
        self,
        name,
        embedding,
        max_samples=50
    ):

        if name not in self.data:
            self.data[name] = []

        self.data[name].append(embedding)

        if len(self.data[name]) > max_samples:
            self.data[name].pop(0)

    def add_embedding_to_existing_person(
        self,
        name,
        embedding,
        max_samples=50
    ):

        if name not in self.data:
            return False

        self.add_embedding(name, embedding, max_samples)
        return True

    def save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with open(self.db_path, "wb") as f:
            pickle.dump(self.data, f)

        print(f"[INFO] Saved {len(self.data)} identities to {self.db_path}")

    def load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                self.data = pickle.load(f)
            print(f"[INFO] Loaded {len(self.data)} identities from {self.db_path}")
        else:
            print(f"[INFO] New database created at {self.db_path}")
