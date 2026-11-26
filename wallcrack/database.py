from __future__ import annotations

from datetime import datetime, timezone
from pymongo import MongoClient, ASCENDING


class CrackRepository:
    def __init__(self, uri: str, database: str, collection: str):
        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Fail early if MongoDB is unreachable.
        self.client.admin.command("ping")
        self.collection = self.client[database][collection]
        self.collection.create_index(
            [("wall_id", ASCENDING), ("inspected_at", ASCENDING)]
        )
        self.collection.create_index("image.sha256")

    def save_inspection(self, document: dict):
        document["inspected_at"] = datetime.now(timezone.utc)
        return self.collection.insert_one(document).inserted_id

    def close(self):
        self.client.close()
