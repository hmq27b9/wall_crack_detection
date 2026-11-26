# config/database.py
"""MongoDB connection helper. The database is taken from the URI."""
from __future__ import annotations

from pymongo import MongoClient
from pymongo.database import Database

from .settings import MONGODB_URI


def get_client() -> MongoClient:
    """Create a MongoClient and verify connectivity."""
    if not MONGODB_URI:
        raise ValueError(
            "MONGODB_URI is not set. Add it to your .env file."
        )
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")  # fail fast
    return client


def get_database() -> Database:
    """Return the database encoded in the connection string."""
    return get_client().get_default_database()