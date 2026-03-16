"""Spotify recommendation helper based on detected emotion."""

from __future__ import annotations

import os
from typing import Any

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

EMOTION_TO_QUERY = {
    "Happy": "happy upbeat music",
    "Sad": "sad emotional songs",
    "Angry": "calming relaxing music",
    "Fear": "motivational music",
    "Surprise": "party songs",
    "Neutral": "lo-fi music",
    "Disgust": "instrumental music",
}


class SpotifyRecommender:
    """Client wrapper that fetches track recommendations using Spotify Search API."""

    def __init__(self, client_id: str | None = None, client_secret: str | None = None) -> None:
        client_id = client_id or os.getenv("SPOTIPY_CLIENT_ID")
        client_secret = client_secret or os.getenv("SPOTIPY_CLIENT_SECRET")

        if not client_id or not client_secret:
            raise ValueError(
                "Spotify credentials are missing. Set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET."
            )

        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        self.client = spotipy.Spotify(auth_manager=auth_manager)

    def recommend_tracks(self, emotion: str, limit: int = 5) -> list[dict[str, Any]]:
        """Fetch top tracks for mapped emotion query."""
        query = EMOTION_TO_QUERY.get(emotion, "mood music")
        results = self.client.search(q=query, type="track", limit=limit)

        tracks = []
        for item in results.get("tracks", {}).get("items", []):
            album_images = item.get("album", {}).get("images", [])
            cover = album_images[0]["url"] if album_images else ""

            tracks.append(
                {
                    "song_name": item.get("name", "Unknown"),
                    "artist": ", ".join(artist["name"] for artist in item.get("artists", [])) or "Unknown",
                    "album_cover": cover,
                    "spotify_url": item.get("external_urls", {}).get("spotify", ""),
                }
            )
        return tracks
