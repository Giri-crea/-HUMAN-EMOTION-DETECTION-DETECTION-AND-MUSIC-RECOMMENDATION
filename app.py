"""Streamlit frontend for emotion-based Spotify recommendations."""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from backend import EmotionDetector
from spotify_recommendation import SpotifyRecommender

st.set_page_config(page_title="Emotion Based Spotify Music Recommendation", page_icon="🎵", layout="centered")
st.title("Emotion Based Spotify Music Recommendation")
st.write("Upload a face image or capture one from your webcam to get Spotify song recommendations.")


@st.cache_resource
def load_detector() -> EmotionDetector:
    return EmotionDetector(model_path="model/emotion_model.h5")


@st.cache_resource
def load_recommender() -> SpotifyRecommender:
    return SpotifyRecommender()


def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    file_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(file_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Unable to decode image.")
    return bgr


def _handle_image(image_bytes: bytes) -> None:
    detector = load_detector()
    recommender = load_recommender()

    bgr = _bytes_to_bgr(image_bytes)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption="Input Image", use_container_width=True)

    try:
        prediction = detector.predict_emotion(bgr)
    except ValueError as err:
        st.error(str(err))
        return

    emotion = prediction["emotion"]
    confidence = prediction["confidence"]

    st.success(f"Detected Emotion: **{emotion}** ({confidence:.2%} confidence)")

    with st.spinner("Fetching Spotify recommendations..."):
        try:
            tracks = recommender.recommend_tracks(emotion=emotion, limit=5)
        except Exception as err:  # noqa: BLE001
            st.error(f"Could not fetch Spotify recommendations: {err}")
            return

    if not tracks:
        st.warning("No tracks found for this emotion.")
        return

    st.subheader("Top 5 Recommended Songs")
    for idx, track in enumerate(tracks, start=1):
        cols = st.columns([1, 3])
        with cols[0]:
            if track["album_cover"]:
                st.image(track["album_cover"], use_container_width=True)
        with cols[1]:
            st.markdown(f"**{idx}. {track['song_name']}**")
            st.write(f"Artist: {track['artist']}")
            if track["spotify_url"]:
                st.markdown(f"[▶ Play on Spotify]({track['spotify_url']})")
        st.divider()


tab_upload, tab_webcam = st.tabs(["Upload Image", "Webcam Capture"])

with tab_upload:
    uploaded_file = st.file_uploader("Upload a JPG/PNG image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        _handle_image(uploaded_file.read())

with tab_webcam:
    camera_image = st.camera_input("Capture an image")
    if camera_image is not None:
        _handle_image(camera_image.getvalue())

st.caption("Make sure SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET are configured as environment variables.")
