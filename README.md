# Emotion Detection + Spotify Music Recommendation App

This project detects a user's facial emotion from an uploaded image/webcam capture and recommends Spotify tracks that match the mood.

## Tech Stack
- Python
- TensorFlow/Keras (CNN for FER2013)
- OpenCV (face detection)
- Streamlit (frontend)
- Spotify Web API via Spotipy

## Project Structure

```text
emotion-spotify-app/
│
├── app.py
├── backend.py
├── train_model.py
├── spotify_recommendation.py
│
├── model/
│   └── emotion_model.h5
│
├── utils/
│   └── preprocess.py
│
├── requirements.txt
└── README.md
```

## Dataset
Use **FER2013** CSV dataset.

1. Download `fer2013.csv`.
2. Place it at `data/fer2013.csv` (or pass a custom path with `--data-path`).

Expected columns:
- `emotion` (0-6)
- `pixels` (space-separated 48x48 grayscale values)

Emotion index mapping in FER2013 aligns with:
- 0 Angry
- 1 Disgust
- 2 Fear
- 3 Happy
- 4 Sad
- 5 Surprise
- 6 Neutral

## Train the Emotion Model

```bash
python train_model.py
```

Optional flags:

```bash
python train_model.py --data-path data/fer2013.csv --epochs 50 --batch-size 64 --output model/emotion_model.h5
```

The training pipeline includes:
- Conv2D + BatchNormalization blocks
- MaxPooling + Dropout regularization
- Dense + Softmax output layer

With proper training and tuning, this architecture typically reaches ~60-75% validation accuracy on FER2013.

## Spotify Setup
1. Create a Spotify developer app: <https://developer.spotify.com/dashboard>
2. Get your credentials:
   - `SPOTIPY_CLIENT_ID`
   - `SPOTIPY_CLIENT_SECRET`
3. Export them in your shell:

```bash
export SPOTIPY_CLIENT_ID="your_client_id"
export SPOTIPY_CLIENT_SECRET="your_client_secret"
```

## Emotion → Spotify Query Mapping
- Happy → `happy upbeat music`
- Sad → `sad emotional songs`
- Angry → `calming relaxing music`
- Fear → `motivational music`
- Surprise → `party songs`
- Neutral → `lo-fi music`
- Disgust → `instrumental music`

## Run the Web App

```bash
streamlit run app.py
```

Then open:
- <http://localhost:8501>

## Features
- Upload image for emotion detection
- Webcam capture via Streamlit
- Face detection using OpenCV Haarcascade
- Emotion prediction using trained CNN model
- Top 5 Spotify track recommendations with:
  - Album cover
  - Song name
  - Artist
  - Spotify play link

## Notes
- Make sure `model/emotion_model.h5` exists before launching `app.py`.
- If no face is detected, the app returns an error message.
- Spotify recommendations require working client credentials.
