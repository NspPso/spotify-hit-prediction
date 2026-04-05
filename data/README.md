# Data Notes

- Source: [maharshipandya/spotify-tracks-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
- Download URL:
  - `https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv`
- Local file:
  - `data/raw/spotify_tracks.csv`

## Expected Columns

- identifiers and text: `track_id`, `artists`, `album_name`, `track_name`
- target source: `popularity`
- metadata/audio features:
  - `duration_ms`
  - `explicit`
  - `danceability`
  - `energy`
  - `key`
  - `loudness`
  - `mode`
  - `speechiness`
  - `acousticness`
  - `instrumentalness`
  - `liveness`
  - `valence`
  - `tempo`
  - `time_signature`
  - `track_genre`

## Project Target

This project derives the binary label as:

- `hit = 1` if `popularity >= 7`
- `hit = 0` otherwise

The raw `popularity` column is never used as a predictor.
