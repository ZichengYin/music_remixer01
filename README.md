# Music Remixer

A Streamlit application that applies audio effects to songs and generates remixed versions.

## Features

- 11 preset effects (pitch shift, speed, reverb, EQ, etc.)
- BPM detection and waveform visualization
- Background audio overlay (crackle, ambient)
- Random theme

## File Structure

- streamlit_app.py - Frontend interface
- remix_engine.py - Audio processing engine
- requirements.txt - Python dependencies

## Usage Limitations

- Main track supports MP3 only
- Background tracks support MP3 or WAV

## Dependencies

Python 3.8+, key packages: streamlit, pydub, pedalboard, librosa, matplotlib
