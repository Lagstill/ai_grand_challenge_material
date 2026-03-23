# Audio Samples

This directory contains test audio files for the Whisper production journey demo.

## Quick Setup

### Option 1: Generate synthetic audio (recommended for demo)
```bash
# Install dependencies
pip install gtts pydub

# Generate samples
python ../scripts/generate_audio.py
```

### Option 2: Download from LibriSpeech
```bash
# Short sample (~10s)
curl -L "https://www.openslr.org/resources/12/dev-clean.tar.gz" -o dev-clean.tar.gz
tar -xzf dev-clean.tar.gz
cp LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac .
ffmpeg -i 1272-128104-0000.flac -ar 16000 -ac 1 sample_10s.wav
```

### Option 3: Create test tones with ffmpeg
```bash
# 5 second sample (440Hz tone)
ffmpeg -f lavfi -i "sine=frequency=440:duration=5" -ar 16000 -ac 1 sample_5s.wav

# 10 second sample
ffmpeg -f lavfi -i "sine=frequency=440:duration=10" -ar 16000 -ac 1 sample_10s.wav

# 30 second sample  
ffmpeg -f lavfi -i "sine=frequency=440:duration=30" -ar 16000 -ac 1 sample_30s.wav
```

### Option 4: Record your own
```bash
# Using sox
sox -d -r 16000 -c 1 sample_10s.wav trim 0 10

# Or use any recording app and convert:
ffmpeg -i your_recording.m4a -ar 16000 -ac 1 sample_10s.wav
```

## Required Files

| File | Duration | Purpose |
|------|----------|---------|
| `sample_5s.wav` | ~5s | Model warmup |
| `sample_10s.wav` | ~10s | Standard load testing |
| `sample_30s.wav` | ~30s | Streaming demo |

## Audio Specifications

For optimal Whisper performance:
- **Format**: WAV (PCM)
- **Sample Rate**: 16000 Hz
- **Channels**: 1 (Mono)
- **Bit Depth**: 16-bit

## Convert Existing Audio

```bash
# Convert any audio to Whisper-optimal format
ffmpeg -i input.mp3 -ar 16000 -ac 1 -acodec pcm_s16le output.wav
```

## Hindi / Multilingual Samples

For Problem Statement teams working with Indian languages (PS05, PS06):

```bash
# Download Common Voice Hindi samples
# Visit: https://commonvoice.mozilla.org/hi/datasets

# Or use Google TTS for Hindi
pip install gtts
python -c "from gtts import gTTS; gTTS('नमस्ते, यह एक परीक्षण है', lang='hi').save('hindi_test.mp3')"
ffmpeg -i hindi_test.mp3 -ar 16000 -ac 1 hindi_sample.wav
```
