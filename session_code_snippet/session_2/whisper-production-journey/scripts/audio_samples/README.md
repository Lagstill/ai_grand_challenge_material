# Audio Samples

## Auto-generated Samples

Run the generation script:
```bash
pip install gtts pydub
python ../scripts/generate_audio.py
```

## Manual Download Options

If generation doesn't work, download sample audio from:

### Option 1: LibriSpeech (English)
```bash
# Download a sample
curl -L "https://www.openslr.org/resources/12/dev-clean.tar.gz" | tar xzv --strip-components=4 -C . -- "LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac"
ffmpeg -i 1272-128104-0000.flac -ar 16000 sample_10s.wav
```

### Option 2: Common Voice (Multiple Languages)
Visit: https://commonvoice.mozilla.org/

### Option 3: Your Own Recording
```bash
# Record 10 seconds using sox
sox -d -r 16000 -c 1 sample_10s.wav trim 0 10
```

## Required Files

The demo needs these files:
- `sample_5s.wav` - Short sample for warmup
- `sample_10s.wav` - Standard test sample
- `sample_30s.wav` - Longer sample for streaming demo

All files should be:
- WAV format
- 16kHz sample rate
- Mono channel
