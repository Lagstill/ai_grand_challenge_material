#!/usr/bin/env python3
"""
Audio Sample Generator for Whisper Demo
========================================

Generates test audio files of various lengths using TTS.

Usage:
    pip install gtts pydub
    python generate_samples.py

For demo purposes, we'll also provide pre-recorded samples
if TTS isn't available.
"""

import os
import sys
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent / "audio_samples"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_with_gtts():
    """Generate audio using Google Text-to-Speech (requires internet)"""
    try:
        from gtts import gTTS
        from pydub import AudioSegment
        import io
        
        samples = {
            "sample_5s.wav": "Hello, this is a short test audio sample for the Whisper transcription demo.",
            "sample_10s.wav": "Hello and welcome to the Whisper production journey workshop. Today we will learn how to take a machine learning model from notebook to production. This involves optimization, scaling, and proper engineering practices.",
            "sample_30s.wav": """Welcome to the AI Grand Challenge workshop on production machine learning. 
            Today's session covers the journey from research to production. 
            Many teams face challenges when deploying AI models at scale. 
            The key issues include latency optimization, memory management, and handling concurrent requests.
            We'll explore solutions like quantization, streaming, and proper API design.
            By the end of this session, you'll have practical knowledge to deploy your own models.
            Let's begin with understanding why notebooks aren't production ready.""",
        }
        
        for filename, text in samples.items():
            filepath = OUTPUT_DIR / filename
            if filepath.exists():
                print(f"⏭️  Skipping {filename} (already exists)")
                continue
                
            print(f"🎤 Generating {filename}...")
            
            # Generate MP3 with gTTS
            tts = gTTS(text=text, lang='en', slow=False)
            mp3_buffer = io.BytesIO()
            tts.write_to_fp(mp3_buffer)
            mp3_buffer.seek(0)
            
            # Convert to WAV (required for Whisper)
            audio = AudioSegment.from_mp3(mp3_buffer)
            audio = audio.set_frame_rate(16000).set_channels(1)  # Whisper optimal settings
            audio.export(str(filepath), format="wav")
            
            duration = len(audio) / 1000
            print(f"   ✅ Created {filename} ({duration:.1f}s)")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  gTTS not available: {e}")
        return False


def generate_with_pyttsx3():
    """Generate audio using pyttsx3 (offline, system TTS)"""
    try:
        import pyttsx3
        import wave
        
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed
        
        samples = {
            "sample_5s.wav": "Hello, this is a test audio sample.",
            "sample_10s.wav": "Welcome to the Whisper production journey. Today we learn about model optimization and deployment.",
            "sample_30s.wav": "This is a longer test sample for the production machine learning workshop. We will cover many topics including latency optimization, memory management, and concurrent request handling.",
        }
        
        for filename, text in samples.items():
            filepath = OUTPUT_DIR / filename
            if filepath.exists():
                print(f"⏭️  Skipping {filename} (already exists)")
                continue
                
            print(f"🎤 Generating {filename}...")
            engine.save_to_file(text, str(filepath))
            engine.runAndWait()
            print(f"   ✅ Created {filename}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  pyttsx3 not available: {e}")
        return False


def generate_silent_samples():
    """Generate silent audio files as fallback"""
    try:
        from pydub import AudioSegment
        from pydub.generators import Sine
        
        # Create samples with a low tone (easier than silence for Whisper)
        durations = [5, 10, 30]
        
        for duration in durations:
            filename = f"sample_{duration}s.wav"
            filepath = OUTPUT_DIR / filename
            
            if filepath.exists():
                print(f"⏭️  Skipping {filename} (already exists)")
                continue
            
            print(f"🔊 Generating {filename} (tone placeholder)...")
            
            # Generate a low frequency tone
            tone = Sine(440).to_audio_segment(duration=duration * 1000)
            tone = tone.set_frame_rate(16000).set_channels(1)
            tone = tone - 20  # Reduce volume
            tone.export(str(filepath), format="wav")
            
            print(f"   ✅ Created {filename} (tone placeholder)")
        
        return True
        
    except ImportError:
        print("⚠️  pydub not available")
        return False


def create_download_instructions():
    """Create instructions for downloading sample audio"""
    readme = OUTPUT_DIR / "README.md"
    
    content = """# Audio Samples

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
"""
    
    readme.write_text(content)
    print(f"📝 Created {readme}")


def main():
    print("=" * 50)
    print("Audio Sample Generator")
    print("=" * 50 + "\n")
    
    # Try different generation methods
    if not generate_with_gtts():
        if not generate_with_pyttsx3():
            if not generate_silent_samples():
                print("\n❌ Could not generate audio files.")
                print("   Please install: pip install gtts pydub")
    
    # Always create instructions
    create_download_instructions()
    
    print("\n" + "=" * 50)
    print("Generated files in:", OUTPUT_DIR)
    print("=" * 50)
    
    # List generated files
    for f in sorted(OUTPUT_DIR.glob("*.wav")):
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
