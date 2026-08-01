# Voice → Whisper (local) → Ollama (Jetson)

Free/open-source STT — **no Whisper subscription**. Ollama stays on the Jetson; STT runs on your laptop (or on the Jetson if you insist).

## Install (Mac or Linux)

```bash
# use your lerobot conda env if you have it
conda activate lerobot   # optional

# One-shot: pip deps + prefetch Whisper weights (default tiny.en)
./install_whisper.sh
# or:  ./install_whisper.sh base.en
#      ./install_whisper.sh tiny.en --faster   # also HF faster-whisper
#      ./install_whisper.sh tiny.en --cpp      # clone whisper.cpp + ggml
#      ./install_whisper.sh tiny.en --all

# manual alternative:
pip install -r requirements-voice.txt

# system deps
# Mac:
#   brew install ffmpeg
# Ubuntu:
#   sudo apt install ffmpeg portaudio19-dev
#   # portaudio helps sounddevice; ffmpeg is fallback capture
```

`install_whisper.sh` caches **openai-whisper** under `~/.cache/whisper`. With `--faster` it pulls **Systran/faster-whisper-*** from Hugging Face. With `--cpp` it clones **ggerganov/whisper.cpp** and fetches the matching `ggml-*.bin`.

## Quick test

```bash
# 1) Can we see Ollama on the Jetson?
python voice_to_ollama.py --host JETSON_IP:11434 --check-ollama

# 2) Mic only (no Ollama)
python voice_to_ollama.py --no-ollama --duration 5

# 3) Full path
python voice_to_ollama.py --host JETSON_IP:11434 --ollama-model llama3.2 --duration 5
```

Replace `JETSON_IP` with the board’s address (e.g. `192.168.137.51`).

On the Jetson itself:

```bash
python voice_to_ollama.py --host 127.0.0.1:11434 --duration 5
```

## Useful flags

| Flag | Meaning |
|------|---------|
| `--host` | Ollama `ip:port` (default `127.0.0.1:11434`) |
| `--ollama-model` | Model already pulled on Jetson |
| `--whisper-model` | `tiny.en` (default) / `base.en` |
| `--duration` | Seconds to record |
| `--audio file.wav` | Skip mic; use a file |
| `--list-devices` | Show microphones |
| `--no-ollama` | STT only |
| `--engine faster` | Use faster-whisper if installed |
| `--keep-audio out.wav` | Save the capture |

## Architecture

```
[Mac or Linux mic]
       │
       ▼
 local Whisper tiny.en   ← free, no API key
       │ text
       ▼
 Ollama on Jetson :11434
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Can’t reach Ollama | Same Wi‑Fi; `curl http://JETSON_IP:11434/api/tags`; `ollama serve` on Jetson |
| No mic on Linux | `sudo apt install portaudio19-dev` then reinstall `sounddevice`; or install `ffmpeg` |
| Empty transcript | Longer `--duration`, closer mic, try `base.en` |
| Wrong mic | `python voice_to_ollama.py --list-devices` then `--device N` |
| Jetson OOM | Keep STT on laptop; only Ollama on Jetson |

## Sandwich-oriented system prompt

Default system prompt asks Ollama for a short confirmation + JSON:

```json
{"ingredients": ["bread", "cheese", "turkey"], "hold": []}
```

Override with `--system "..."`.
