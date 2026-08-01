#!/usr/bin/env python3
"""
Voice → local Whisper (free) → Ollama on Jetson.

Works on macOS and Linux. No OpenAI Whisper subscription/API key.

Examples:
  # Record 5s on Mac/Linux, tiny.en STT, send to Jetson Ollama
  python voice_to_ollama.py --host 192.168.7.165

  # Longer clip + larger STT model if tiny is too weak
  python voice_to_ollama.py --host 172.20.10.2 --duration 8 --whisper-model base.en

  # Transcribe an existing file (skip mic)
  python voice_to_ollama.py --host 192.168.7.165 --audio order.wav

  # Just print transcript (no Ollama)
  python voice_to_ollama.py --no-ollama --duration 4

  # List mics
  python voice_to_ollama.py --list-devices
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SAMPLE_RATE = 16000  # Whisper expects ~16 kHz mono


# ---------------------------------------------------------------------------
# Audio capture (Mac + Linux)
# ---------------------------------------------------------------------------

def list_input_devices() -> None:
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. pip install sounddevice numpy", file=sys.stderr)
        sys.exit(1)
    print(sd.query_devices())
    print(f"\nDefault input device index: {sd.default.device[0]}")


def record_with_sounddevice(path: Path, duration: float, device: Optional[int]) -> None:
    import numpy as np
    import sounddevice as sd

    channels = 1
    print(f"Recording {duration:.1f}s @ {SAMPLE_RATE} Hz (sounddevice)…", flush=True)
    print("  Speak now.", flush=True)
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=channels,
        dtype="float32",
        device=device,
    )
    sd.wait()
    # float32 [-1, 1] → int16 PCM wav
    pcm = np.clip(audio.reshape(-1), -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_i16.tobytes())
    print(f"  Saved {path} ({path.stat().st_size} bytes)", flush=True)


def record_with_ffmpeg(path: Path, duration: float) -> None:
    """Fallback if sounddevice fails. Uses OS-native ffmpeg capture."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "Neither usable sounddevice nor ffmpeg found. "
            "Install: pip install sounddevice numpy   and/or   brew/apt install ffmpeg"
        )

    system = platform.system()
    if system == "Darwin":
        # Default mic is usually ":0" (none:audio_device)
        input_args = ["-f", "avfoundation", "-i", ":0"]
    elif system == "Linux":
        # Pulse first (desktops); ALSA default as fallback via pulse's default
        input_args = ["-f", "pulse", "-i", "default"]
    else:
        raise RuntimeError(f"Unsupported OS for ffmpeg capture: {system}")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *input_args,
        "-t",
        str(duration),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        str(path),
    ]
    print(f"Recording {duration:.1f}s via ffmpeg ({system})…", flush=True)
    print("  Speak now.", flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        if system == "Linux":
            # Retry with ALSA
            cmd_alsa = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "alsa",
                "-i",
                "default",
                "-t",
                str(duration),
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                str(path),
            ]
            print("  Pulse failed; retrying with ALSA default…", flush=True)
            subprocess.run(cmd_alsa, check=True)
        else:
            raise e
    print(f"  Saved {path}", flush=True)


def record_audio(path: Path, duration: float, device: Optional[int]) -> None:
    try:
        import sounddevice  # noqa: F401
        import numpy  # noqa: F401

        record_with_sounddevice(path, duration, device)
        return
    except Exception as e:
        print(f"sounddevice path failed ({e}); trying ffmpeg…", flush=True)
    record_with_ffmpeg(path, duration)


# ---------------------------------------------------------------------------
# Local Whisper STT (no API key / subscription)
# ---------------------------------------------------------------------------

def transcribe_openai_whisper(audio_path: Path, model_name: str, language: str) -> str:
    import whisper

    print(f"Loading Whisper model '{model_name}' (local, free)…", flush=True)
    model = whisper.load_model(model_name)
    print("Transcribing…", flush=True)
    result = model.transcribe(
        str(audio_path),
        language=language or None,
        fp16=False,  # safer on CPU / mixed devices
    )
    text = (result.get("text") or "").strip()
    return text


def transcribe_faster_whisper(audio_path: Path, model_name: str, language: str) -> str:
    from faster_whisper import WhisperModel

    # tiny/base on CPU int8 is light; use cuda if available later
    print(f"Loading faster-whisper '{model_name}' (local, free)…", flush=True)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    print("Transcribing…", flush=True)
    segments, _info = model.transcribe(
        str(audio_path),
        language=language or None,
        beam_size=1,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text


def transcribe(audio_path: Path, model_name: str, language: str, engine: str) -> str:
    if engine == "auto":
        try:
            import faster_whisper  # noqa: F401

            engine = "faster"
        except ImportError:
            try:
                import whisper  # noqa: F401

                engine = "openai"
            except ImportError:
                print(
                    "No local Whisper installed.\n"
                    "  pip install -U openai-whisper\n"
                    "  # or lighter:\n"
                    "  pip install faster-whisper\n",
                    file=sys.stderr,
                )
                sys.exit(1)

    if engine == "faster":
        return transcribe_faster_whisper(audio_path, model_name, language)
    if engine == "openai":
        return transcribe_openai_whisper(audio_path, model_name, language)
    raise ValueError(f"Unknown engine: {engine}")


# ---------------------------------------------------------------------------
# Ollama (Jetson)
# ---------------------------------------------------------------------------

def ollama_generate(
    host: str,
    model: str,
    prompt: str,
    system: Optional[str] = None,
    timeout: float = 120.0,
) -> str:
    """Call Ollama /api/chat (preferred) with fallback to /api/generate."""
    base = host.rstrip("/")
    if not base.startswith("http"):
        base = f"http://{base}"

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    chat_body = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    gen_body = {
        "model": model,
        "prompt": (f"{system}\n\n{prompt}" if system else prompt),
        "stream": False,
    }

    for path, body, key in (
        ("/api/chat", chat_body, "message"),
        ("/api/generate", gen_body, "response"),
    ):
        url = f"{base}{path}"
        data = json.dumps(body).encode("utf-8")
        req = Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=timeout) as resp:
                payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
            if key == "message":
                msg = payload.get("message") or {}
                return (msg.get("content") or "").strip()
            return (payload.get("response") or "").strip()
        except HTTPError as e:
            if e.code == 404 and path == "/api/chat":
                continue
            raise RuntimeError(f"Ollama HTTP {e.code} at {url}: {e.read().decode()}") from e
        except URLError as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {url}\n"
                f"  Is the Jetson on the same Wi-Fi? Is ollama serve running?\n"
                f"  Try: curl {base}/api/tags\n"
                f"  Error: {e}"
            ) from e
    raise RuntimeError("Ollama request failed for both /api/chat and /api/generate")


def ollama_tags(host: str, timeout: float = 10.0) -> list[str]:
    base = host.rstrip("/")
    if not base.startswith("http"):
        base = f"http://{base}"
    url = f"{base}/api/tags"
    req = Request(url, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return [m.get("name", "") for m in payload.get("models", [])]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM = (
    "You are a kitchen robot assistant for a sandwich station. "
    "The user speaks an order. Reply with a short confirmation and a single JSON object "
    'on the last line: {"ingredients": ["bread", ...], "hold": []} '
    "Only use simple ingredients (bread, cheese, turkey, ham, tomato, lettuce, pickle). "
    "Always include bread unless they refuse bread."
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mic → local Whisper → Ollama on Jetson (Mac/Linux)"
    )
    p.add_argument(
        "--host",
        default="127.0.0.1:11434",
        help="Ollama host:port on Jetson (default 127.0.0.1:11434)",
    )
    p.add_argument(
        "--ollama-model",
        default="llama3.2",
        help="Ollama model name (must be pulled on Jetson)",
    )
    p.add_argument(
        "--whisper-model",
        default="tiny.en",
        help="Local Whisper model: tiny.en | base.en | tiny | base (default tiny.en)",
    )
    p.add_argument(
        "--engine",
        choices=("auto", "openai", "faster"),
        default="auto",
        help="STT backend: auto | openai (openai-whisper) | faster (faster-whisper)",
    )
    p.add_argument(
        "--language",
        default="en",
        help="Language code for Whisper (default en)",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Record duration in seconds (default 5)",
    )
    p.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Use existing audio file instead of recording",
    )
    p.add_argument(
        "--device",
        type=int,
        default=None,
        help="sounddevice input device index (see --list-devices)",
    )
    p.add_argument(
        "--prompt-prefix",
        default="Customer said:",
        help="Prefix before transcript sent to Ollama",
    )
    p.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help="System prompt for Ollama",
    )
    p.add_argument(
        "--no-ollama",
        action="store_true",
        help="Only transcribe; do not call Ollama",
    )
    p.add_argument(
        "--list-devices",
        action="store_true",
        help="List sounddevice input devices and exit",
    )
    p.add_argument(
        "--check-ollama",
        action="store_true",
        help="List models on Ollama host and exit",
    )
    p.add_argument(
        "--keep-audio",
        type=Path,
        default=None,
        help="Copy recorded audio to this path",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        list_input_devices()
        return 0

    if args.check_ollama:
        try:
            models = ollama_tags(args.host)
        except Exception as e:
            print(f"Failed: {e}", file=sys.stderr)
            return 1
        print(f"Ollama at {args.host}:")
        for m in models:
            print(f"  - {m}")
        if not models:
            print("  (no models — run: ollama pull llama3.2  on the Jetson)")
        return 0

    # --- audio ---
    tmp_dir = None
    if args.audio:
        audio_path = args.audio.expanduser().resolve()
        if not audio_path.is_file():
            print(f"Audio file not found: {audio_path}", file=sys.stderr)
            return 1
    else:
        tmp_dir = tempfile.TemporaryDirectory(prefix="voice_ollama_")
        audio_path = Path(tmp_dir.name) / "capture.wav"
        try:
            record_audio(audio_path, args.duration, args.device)
        except Exception as e:
            print(f"Recording failed: {e}", file=sys.stderr)
            return 1
        if args.keep_audio:
            dest = args.keep_audio.expanduser().resolve()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(audio_path.read_bytes())
            print(f"Kept audio at {dest}")

    # --- STT ---
    try:
        transcript = transcribe(
            audio_path,
            model_name=args.whisper_model,
            language=args.language,
            engine=args.engine,
        )
    except Exception as e:
        print(f"Transcription failed: {e}", file=sys.stderr)
        return 1

    print("\n=== Transcript ===")
    print(transcript if transcript else "(empty)")
    print("==================\n")

    if not transcript:
        print("Nothing transcribed — try longer --duration or a quieter room.", file=sys.stderr)
        return 2

    if args.no_ollama:
        return 0

    # --- Ollama ---
    user_prompt = f"{args.prompt_prefix}\n\n{transcript}"
    print(f"Sending to Ollama @ {args.host} model={args.ollama_model}…", flush=True)
    try:
        reply = ollama_generate(
            host=args.host,
            model=args.ollama_model,
            prompt=user_prompt,
            system=args.system,
        )
    except Exception as e:
        print(f"Ollama failed: {e}", file=sys.stderr)
        return 1

    print("=== Ollama ===")
    print(reply)
    print("==============")

    if tmp_dir is not None:
        tmp_dir.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
