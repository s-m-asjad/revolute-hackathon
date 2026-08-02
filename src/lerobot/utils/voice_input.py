# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mic -> local Whisper transcription, used to turn a spoken instruction into the same free-text
`prompt` that `lerobot-play-by-prompt` would otherwise take from the command line.

This is intentionally separate from `ollama_client.py`: it never talks to an LLM, it just converts audio
to text. No API key or subscription is needed -- speech-to-text runs fully locally via `openai-whisper`
or `faster-whisper` (whichever is installed; install one with `pip install lerobot[voice]`).
"""

import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # Whisper expects ~16 kHz mono audio.


def _import_sounddevice():
    try:
        import sounddevice as sd

        return sd
    except ImportError as e:
        raise RuntimeError(
            "sounddevice is not installed. Install mic support with:\n"
            "  sudo apt install portaudio19-dev\n"
            "  pip install lerobot[voice]\n"
        ) from e


def list_input_devices() -> None:
    """Prints available microphones and the current default input device index."""
    sd = _import_sounddevice()

    print(sd.query_devices())
    print(f"\nDefault input device index: {sd.default.device[0]}")


def _resolve_input_device(sd, device: int | None) -> int | None:
    """Picks which input device to record from.

    If the caller didn't pin one down with `--voice_device`, don't just trust the OS/ALSA "default"
    device blindly: on this hardware it silently resolves to a virtual device (e.g. the Jetson's internal
    audio-processing-engine) with no real microphone attached, which records total silence with no error
    of any kind. Prefer an actual USB mic if one shows up in the device list.
    """
    if device is not None:
        return device

    try:
        devices = sd.query_devices()
    except Exception:
        return None

    usb_mics = [
        i
        for i, d in enumerate(devices)
        if d.get("max_input_channels", 0) > 0 and "usb" in d.get("name", "").lower()
    ]
    if not usb_mics:
        return None

    chosen = usb_mics[0]
    logger.info(
        "No --voice_device given; auto-selected input device %d (%r) since it looks like a USB mic -- "
        "the OS 'default' input device can silently resolve to something with no real microphone. "
        "Pass --voice_device=N to override (see --voice_list_devices=true).",
        chosen,
        devices[chosen]["name"],
    )
    return chosen


def _block_until_enter_or_timeout(max_duration: float) -> None:
    """Blocks until either Enter is pressed on stdin or `max_duration` seconds elapse, whichever first."""
    enter_pressed = threading.Event()

    def _wait_for_enter() -> None:
        try:
            input()
        except EOFError:
            pass
        enter_pressed.set()

    # Daemon thread: if we hit the max_duration timeout instead of Enter, this thread is simply abandoned
    # (it'll keep waiting on stdin, harmless since it's not read again before process exit).
    threading.Thread(target=_wait_for_enter, daemon=True).start()
    enter_pressed.wait(timeout=max_duration)


def _find_pulse_mic_source(keyword: str = "usb") -> str | None:
    """Returns the PulseAudio source name for a mic matching `keyword` (case-insensitive), skipping
    `.monitor` sources (those are loopbacks of an output, not a real mic). Returns `None` if PulseAudio
    isn't running or no match is found."""
    if not shutil.which("pactl"):
        return None
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sources"], capture_output=True, text=True, timeout=5
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        name = fields[1]
        if name.endswith(".monitor"):
            continue
        if keyword.lower() in name.lower():
            return name
    return None


def _record_with_arecord(path: Path, max_duration: float, source_name: str | None) -> bool:
    """Records via the ALSA `pulse` I/O plugin (i.e. through PulseAudio), optionally pinned to a specific
    source with the `PULSE_SOURCE` env var. Returns True on success (a non-empty WAV was written).

    This sidesteps a real gotcha seen on this hardware: PortAudio's ALSA backend opening the raw hardware
    device directly conflicts with PulseAudio already holding it exclusively, and/or ALSA's plain
    'default' device can silently route to the wrong (e.g. suspended) PulseAudio source -- both failure
    modes record total silence with no error of any kind, so this is preferred over `sounddevice`
    whenever `arecord` is available.
    """
    if not shutil.which("arecord"):
        return False

    cmd = ["arecord", "-D", "pulse", "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-t", "wav", str(path)]
    env = dict(os.environ)
    if source_name:
        env["PULSE_SOURCE"] = source_name

    try:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except OSError:
        return False

    _block_until_enter_or_timeout(max_duration)

    proc.send_signal(signal.SIGINT)  # arecord finalizes the WAV header cleanly on SIGINT.
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    return path.exists() and path.stat().st_size > 44  # more than just a bare WAV header


def _record_with_sounddevice(path: Path, device: int | None, max_duration: float) -> None:
    """Fallback recorder using PortAudio (via `sounddevice`) for platforms without `arecord`/PulseAudio
    (e.g. macOS), or when `--voice_device` pins an explicit PortAudio device index."""
    import numpy as np

    sd = _import_sounddevice()
    device = _resolve_input_device(sd, device)

    chunks: list[np.ndarray] = []

    def _callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
        if status:
            logger.debug("sounddevice status: %s", status)
        chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device, callback=_callback
    ):
        _block_until_enter_or_timeout(max_duration)

    audio = np.concatenate(chunks, axis=0).reshape(-1) if chunks else np.zeros(0, dtype="float32")
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_i16.tobytes())


def _print_recording_summary(path: Path) -> None:
    """Reports how long the recording is and its peak level, so a silent/broken capture is obvious
    immediately instead of only showing up later as an empty Whisper transcript."""
    import numpy as np

    with wave.open(str(path), "rb") as wf:
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        framerate = wf.getframerate() or SAMPLE_RATE

    samples = np.frombuffer(raw, dtype=np.int16) if raw else np.zeros(0, dtype=np.int16)
    peak = float(np.max(np.abs(samples))) / 32768.0 if len(samples) else 0.0
    print(f"  Recorded {n_frames / framerate:.1f}s (peak level: {peak:.3f}).", flush=True)
    if peak < 0.01:
        print(
            "  Warning: that recording looks silent or near-silent. The mic being 'on' (e.g. an LED "
            "lighting up) doesn't guarantee audio is actually being captured from it -- check "
            "`pactl list short sources` for the right PulseAudio source, or run with "
            "--voice_list_devices=true and pass the right index via --voice_device=N.",
            flush=True,
        )


def record_audio_until_enter(path: Path, device: int | None = None, max_duration: float = 60.0) -> None:
    """Records mono 16kHz audio from the mic until Enter is pressed (or `max_duration` seconds elapse as
    a safety cap), and saves it as a WAV file at `path`.

    Prefers recording through PulseAudio (`arecord -D pulse`) when available and no explicit
    `--voice_device` was given, auto-targeting a USB mic if one is present -- see `_record_with_arecord`
    for why this is more reliable than talking to PortAudio directly on Linux. Falls back to
    `sounddevice`/PortAudio otherwise.
    """
    print(
        f"Recording started -- say your instruction now, then press Enter to stop "
        f"(auto-stops after {max_duration:.0f}s).",
        flush=True,
    )

    if device is None and shutil.which("arecord"):
        source = _find_pulse_mic_source()
        if _record_with_arecord(path, max_duration, source):
            _print_recording_summary(path)
            return
        logger.warning(
            "Recording via arecord/PulseAudio failed or produced no audio; falling back to PortAudio "
            "(sounddevice)."
        )

    _record_with_sounddevice(path, device, max_duration)
    _print_recording_summary(path)


def _transcribe_faster_whisper(audio_path: Path, model_name: str, language: str) -> str:
    from faster_whisper import WhisperModel

    logger.info(f"Loading faster-whisper '{model_name}'...")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(str(audio_path), language=language or None, beam_size=1)
    return " ".join(seg.text.strip() for seg in segments).strip()


def _transcribe_openai_whisper(audio_path: Path, model_name: str, language: str) -> str:
    import whisper

    logger.info(f"Loading Whisper '{model_name}'...")
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path), language=language or None, fp16=False)
    return (result.get("text") or "").strip()


def transcribe(audio_path: Path, model_name: str = "tiny.en", language: str = "en", engine: str = "auto") -> str:
    """Transcribes a WAV file with a local Whisper engine. `engine="auto"` prefers faster-whisper (lighter)
    and falls back to openai-whisper, whichever is installed."""
    if engine == "auto":
        try:
            import faster_whisper  # noqa: F401

            engine = "faster"
        except ImportError:
            try:
                import whisper  # noqa: F401

                engine = "openai"
            except ImportError as e:
                raise RuntimeError(
                    "No local speech-to-text engine installed. Install one with:\n"
                    "  pip install lerobot[voice]   # openai-whisper\n"
                    "  # or the lighter:\n"
                    "  pip install faster-whisper\n"
                ) from e

    if engine == "faster":
        return _transcribe_faster_whisper(audio_path, model_name, language)
    if engine == "openai":
        return _transcribe_openai_whisper(audio_path, model_name, language)
    raise ValueError(f"Unknown speech-to-text engine: {engine!r}")


def extract_after_wake_word(text: str, wake_word: str) -> str | None:
    """Returns whatever comes after the last occurrence of `wake_word` in `text` (case-insensitive,
    whitespace-tolerant between the wake word's own words), stripped of leading punctuation/whitespace.

    Returns `None` if `wake_word` doesn't appear in `text` at all -- callers should treat that as "no
    valid command was spoken" rather than falling back to the full transcript, since Whisper sometimes
    mishears the wake word into something unrelated. Returns `""` (not `None`) if the wake word was heard
    but nothing followed it.
    """
    if not wake_word:
        return text

    pattern = r"\s+".join(re.escape(word) for word in wake_word.split())
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    if not matches:
        return None

    tail = text[matches[-1].end() :]
    return tail.strip(" \t\n,.:;-")


def record_and_transcribe(
    model_name: str = "tiny.en",
    language: str = "en",
    engine: str = "auto",
    device: int | None = None,
    keep_audio: Path | None = None,
    max_duration: float = 60.0,
) -> str:
    """Records from the mic until Enter is pressed (or `max_duration` elapses) and returns the
    transcribed text. Optionally saves the raw audio to `keep_audio` for debugging."""
    with tempfile.TemporaryDirectory(prefix="lerobot_voice_") as tmp_dir:
        audio_path = Path(tmp_dir) / "capture.wav"
        record_audio_until_enter(audio_path, device=device, max_duration=max_duration)

        if keep_audio is not None:
            keep_audio.parent.mkdir(parents=True, exist_ok=True)
            keep_audio.write_bytes(audio_path.read_bytes())

        print("Transcribing with local Whisper...", flush=True)
        return transcribe(audio_path, model_name=model_name, language=language, engine=engine)
