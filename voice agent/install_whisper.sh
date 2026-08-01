#!/usr/bin/env bash
# Prefetch local Whisper weights (no OpenAI subscription).
# Mac + Linux. Safe to re-run.
#
# Usage:
#   ./install_whisper.sh                  # tiny.en (default) via openai-whisper
#   ./install_whisper.sh base.en
#   ./install_whisper.sh tiny.en --faster  # also pull faster-whisper HF weights
#   ./install_whisper.sh tiny.en --cpp     # clone whisper.cpp + ggml model
#   ./install_whisper.sh tiny.en --all     # openai + faster + whisper.cpp
#   ./install_whisper.sh --pip-only        # just pip deps, no model download
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL="tiny.en"
DO_OPENAI=1
DO_FASTER=0
DO_CPP=0
DO_PIP=1
PIP_ONLY=0

HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

usage() {
  sed -n '2,14p' "$0" | sed 's/^# \?//'
  exit 0
}

log()  { printf '==> %s\n' "$*"; }
warn() { printf '!!  %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# --- parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage ;;
    --faster)  DO_FASTER=1; shift ;;
    --cpp)     DO_CPP=1; shift ;;
    --all)     DO_FASTER=1; DO_CPP=1; DO_OPENAI=1; shift ;;
    --no-openai) DO_OPENAI=0; shift ;;
    --pip-only) PIP_ONLY=1; shift ;;
    --no-pip)  DO_PIP=0; shift ;;
    -*)        die "unknown flag: $1" ;;
    *)         MODEL="$1"; shift ;;
  esac
done

# Map friendly names → HF / engine ids
# openai-whisper accepts: tiny, tiny.en, base, base.en, small, ...
case "$MODEL" in
  tiny|tiny.en|base|base.en|small|small.en|medium|medium.en|large|large-v2|large-v3) ;;
  *) warn "unusual model name '$MODEL' — continuing anyway" ;;
esac

# faster-whisper HF repos (Systran)
faster_repo_for() {
  case "$1" in
    tiny)      echo "Systran/faster-whisper-tiny" ;;
    tiny.en)   echo "Systran/faster-whisper-tiny.en" ;;
    base)      echo "Systran/faster-whisper-base" ;;
    base.en)   echo "Systran/faster-whisper-base.en" ;;
    small)     echo "Systran/faster-whisper-small" ;;
    small.en)  echo "Systran/faster-whisper-small.en" ;;
    medium)    echo "Systran/faster-whisper-medium" ;;
    medium.en) echo "Systran/faster-whisper-medium.en" ;;
    large|large-v2) echo "Systran/faster-whisper-large-v2" ;;
    large-v3)  echo "Systran/faster-whisper-large-v3" ;;
    *)         echo "Systran/faster-whisper-$1" ;;
  esac
}

# whisper.cpp ggml filenames
cpp_ggml_for() {
  case "$1" in
    tiny)      echo "ggml-tiny.bin" ;;
    tiny.en)   echo "ggml-tiny.en.bin" ;;
    base)      echo "ggml-base.bin" ;;
    base.en)   echo "ggml-base.en.bin" ;;
    small)     echo "ggml-small.bin" ;;
    small.en)  echo "ggml-small.en.bin" ;;
    medium)    echo "ggml-medium.bin" ;;
    medium.en) echo "ggml-medium.en.bin" ;;
    large|large-v2) echo "ggml-large-v2.bin" ;;
    large-v3)  echo "ggml-large-v3.bin" ;;
    *)         echo "ggml-$1.bin" ;;
  esac
}

have() { command -v "$1" >/dev/null 2>&1; }

pick_python() {
  if [[ -n "${VIRTUAL_ENV:-}" ]] && have python; then
    echo python
  elif have conda && conda run -n lerobot true 2>/dev/null; then
    echo "conda run -n lerobot python"
  elif have python3; then
    echo python3
  elif have python; then
    echo python
  else
    die "no python found"
  fi
}

PY="$(pick_python)"
log "Python: $PY"
log "Model:  $MODEL"

# --- pip deps ---
if [[ "$DO_PIP" -eq 1 ]]; then
  log "Installing Python deps (numpy, sounddevice, openai-whisper)…"
  # shellcheck disable=SC2086
  $PY -m pip install -q -U pip
  # shellcheck disable=SC2086
  $PY -m pip install -q -U "numpy>=1.24" "sounddevice>=0.4.6" "openai-whisper>=20231117"
  if [[ "$DO_FASTER" -eq 1 || "$PIP_ONLY" -eq 1 ]]; then
    # shellcheck disable=SC2086
    $PY -m pip install -q -U "faster-whisper>=1.0.0" "huggingface_hub>=0.20"
  fi
fi

if [[ "$PIP_ONLY" -eq 1 ]]; then
  log "pip-only done."
  exit 0
fi

# --- openai-whisper cache (downloads from OpenAI CDN on first load) ---
if [[ "$DO_OPENAI" -eq 1 ]]; then
  log "Prefetch openai-whisper model '$MODEL' (cached under ~/.cache/whisper)…"
  # shellcheck disable=SC2086
  $PY - <<PY
import whisper
print("loading:", "${MODEL}")
m = whisper.load_model("${MODEL}")
print("ok:", type(m).__name__)
PY
  log "openai-whisper ready."
fi

# --- faster-whisper from Hugging Face ---
if [[ "$DO_FASTER" -eq 1 ]]; then
  REPO="$(faster_repo_for "$MODEL")"
  DEST="${HF_HOME:-$HOME/.cache/huggingface}/hub"
  log "Prefetch faster-whisper from Hugging Face: $REPO"
  if have huggingface-cli; then
    huggingface-cli download "$REPO" --quiet || \
      huggingface-cli download "$REPO"
  else
    # shellcheck disable=SC2086
    $PY - <<PY
from huggingface_hub import snapshot_download
repo = "${REPO}"
print("snapshot_download:", repo)
path = snapshot_download(repo_id=repo)
print("cached at:", path)
PY
  fi
  # also force faster-whisper to load once (validates files)
  # shellcheck disable=SC2086
  $PY - <<PY
from faster_whisper import WhisperModel
name = "${MODEL}"
print("WhisperModel(", name, ", device=cpu, compute_type=int8)")
WhisperModel(name, device="cpu", compute_type="int8")
print("faster-whisper ready")
PY
  log "faster-whisper ready."
fi

# --- whisper.cpp clone + ggml from HF ---
if [[ "$DO_CPP" -eq 1 ]]; then
  CPP_DIR="${WHISPER_CPP_DIR:-$ROOT/third_party/whisper.cpp}"
  GGML="$(cpp_ggml_for "$MODEL")"
  # Official ggml models mirrored on Hugging Face
  HF_CPP_REPO="ggerganov/whisper.cpp"
  log "whisper.cpp → $CPP_DIR"

  if [[ ! -d "$CPP_DIR/.git" ]]; then
    mkdir -p "$(dirname "$CPP_DIR")"
    if have git; then
      git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git "$CPP_DIR"
    else
      die "git required to clone whisper.cpp"
    fi
  else
    log "repo already present (skip clone)"
  fi

  MODELS_DIR="$CPP_DIR/models"
  mkdir -p "$MODELS_DIR"
  OUT="$MODELS_DIR/$GGML"

  if [[ -f "$OUT" ]]; then
    log "already have $OUT"
  else
    # Prefer repo's download script if present; else curl HF
    if [[ -x "$MODELS_DIR/download-ggml-model.sh" ]]; then
      log "download-ggml-model.sh ${MODEL}"
      (cd "$CPP_DIR" && bash ./models/download-ggml-model.sh "${MODEL}")
    else
      URL="${HF_ENDPOINT}/${HF_CPP_REPO}/resolve/main/models/${GGML}"
      log "curl $URL"
      if have curl; then
        curl -fL --progress-bar -o "$OUT" "$URL" \
          || curl -fL --progress-bar -o "$OUT" \
               "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${GGML}"
      elif have wget; then
        wget -O "$OUT" "$URL"
      else
        die "need curl or wget to fetch ggml model"
      fi
    fi
  fi

  if [[ ! -f "$OUT" ]]; then
    # last resort: HF hub python
    # shellcheck disable=SC2086
    $PY - <<PY
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(repo_id="ggerganov/whisper.cpp", filename="models/${GGML}")
shutil.copy(path, "${OUT}")
print("copied to", "${OUT}")
PY
  fi

  [[ -f "$OUT" ]] || die "failed to get $GGML"
  log "ggml model: $OUT ($(du -h "$OUT" | awk '{print $1}'))"

  # optional build if cmake/make available
  if have cmake && [[ ! -x "$CPP_DIR/build/bin/whisper-cli" && ! -x "$CPP_DIR/main" ]]; then
    log "Building whisper.cpp (optional)…"
    cmake -B "$CPP_DIR/build" -S "$CPP_DIR" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$CPP_DIR/build" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 2)"
  else
    log "skip build (already built or cmake missing)"
  fi
fi

log "Done."
echo ""
echo "Try:"
echo "  python ${ROOT}/voice_to_ollama.py --no-ollama --duration 3 --whisper-model ${MODEL}"
echo "  python ${ROOT}/voice_to_ollama.py --host JETSON_IP:11434 --whisper-model ${MODEL}"
