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

"""
Thin client for a local Ollama server, used to (1) decide whether a free-text instruction (e.g. "make a
sandwich") matches a known task, and (2) look at a photo (via a small Qwen vision-language model) to
describe a scene or check whether specific ingredients/objects are visible.

Why Ollama: it ships a self-contained, GPU-accelerated (CUDA) runtime for aarch64/Jetson that doesn't
depend on the system's `torch`/`transformers` install, so it works even when those are CPU-only (as they
are on this machine). Small models (a few billion parameters) are plenty for both the text intent-match
and the vision ingredient-check use cases here.

Note on prompting: asking Ollama for strict grammar-constrained JSON (`format: "json"`) tends to make
small (1-3B) models degenerate to trivial/empty answers on this kind of reasoning task. Instead we ask the
model to think briefly in free text and end with a `FINAL_JSON: {...}` marker line, then regex out the
JSON -- this reliably keeps the model's reasoning quality while still being trivial to parse.
"""

import base64
import io
import json
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import requests
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

logger = logging.getLogger(__name__)

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:3b"
# Smallest Qwen vision-language model on Ollama (~3.2GB) -- enough to describe a tabletop scene and spot
# whether a handful of named ingredients/objects are present.
DEFAULT_VLM_MODEL = "qwen2.5vl:3b"

_FINAL_JSON_RE = re.compile(r"FINAL_JSON:\s*(\{.*\})", re.DOTALL)
_ANY_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _ollama_binary() -> str | None:
    return shutil.which("ollama") or next(
        (p for p in [str(Path.home() / ".local" / "bin" / "ollama")] if Path(p).exists()), None
    )


def is_server_running(host: str = DEFAULT_HOST, timeout: float = 1.0) -> bool:
    try:
        resp = requests.get(f"{host}/api/version", timeout=timeout)
        return resp.ok
    except requests.RequestException:
        return False


def ensure_server_running(host: str = DEFAULT_HOST, startup_timeout: float = 20.0) -> None:
    """Makes sure an Ollama server is reachable at `host`, starting one in the background if needed."""
    if is_server_running(host):
        return

    binary = _ollama_binary()
    if binary is None:
        raise RuntimeError(
            "Ollama is not installed (or not on PATH). Install it from https://ollama.com/download "
            "and make sure the `ollama` binary is reachable, then try again."
        )

    logger.info("No Ollama server reachable at %s, starting one with `%s serve`...", host, binary)
    log_path = Path("/tmp/ollama-serve.log")
    with open(log_path, "ab") as log_file:
        subprocess.Popen(
            [binary, "serve"],
            stdout=log_file,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if is_server_running(host):
            return
        time.sleep(0.5)

    raise RuntimeError(
        f"Started `{binary} serve` but it never became reachable at {host}. Check {log_path} for errors."
    )


def has_model(model: str, host: str = DEFAULT_HOST) -> bool:
    resp = requests.get(f"{host}/api/tags", timeout=5)
    resp.raise_for_status()
    names = {m["name"] for m in resp.json().get("models", [])}
    # Ollama normalizes untagged names to ":latest", accept either form.
    return model in names or f"{model}:latest" in names


def ensure_model_available(model: str, host: str = DEFAULT_HOST) -> None:
    """Pulls `model` if it isn't already present locally, printing progress as it downloads."""
    if has_model(model, host):
        return

    print(f"Model '{model}' isn't downloaded yet, pulling it now (this can take a few minutes)...")
    with requests.post(f"{host}/api/pull", json={"model": model, "stream": True}, stream=True) as resp:
        resp.raise_for_status()
        last_status = None
        for line in resp.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            status = event.get("status")
            if status and status != last_status:
                print(f"  {status}")
                last_status = status
            if event.get("error"):
                raise RuntimeError(f"Failed to pull '{model}': {event['error']}")
    print(f"Model '{model}' ready.")


def unload_model(model: str, host: str = DEFAULT_HOST) -> None:
    """Immediately unloads `model` from the Ollama server's memory instead of waiting for its
    `keep_alive` timeout (default 5 minutes).

    Useful on memory-constrained hardware (e.g. a Jetson's shared CPU/GPU memory pool) where loading a
    second model (e.g. the vision model) while a previous one (e.g. the text classifier) is still resident
    can exhaust available memory -- observed as `cudaMalloc failed: out of memory` in Ollama's log and a
    500 from `/api/chat`, even though neither model is large on its own.
    """
    try:
        requests.post(f"{host}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
    except requests.RequestException as e:
        logger.warning("Failed to unload model %r (continuing anyway): %s", model, e)


def _extract_json(content: str) -> dict:
    match = _FINAL_JSON_RE.search(content)
    if match:
        return json.loads(match.group(1))
    match = _ANY_JSON_OBJECT_RE.search(content)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not find a JSON answer in the model's response:\n{content}")


def classify_match(
    prompt: str,
    task_description: str,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    temperature: float = 0.0,
) -> tuple[bool, str]:
    """Asks the local LLM whether `prompt` is a request to do `task_description`.

    Returns a tuple of (matched, raw_model_output).
    """
    system_prompt = (
        "You decide whether a user's instruction is asking a robot to perform ONE specific task. "
        "Paraphrases, different phrasing, or extra detail still count as a match. Unrelated requests, "
        "other tasks, or requests to do nothing should NOT match. "
        'Think briefly, then on the FINAL line output exactly: FINAL_JSON: {"match": true} or '
        'FINAL_JSON: {"match": false}.'
    )
    user_content = f'The task is: "{task_description}"\n\nUser instruction: "{prompt}"'

    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            # Small context window: this prompt is short, and keeping it small reduces the model's
            # memory footprint (relevant on memory-constrained hardware, see `unload_model`).
            "options": {"temperature": temperature, "num_ctx": 1024},
        },
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]

    parsed = _extract_json(content)
    match = parsed.get("match")
    if not isinstance(match, bool):
        raise ValueError(f"Model returned a non-boolean 'match' field: {match!r}")

    return match, content


def _image_to_base64(image: NDArray[Any], fmt: str = "JPEG") -> str:
    """Encodes an HxWx3 RGB numpy array as a base64 string, the format Ollama's API expects for images."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def describe_scene(
    image: NDArray[Any],
    question: str = (
        "Describe what you see in this photo of a table, listing every object/item you can identify."
    ),
    model: str = DEFAULT_VLM_MODEL,
    host: str = DEFAULT_HOST,
    temperature: float = 0.0,
) -> str:
    """Asks the local vision-language model to describe an image. Returns the raw text answer."""
    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": question, "images": [_image_to_base64(image)]},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def check_ingredients(
    image: NDArray[Any],
    ingredients: list[str],
    model: str = DEFAULT_VLM_MODEL,
    host: str = DEFAULT_HOST,
    temperature: float = 0.0,
) -> tuple[dict[str, list[str]], str]:
    """Asks the local VLM which of `ingredients` are visible on the table in `image`.

    An item only counts as "present" if the model can read its name as literal TEXT somewhere in the
    image (a label, packaging, a sign/card, etc.), not merely by visually recognizing a matching
    food/object -- this avoids the false positives a 3B vision model tends to produce when asked to
    visually identify food items.

    Returns a tuple of ({"present": [...], "missing": [...]}, raw_model_output), where both lists contain
    strings taken verbatim from `ingredients`.
    """
    ingredients_list = ", ".join(f'"{i}"' for i in ingredients)
    system_prompt = (
        "You are a robot's vision system checking a table before starting a task. You will be shown a "
        "photo and a list of required items.\n\n"
        "IMPORTANT: only count a required item as present if you can actually READ its name written as "
        "TEXT somewhere in the image -- e.g. printed on packaging/a label, on a sticker, on a handwritten "
        "or printed sign/card, etc. Do NOT count an item as present just because you see food or an "
        "object that visually looks like it -- you must be able to read the literal word (a close "
        "spelling variant or an unambiguous abbreviation is fine). If you cannot read text naming an "
        "item anywhere in the image, it counts as missing, even if something resembling it is visible.\n\n"
        'Think briefly about what text you can read, then on the FINAL line output exactly: '
        'FINAL_JSON: {"present": [...], "missing": [...]} where both are lists of strings taken '
        "verbatim from the required items list (every required item must appear in exactly one of the "
        "two lists)."
    )
    user_content = (
        f"Required items: [{ingredients_list}]\n\n"
        "Read any text/labels visible in the image. Which of the required items did you actually see "
        "written down, and which are missing?"
    )

    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content, "images": [_image_to_base64(image)]},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]

    parsed = _extract_json(content)
    present = parsed.get("present")
    missing = parsed.get("missing")
    if not isinstance(present, list) or not isinstance(missing, list):
        raise ValueError(f"Model returned malformed 'present'/'missing' fields:\n{content}")

    return {"present": [str(x) for x in present], "missing": [str(x) for x in missing]}, content


def analyze_table(
    image: NDArray[Any],
    ingredients: list[str],
    model: str = DEFAULT_VLM_MODEL,
    host: str = DEFAULT_HOST,
    temperature: float = 0.0,
    max_tokens: int = 200,
) -> tuple[dict[str, Any], str]:
    """One-shot combo of `describe_scene` + `check_ingredients`: describes the table AND reports which of
    `ingredients` are visible, in a single Ollama call.

    An item only counts as "present" if the model can read its name as literal TEXT somewhere in the
    image (a label, packaging, a sign/card, etc.), not merely by visually recognizing a matching
    food/object. This trades recall for precision -- it won't flag an unlabeled tomato as "tomato" -- but
    avoids the false positives a 3B vision model tends to produce when asked to visually identify food
    items (e.g. mistaking a similar-looking item, or guessing generously).

    Why this matters for speed: on compute-constrained hardware (e.g. a Jetson's GPU), each vision call
    pays a large ~fixed cost just to encode the image into vision tokens (observed ~7s on a Jetson Orin,
    independent of image resolution or how much text gets generated). Calling `describe_scene` then
    `check_ingredients` separately pays that cost twice; asking for both in one prompt pays it once,
    roughly halving wall-clock latency. Explicitly asking for a single concise line (rather than letting
    the model "think out loud") also cuts generated tokens substantially (measured ~2-4x fewer tokens),
    which matters because token generation is the other slow part on limited hardware. `max_tokens` (via
    Ollama's `num_predict`) caps worst-case generation time if the model ignores the brevity instruction.

    Returns a tuple of ({"description": str, "present": [...], "missing": [...]}, raw_model_output).
    """
    ingredients_list = ", ".join(f'"{i}"' for i in ingredients)
    system_prompt = (
        "You are a robot's vision system checking a table before starting a task. Be concise: do not "
        "narrate your reasoning step by step, just answer directly.\n\n"
        "IMPORTANT: only count a required item as present if you can actually READ its name written as "
        "TEXT somewhere in the image -- e.g. printed on packaging/a label, on a sticker, on a handwritten "
        "or printed sign/card, etc. Do NOT count an item as present just because you see food or an "
        "object that visually looks like it -- you must be able to read the literal word (a close "
        "spelling variant or an unambiguous abbreviation is fine, e.g. 'tom.' for 'tomato'). If you "
        "cannot read text naming an item anywhere in the image, it counts as missing, even if something "
        "resembling it is visible.\n\n"
        'Respond with ONLY a single line: FINAL_JSON: {"description": "<one short sentence describing '
        'the table, mentioning any text/labels you read>", "present": [...], "missing": [...]}, where '
        "present/missing are lists of strings taken verbatim from the required items list (every "
        "required item must appear in exactly one of the two lists)."
    )
    user_content = (
        f"Required items: [{ingredients_list}]\n\n"
        "Read any text/labels visible in the image, briefly describe the table, and say which required "
        "items you actually saw written down vs not."
    )

    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content, "images": [_image_to_base64(image)]},
            ],
            "stream": False,
            # Small-ish context window: the encoded image + prompt is ~1200 tokens and we cap generation
            # to `max_tokens`, so 2048 leaves headroom while using much less memory than Ollama's default
            # 4096 (relevant on memory-constrained hardware, see `unload_model`).
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 2048},
        },
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]

    parsed = _extract_json(content)
    description = parsed.get("description")
    present = parsed.get("present")
    missing = parsed.get("missing")
    if not isinstance(description, str) or not isinstance(present, list) or not isinstance(missing, list):
        raise ValueError(f"Model returned malformed description/present/missing fields:\n{content}")

    result = {
        "description": description,
        "present": [str(x) for x in present],
        "missing": [str(x) for x in missing],
    }
    return result, content
