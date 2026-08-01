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
Minimal JSON-based storage format for hand-recorded robot trajectories, used by
`lerobot-teleop-record` and `lerobot-playback`. This intentionally avoids the full
`LeRobotDataset` machinery (videos, HF hub, episode metadata, etc.) so a trajectory is
just a portable file with a list of timestamped actions.

File format:

```json
{
    "robot_type": "seeed_b601_rs_follower",
    "teleop_type": "rebot_arm_102_leader",
    "fps": 60,
    "joint_names": ["shoulder_pan", "shoulder_lift", ...],
    "recorded_at": "2026-08-01T18:45:00",
    "duration_s": 12.34,
    "num_frames": 740,
    "frames": [
        {"t": 0.0, "action": {"shoulder_pan.pos": 1.2, ...}},
        ...
    ]
}
```
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def save_trajectory(
    path: str | Path,
    joint_names: list[str],
    fps: int,
    robot_type: str,
    teleop_type: str | None,
    frames: list[dict[str, Any]],
) -> Path:
    """Saves a recorded trajectory (list of `{"t": float, "action": dict}` frames) to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "robot_type": robot_type,
        "teleop_type": teleop_type,
        "fps": fps,
        "joint_names": joint_names,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": frames[-1]["t"] if frames else 0.0,
        "num_frames": len(frames),
        "frames": frames,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return path


def load_trajectory(path: str | Path) -> dict[str, Any]:
    """Loads a trajectory JSON file previously saved with `save_trajectory`."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    if "frames" not in data or not isinstance(data["frames"], list):
        raise ValueError(f"'{path}' does not look like a valid trajectory file (missing 'frames' list).")

    return data


def list_trajectories(directory: str | Path) -> list[Path]:
    """Returns trajectory JSON files found directly inside `directory`, sorted by name."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))
