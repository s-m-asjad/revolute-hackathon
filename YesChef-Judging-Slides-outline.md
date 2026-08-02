# YesChef — 3-slide judging deck outline

Source story: message.txt (Devpost). Rubric: Revolute 100 pts.

## Slide 1 — Product / Taste (30 pts)
**Title:** YesChef  
**Tagline:** "Hey chef, make me a sandwich" — then it actually does.

| Card | Criterion | Message |
|------|-----------|---------|
| Creativity | 10 | Canonical hard joke made real; NL + soft objects; order sequence preserved |
| Edibility | 10 | Real ingredients; custom containers; TPU compliant gripper |
| Chef's take | 10 | Guest-facing product + kill-switch ladder so the deli never dies |

## Slide 2 — Technical Difficulty (30 pts)
**Pipeline:** Wake/STT → Intent (Qwen) → Parse (text matcher) → Vision gate (Qwen-VL) → Trajectory stack

| Pillar | Pts | Proof points |
|--------|-----|----------------|
| Ambition | 10 | Full loop on Jetson+reBot; free models; compliant gripper |
| Accuracy | 10 | Sequence-preserving parse; label-aware vision; per-item trajectories |
| Consistency | 10 | Unwrap/wrist/gripper fixes; VRAM unload; teleop fallback |

## Slide 3 — Documentation (40 pts) + story close
| Pts | Deliverable | Asset |
|-----|-------------|--------|
| 10 | GitHub + README | repo / README / commands.txt / voice_agent / trajectories |
| 15 | Live demo | table script |
| 10 | Demo video | video/ + Devpost |
| 5 | PPT | this deck |

**Proud of:** E2E by deadline · clean layers · free stack · hardware honesty  
**Next:** dynamic localization · layer feedback · learned policies
