# Pixal3DMD

DMD2 distillation of Pixal3D (geometry only for now; texture out of scope).
Helios is the reference recipe, but its direct application is non-trivial.
Succeeds the vecset-based `3D_dmd` (Hunyuan3D-2.1) line.

- Teacher codebase: `/gs/fs/tga-koike-shanda2/sk/Pixal3D` (TRELLIS.2 backbone + ProjectAttention)
- Discussions live in `docs/design-log/` (one dated file per session, append-only).

## Status

Method decided: DMD2, following Helios's DiT-based implementation (`3D_dmd`/M0 line).
Issue-mapping phase — see `docs/design-log/001-goal.md` for the goal and the open-issue map
(representation gap vs vecset, cascade strategy, pixel-align, teacher choice, +8 more).
