# CLIP ManimGL Scenes

This repository contains ManimGL scenes for a CLIP-focused explainer project.
It includes both older scene files and a newer one-scene-per-file refinement workflow,
plus a lightweight verification layer for catching layout and scene-construction issues
before a full render.

## What Newcomers Should Know First

- This repo is in the middle of a file-naming migration.
- Older files such as `clip_encoding_scene_1.py` through `clip_encoding_scene_6.py`
  may still appear on GitHub.
- Newer work is moving toward descriptive filenames like
  `clip_shared_embedding_space_sc1.py`,
  `clip_similarity_matrix_sc2.py`,
  `mmd_distribution_difference_scene.py`, and
  `silhouette_weak_separation_scene.py`.
- If both styles exist locally, prefer the descriptive one-scene-per-file layout.

## Repo Map

- `clip_encoding.py`
  Legacy end-to-end CLIP overview scene.
- `clip_encoding_scene_*.py`
  Older split-scene files still present during migration.
- `mmd_distribution_difference_scene.py`
  Example of a refined distribution-metric scene with shared helpers.
- `silhouette_weak_separation_scene.py`
  Refined silhouette-score scene with explicit aggregation and verifier checks.
- `distribution_metrics_shared.py`
  Shared colors, scatter generation, and metric helpers for the distribution scenes.
- `layout_verifier.py`
  Geometry-based scene verification helpers.
- `verify_scenes.py`
  Fast verifier that constructs scenes without doing a full movie render.
- `scene_description.txt`
  The intended beat-by-beat plan for each scene.
- `blogpost_content.txt`
  Narrative and research context for the animation.
- `scene_refinement_playbook.md`
  Living guidance for scene structure, verification, and refinement lessons.
- `custom_config.yml`
  Output configuration. Rendered assets go under `videos/`.

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. If `manimgl==1.7.2` gives import/runtime issues on your machine, especially on
   Python 3.13, install ManimGL from a local source checkout instead:

```bash
pip uninstall -y manimgl
pip install /path/to/manimgl
```

Notes:

- `requirements.txt` already pins `setuptools<81` and `ipython<9` for current
  ManimGL compatibility.
- `audioop-lts` is included for Python 3.13 compatibility.

## Running Scenes

Use the project virtualenv explicitly so commands do not depend on your global PATH.

Interactive preview with autoreload:

```bash
./.venv/bin/manimgl --config_file custom_config.yml clip_encoding.py CLIPEncoding -i --autoreload
```

Render a movie:

```bash
./.venv/bin/manimgl --config_file custom_config.yml mmd_distribution_difference_scene.py MMDDistributionDifferenceScene -w --hd
```

Save a last frame for quick visual inspection:

```bash
./.venv/bin/manimgl --config_file custom_config.yml silhouette_weak_separation_scene.py SilhouetteWeakSeparationScene -s
```

By default, outputs land in `videos/` because of `custom_config.yml`.

## Verification Workflow

The recommended workflow for most scene edits is:

1. Run the fast verifier.
2. Fix any import, construction, or layout issues.
3. Render a last frame or full movie for human inspection.

Fast verifier example:

```bash
./.venv/bin/python verify_scenes.py silhouette_weak_separation_scene.py SilhouetteWeakSeparationScene
```

What `verify_scenes.py` does:

- imports the target file directly
- finds `Scene` subclasses
- runs them with `skip_animations=True`
- disables `self.embed()`
- surfaces exceptions without requiring a full render

What the layout verifier adds:

- overlap checks
- minimum vertical gap checks
- frame-bound checks
- scene-specific chart or annotation checks when used

Important limitation:

- verification is a fast guardrail, not a substitute for visual review
- a scene can pass geometry checks and still feel confusing or editorially wrong

## Suggested Commands

Verify every scene class in a file:

```bash
./.venv/bin/python verify_scenes.py mmd_distribution_difference_scene.py
```

Render a refined scene in HD:

```bash
./.venv/bin/manimgl --config_file custom_config.yml silhouette_weak_separation_scene.py SilhouetteWeakSeparationScene -w --hd
```

## How To Work On A New Or Refined Scene

Start here:

1. Read `scene_description.txt` for the intended visual beats.
2. Read `blogpost_content.txt` for the narrative claim.
3. Read `scene_refinement_playbook.md` for structure and style guidance.

Then follow this default loop:

1. Block the scene layout with stable anchors.
2. Add the minimum motion needed to explain the idea.
3. Add `LayoutVerifier` checks for text-heavy or annotation-heavy sections.
4. Run `verify_scenes.py`.
5. Inspect a last frame or full render.

## Design Conventions

Across the refined files, the project generally prefers:

- one clear claim per scene
- stable semantic color meaning
- deterministic toy data instead of heavy data pipelines
- transforms that preserve visual continuity
- final frames that still make sense as screenshots

## Current Best References

If you want examples of the current refinement direction, start with:

- `clip_shared_embedding_space_sc1.py`
- `clip_similarity_matrix_sc2.py`
- `umap_visualization_scene_sc4.py`
- `mlp_decision_boundary_scene_sc5.py`
- `mmd_distribution_difference_scene.py`
- `silhouette_weak_separation_scene.py`

The playbook explains why these are useful references and what patterns to reuse.
