#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
import traceback
from pathlib import Path

from manimlib.scene.scene import Scene


def load_module_from_file(file_path: Path):
    module_name = file_path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_scene_classes(module):
    return [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, Scene)
        and obj is not Scene
        and obj.__module__.startswith(module.__name__)
    ]


def verify_scene(scene_class):
    original_embed = Scene.embed
    Scene.embed = lambda self, *args, **kwargs: None
    try:
        scene = scene_class(
            skip_animations=True,
            preview_while_skipping=False,
            show_animation_progress=False,
            leave_progress_bars=False,
            file_writer_config={
                "write_to_movie": False,
                "save_last_frame": False,
                "quiet": True,
            },
        )
        scene.run()
    finally:
        Scene.embed = original_embed


def main():
    parser = argparse.ArgumentParser(
        description="Fast scene verification without full rendering."
    )
    parser.add_argument("file", help="Path to the Python scene file")
    parser.add_argument(
        "scene_names",
        nargs="*",
        help="Optional scene class names to verify. Defaults to all scenes in the file.",
    )
    args = parser.parse_args()

    file_path = Path(args.file).resolve()
    sys.path.insert(0, str(file_path.parent))

    module = load_module_from_file(file_path)
    scene_classes = get_scene_classes(module)
    if args.scene_names:
        wanted = set(args.scene_names)
        scene_classes = [scene for scene in scene_classes if scene.__name__ in wanted]

    if not scene_classes:
        raise SystemExit("No matching scene classes found.")

    failures = []
    for scene_class in scene_classes:
        print(f"[verify] {scene_class.__name__}")
        try:
            verify_scene(scene_class)
        except Exception as exc:
            failures.append((scene_class.__name__, exc, traceback.format_exc()))
            print(f"[fail] {scene_class.__name__}: {exc}")
        else:
            print(f"[ok] {scene_class.__name__}")

    if failures:
        print("\nVerification failures:")
        for name, _, tb in failures:
            print(f"\n--- {name} ---")
            print(tb)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
