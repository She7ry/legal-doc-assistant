from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path


def run_eval() -> None:
    _run_script("run_rag_eval.py")


def generate_fixtures() -> None:
    _run_script("generate_eval_fixtures.py")


def _run_script(file_name: str) -> None:
    namespace = runpy.run_path(str(_script_path(file_name)))
    main = namespace.get("main")
    if not callable(main):
        raise RuntimeError(f"{file_name} does not define a callable main().")
    _call_main(main)


def _script_path(file_name: str) -> Path:
    project_root = Path(__file__).resolve().parents[3]
    script_path = project_root / "scripts" / file_name
    if not script_path.exists():
        raise RuntimeError(f"Could not find script entry point: {script_path}")
    return script_path


def _call_main(main: Callable[[], None]) -> None:
    main()
