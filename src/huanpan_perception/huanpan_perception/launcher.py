"""Make the perception entry points work under ``ros2 run`` and ``launch``.

Torch and Ultralytics are installed in the workspace virtualenv rather than the
system interpreter, while ``ros2 run`` and ``launch_ros`` start nodes with the
system interpreter. Instead of asking the user to remember a special python
command, each entry point calls :func:`ensure_runtime`, which re-executes the
process with the virtualenv interpreter when the heavy dependency is missing.

The re-exec is cheap, idempotent and skipped when the dependency is already
importable (or when the interpreter cannot be found, in which case the node
continues and reports the real import error).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REEXEC_FLAG = "HUANPAN_PERCEPTION_REEXEC"


def find_perception_python() -> Path | None:
    """Locate the virtualenv interpreter that has torch + ultralytics."""
    explicit = os.environ.get("HUANPAN_PERCEPTION_PYTHON")
    if explicit and Path(explicit).exists():
        return Path(explicit)

    workspace = os.environ.get("HUANPAN_WS_ROOT")
    if workspace:
        candidate = Path(workspace) / ".venv-perception" / "bin" / "python"
        if candidate.exists():
            return candidate

    # Walk up from this file looking for the workspace virtualenv. This covers
    # both the source tree and install/ with --symlink-install.
    here = Path(__file__).resolve()
    for parent in list(here.parents)[:8]:
        candidate = parent / ".venv-perception" / "bin" / "python"
        if candidate.exists():
            return candidate

    home_candidate = Path.home() / "huanpan_ws" / ".venv-perception" / "bin" / "python"
    if home_candidate.exists():
        return home_candidate
    return None


def configure_environment() -> None:
    """Keep Ultralytics settings inside the workspace.

    Without this, Ultralytics falls back to ``/tmp/Ultralytics`` (and prints a
    warning) because the sandbox does not allow writing ``~/.config``.
    """
    if os.environ.get("YOLO_CONFIG_DIR"):
        return
    interpreter = find_perception_python()
    if interpreter is None:
        return
    workspace = Path(interpreter).parent.parent.parent
    config_dir = workspace / "outputs" / "perception" / ".ultralytics"
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)


def ensure_runtime(module: str, argv: list[str] | None = None) -> None:
    """Re-exec ``module`` under the perception virtualenv if needed."""
    configure_environment()
    if os.environ.get(REEXEC_FLAG) == "1":
        return
    try:
        import ultralytics  # noqa: F401
        return
    except ImportError:
        pass

    interpreter = find_perception_python()
    if interpreter is None:
        return
    # A venv's ``bin/python`` is a symlink to the system interpreter, so
    # resolving paths (or calling ``samefile``) cannot tell the two apart.
    # Compare the unresolved paths instead.
    if os.path.abspath(sys.executable) == os.path.abspath(str(interpreter)):
        return
    if os.path.abspath(sys.prefix) == os.path.abspath(
        str(Path(interpreter).parent.parent)
    ):
        return

    args = list(sys.argv[1:] if argv is None else argv)
    environment = dict(os.environ)
    environment[REEXEC_FLAG] = "1"
    os.execve(
        str(interpreter),
        [str(interpreter), "-m", module, *args],
        environment,
    )
