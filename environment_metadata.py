"""Read-only generation environment endpoint; no workflow node is involved."""
from __future__ import annotations

import asyncio
import configparser
import inspect
import platform
import re
import subprocess
import time
try:
    import tomllib
except ImportError:  # ComfyUI also supports Python 3.10; Git facts still work.
    tomllib = None
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

_cached: dict | None = None
_cached_at = 0.0
_lock = asyncio.Lock()


def _repository_url(value: str) -> str | None:
    if value.startswith("git@") and ":" in value:
        host, path = value[4:].split(":", 1)
        value = f"https://{host}/{path}"
    try:
        parts = urlsplit(value)
        if parts.scheme not in {"https", "http", "ssh", "git"} or not parts.hostname:
            return None
        host = parts.hostname + (f":{parts.port}" if parts.port else "")
        return urlunsplit(("https" if parts.scheme in {"ssh", "git"} else parts.scheme, host,
                           parts.path.rstrip("/").removesuffix(".git"), "", ""))
    except ValueError:
        return None


def git_identity(root: Path) -> tuple[str | None, str | None]:
    """Read a checkout's full HEAD, including worktrees/packed refs, without Git."""
    try:
        git_dir = root / ".git"
        if git_dir.is_file():
            marker = git_dir.read_text().strip()
            if not marker.startswith("gitdir: "):
                return None, None
            git_dir = (root / marker[8:]).resolve()
        common = git_dir
        if (git_dir / "commondir").is_file():
            common = (git_dir / (git_dir / "commondir").read_text().strip()).resolve()
        head = (git_dir / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            if not ref.startswith("refs/") or ".." in Path(ref).parts:
                return None, None
            loose = common / ref
            if loose.is_file():
                head = loose.read_text().strip()
            else:
                head = next((line.split(" ", 1)[0] for line in (common / "packed-refs").read_text().splitlines()
                             if line.endswith(" " + ref)), "")
        commit = head.lower() if re.fullmatch(r"[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?", head) else None
        config = configparser.RawConfigParser()
        config.read(common / "config")
        repo = _repository_url(config.get('remote "origin"', "url", fallback=""))
        return repo, commit
    except (OSError, ValueError, configparser.Error):
        return None, None


def collect_environment() -> dict:
    import folder_paths
    import nodes
    import torch

    root = Path(folder_paths.base_path)
    _, comfy_commit = git_identity(root)
    packs = []
    pack_roots = []
    for custom_root in folder_paths.get_folder_paths("custom_nodes"):
        try:
            entries = sorted(Path(custom_root).iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith(".") or entry.name == "__pycache__":
                continue
            repo, commit = git_identity(entry)
            registry_id = version = None
            try:
                project = tomllib.loads((entry / "pyproject.toml").read_text(encoding="utf-8")).get("project", {}) if tomllib else {}
                if (entry / ".tracking").is_file():
                    registry_id, version = project.get("name"), project.get("version")
            except (OSError, ValueError):
                pass
            pack_roots.append(entry.resolve())
            packs.append({"repo": repo, "commit": commit, "registry_id": registry_id, "version": version,
                          "module": "custom_nodes." + entry.name, "disabled": entry.name.endswith(".disabled"),
                          "node_classes": []})
    core_classes = []
    for class_name, node_class in nodes.NODE_CLASS_MAPPINGS.items():
        try:
            source = Path(inspect.getfile(node_class)).resolve()
        except (TypeError, OSError):
            continue
        matches = [i for i, pack_root in enumerate(pack_roots) if source.is_relative_to(pack_root)]
        if len(matches) == 1 and not packs[matches[0]]["disabled"]:
            packs[matches[0]]["node_classes"].append(class_name)
        elif not matches and source.is_relative_to(root.resolve()) and not any(source.is_relative_to(Path(p).resolve()) for p in folder_paths.get_folder_paths("custom_nodes")):
            core_classes.append(class_name)
    devices = []
    drivers = {}
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=index,driver_version", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=1, check=True).stdout
        drivers = {int(index.strip()): driver.strip() for index, driver in (line.split(",", 1) for line in output.splitlines())}
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({"name": properties.name, "type": "cuda", "index": index,
                            "vram_total": properties.total_memory, "driver": drivers.get(index)})
    else:
        devices.append({"name": platform.processor() or "CPU", "type": "cpu", "index": None,
                        "vram_total": None, "driver": None})
    cudnn = torch.backends.cudnn.version()
    return {"version": 1, "observed_at": datetime.now(timezone.utc).isoformat(),
            "sources": ["sweet_tea_environment"], "comfyui": {"version": None, "commit": comfy_commit},
            "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "cuda": torch.version.cuda, "cudnn": str(cudnn) if cudnn is not None else None},
            "devices": devices, "custom_node_packs": packs, "core_node_classes": core_classes,
            "unresolved_node_classes": []}


def register_environment_route() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return
    if PromptServer.instance is None:
        return

    @PromptServer.instance.routes.get("/sweet-tea/environment")
    async def environment_metadata(_request):
        global _cached, _cached_at
        async with _lock:
            if _cached is None or time.monotonic() - _cached_at >= 30:
                _cached = await asyncio.to_thread(collect_environment)
                _cached_at = time.monotonic()
        return web.json_response(_cached)
