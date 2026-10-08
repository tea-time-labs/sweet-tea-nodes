"""Portable checkout identity supports loose/packed refs and linked worktrees."""
import importlib.util
import subprocess
from pathlib import Path

spec = importlib.util.spec_from_file_location("environment_metadata", Path(__file__).parents[1] / "environment_metadata.py")
environment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(environment)

def test_checkout_identity_has_full_commit_and_credential_free_repo(tmp_path):
    root = tmp_path / "pack"
    root.mkdir()
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-b", "main")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("remote", "add", "origin", "https://user:password@github.com/example/pack.git?token=secret")
    (root / "node.py").write_text("class Example: pass\n")
    git("add", ".")
    git("commit", "-m", "fixture")
    commit = git("rev-parse", "HEAD")
    assert environment.git_identity(root) == ("https://github.com/example/pack", commit)
    git("pack-refs", "--all")
    assert environment.git_identity(root)[1] == commit
    worktree = tmp_path / "renamed-pack"
    git("worktree", "add", "--detach", str(worktree), "HEAD")
    assert environment.git_identity(worktree) == ("https://github.com/example/pack", commit)
    assert environment.git_identity(tmp_path / "missing") == (None, None)
