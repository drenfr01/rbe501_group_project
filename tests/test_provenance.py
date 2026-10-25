"""Run manifests: JSON round trip, script/argv capture, dirty-tree handling."""

import importlib.util
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rbe_501_so101 import provenance
from rbe_501_so101.config import PROJECT_ROOT, ProvenanceConfig
from rbe_501_so101.provenance import PackageVersion, RunManifest, require_clean_tree

SAG_CHECK_SCRIPT = PROJECT_ROOT / "scripts" / "sim2real" / "collect" / "sag_check.py"


def _git(repo: Path, *args: str) -> None:
    """Run a git command quietly inside `repo`."""
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A throwaway repository with one committed file."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "test")
    (tmp_path / "tracked.txt").write_text("v1\n")
    (tmp_path / "scripts").mkdir()
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_manifest_round_trips_through_json():
    manifest = RunManifest(
        script="scripts/x.py",
        argv=("--a", "1"),
        git_commit="abc",
        git_dirty=False,
        started_at=datetime(2026, 10, 25, 12, 0, tzinfo=UTC),
        finished_at=None,
        package_versions=(PackageVersion("mujoco", "3.0"),),
        robot_id="arm",
    )
    assert RunManifest.from_json(manifest.to_json()) == manifest


def test_capture_records_relative_script_and_exact_argv(git_repo):
    script = git_repo / "scripts" / "run.py"
    manifest = RunManifest.capture(script=script, argv=("--x", "2"), repo_root=git_repo)
    assert manifest.script == "scripts/run.py"
    assert manifest.argv == ("--x", "2")
    assert manifest.git_dirty is False
    assert len(manifest.git_commit) == 40
    assert manifest.finished_at is None


def test_dirty_tree_writes_patch_and_flags_manifest(git_repo, tmp_path_factory):
    (git_repo / "tracked.txt").write_text("v2\n")
    manifest = RunManifest.capture(script=git_repo / "scripts" / "run.py", argv=(), repo_root=git_repo)
    out = tmp_path_factory.mktemp("out")
    manifest.write(out, repo_root=git_repo)
    cfg = ProvenanceConfig()
    assert manifest.git_dirty is True
    assert "+v2" in (out / cfg.patch_filename).read_text()
    assert RunManifest.from_json((out / cfg.manifest_filename).read_text()) == manifest


def test_finalize_sets_finished_at_and_rewrites(git_repo, tmp_path_factory):
    manifest = RunManifest.capture(script=git_repo / "scripts" / "run.py", argv=(), repo_root=git_repo)
    out = tmp_path_factory.mktemp("out")
    manifest.write(out, repo_root=git_repo)
    done = manifest.finalize(out)
    assert done.finished_at is not None
    saved = RunManifest.from_json((out / ProvenanceConfig().manifest_filename).read_text())
    assert saved.finished_at == done.finished_at


def test_require_clean_tree_refuses_dirty_without_flag(git_repo):
    (git_repo / "tracked.txt").write_text("v2\n")
    with pytest.raises(SystemExit):
        require_clean_tree(allow_dirty=False, repo_root=git_repo)
    require_clean_tree(allow_dirty=True, repo_root=git_repo)


def _load_script(path: Path):
    """Import a script file as a module without running its `__main__` block."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_entry_point_exits_on_dirty_tree(monkeypatch):
    monkeypatch.setattr(provenance, "git_is_dirty", lambda repo_root=PROJECT_ROOT: True)
    sag_check = _load_script(SAG_CHECK_SCRIPT)
    with pytest.raises(SystemExit) as exc:
        sag_check.main(["run"])
    assert exc.value.code not in (0, None)
