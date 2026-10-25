"""Run manifests: which script, arguments, code version, and environment made an output folder."""

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Self

from rbe_501_so101.config import EXPERIMENTS_DIR, PROJECT_ROOT, ProvenanceConfig

NOT_INSTALLED = "not installed"
UNKNOWN_COMMIT = "unknown"


@dataclass(frozen=True)
class PackageVersion:
    """One installed package and its version."""

    name: str
    version: str


@dataclass(frozen=True)
class RunManifest:
    """Where one output folder came from: script, arguments, code version, and environment."""

    script: str
    argv: tuple[str, ...]
    git_commit: str
    git_dirty: bool
    started_at: datetime
    finished_at: datetime | None
    package_versions: tuple[PackageVersion, ...]
    robot_id: str | None

    @classmethod
    def capture(
        cls,
        robot_id: str | None = None,
        *,
        script: Path | None = None,
        argv: tuple[str, ...] | None = None,
        repo_root: Path = PROJECT_ROOT,
        cfg: ProvenanceConfig = ProvenanceConfig(),
    ) -> Self:
        """Record everything except `finished_at`; defaults come from `sys.argv`."""
        return cls(
            script=_relative_script(script or Path(sys.argv[0]), repo_root),
            argv=tuple(sys.argv[1:] if argv is None else argv),
            git_commit=git_commit(repo_root),
            git_dirty=git_is_dirty(repo_root),
            started_at=datetime.now(UTC),
            finished_at=None,
            package_versions=tuple(_package_version(name) for name in cfg.packages),
            robot_id=robot_id,
        )

    def write(self, out_dir: Path, repo_root: Path = PROJECT_ROOT, cfg: ProvenanceConfig = ProvenanceConfig()) -> None:
        """Write `manifest.json`, plus `uncommitted.patch` when the tree was dirty."""
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / cfg.manifest_filename).write_text(self.to_json())
        if self.git_dirty:
            (out_dir / cfg.patch_filename).write_text(git_diff(repo_root))

    def finalize(self, out_dir: Path, cfg: ProvenanceConfig = ProvenanceConfig()) -> Self:
        """Stamp `finished_at` and rewrite the manifest; a folder without it marks an aborted run."""
        done = replace(self, finished_at=datetime.now(UTC))
        (out_dir / cfg.manifest_filename).write_text(done.to_json())
        return done

    def to_json(self) -> str:
        """Serialize with ISO-8601 timestamps."""
        data: dict[str, Any] = asdict(self)
        data["started_at"] = self.started_at.isoformat()
        data["finished_at"] = self.finished_at.isoformat() if self.finished_at else None
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, text: str) -> Self:
        """Inverse of `to_json`."""
        data = json.loads(text)
        finished = data["finished_at"]
        return cls(
            script=data["script"],
            argv=tuple(data["argv"]),
            git_commit=data["git_commit"],
            git_dirty=data["git_dirty"],
            started_at=datetime.fromisoformat(data["started_at"]),
            finished_at=datetime.fromisoformat(finished) if finished else None,
            package_versions=tuple(PackageVersion(**p) for p in data["package_versions"]),
            robot_id=data["robot_id"],
        )


def experiment_dir(suffix: str, base: Path = EXPERIMENTS_DIR, day: date | None = None) -> Path:
    """`<base>/<YYYY-MM-DD>_<suffix>/` (not created)."""
    return base / f"{(day or date.today()).isoformat()}_{suffix}"


def begin_run(out_dir: Path, robot_id: str | None = None, repo_root: Path = PROJECT_ROOT) -> RunManifest:
    """Capture and write the manifest for a new output folder."""
    manifest = RunManifest.capture(robot_id, repo_root=repo_root)
    manifest.write(out_dir, repo_root)
    return manifest


def require_clean_tree(allow_dirty: bool, repo_root: Path = PROJECT_ROOT) -> None:
    """Exit with an error on uncommitted changes unless `allow_dirty` is set."""
    if allow_dirty or not git_is_dirty(repo_root):
        return
    sys.exit("Uncommitted changes: commit them, or pass --allow-dirty (hardware sessions are costly to repeat).")


def git_commit(repo_root: Path = PROJECT_ROOT) -> str:
    """HEAD commit hash, or `unknown` outside a repository."""
    return _git(repo_root, "rev-parse", "HEAD") or UNKNOWN_COMMIT


def git_is_dirty(repo_root: Path = PROJECT_ROOT) -> bool:
    """True if tracked files differ from HEAD (untracked files are ignored)."""
    return bool(_git(repo_root, "status", "--porcelain", "--untracked-files=no"))


def git_diff(repo_root: Path = PROJECT_ROOT) -> str:
    """`git diff HEAD`: every uncommitted change to tracked files."""
    return _git(repo_root, "diff", "HEAD", strip=False)


def _git(repo_root: Path, *args: str, strip: bool = True) -> str:
    """Run git in `repo_root` and return stdout ('' on failure)."""
    result = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return ""
    return result.stdout.strip() if strip else result.stdout


def _relative_script(script: Path, repo_root: Path) -> str:
    """Script path relative to the repo root (absolute if it lies outside)."""
    resolved = script.resolve()
    root = repo_root.resolve()
    return str(resolved.relative_to(root)) if resolved.is_relative_to(root) else str(resolved)


def _package_version(name: str) -> PackageVersion:
    """Installed version of one distribution, or `not installed`."""
    try:
        return PackageVersion(name, version(name))
    except PackageNotFoundError:
        return PackageVersion(name, NOT_INSTALLED)
