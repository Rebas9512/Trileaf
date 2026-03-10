"""
Distribution integrity tests.

Verifies that all files required for a clean-clone install are present,
correctly tracked by git, and internally consistent.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── Required files ────────────────────────────────────────────────────────────

REQUIRED_TRACKED = [
    "README.md",
    "MANIFEST.in",
    ".gitignore",
    "requirements.txt",
    "run.py",
    "setup.sh",
    "setup.ps1",
    "scripts/_version.py",
    "scripts/onboarding.py",
    "scripts/check_env.py",
    "scripts/rewrite_config.py",
    "scripts/rewrite_provider_cli.py",
    "scripts/chunker.py",
    "scripts/models_runtime.py",
    "scripts/orchestrator.py",
    "scripts/download_scripts/__init__.py",
    "scripts/download_scripts/desklib_detector_download.py",
    "scripts/download_scripts/mpnet_download.py",
    "scripts/download_scripts/qwen3_vl_download.py",
    "api/optimizer_api.py",
    "api/static/index.html",
    "api/static/app.js",
    "models/.gitkeep",
    "pyproject.toml",
    "trileaf_cli.py",
]


@pytest.mark.parametrize("rel_path", REQUIRED_TRACKED)
def test_required_file_exists(rel_path: str) -> None:
    assert (PROJECT_ROOT / rel_path).exists(), f"Missing distribution file: {rel_path}"


def _git_tracked() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True, text=True, cwd=PROJECT_ROOT, check=True,
    )
    return set(result.stdout.splitlines())


@pytest.mark.parametrize("rel_path", REQUIRED_TRACKED)
def test_required_file_is_git_tracked(rel_path: str) -> None:
    tracked = _git_tracked()
    assert rel_path in tracked, f"File exists but is not tracked by git: {rel_path}"


def test_setup_sh_is_executable_in_git() -> None:
    result = subprocess.run(
        ["git", "ls-files", "-s", "setup.sh"],
        capture_output=True, text=True, cwd=PROJECT_ROOT, check=True,
    )
    mode = result.stdout.split()[0]
    assert mode == "100755", f"setup.sh git mode should be 100755 (executable), got {mode}"


# ── Python syntax ─────────────────────────────────────────────────────────────

ALL_PY = [
    p.relative_to(PROJECT_ROOT).as_posix()
    for p in PROJECT_ROOT.rglob("*.py")
    if not any(part in p.parts for part in (".venv", "openclaw", "__pycache__"))
]


@pytest.mark.parametrize("rel_path", ALL_PY)
def test_python_syntax(rel_path: str) -> None:
    src = (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")
    try:
        ast.parse(src, filename=rel_path)
    except SyntaxError as exc:
        pytest.fail(f"SyntaxError in {rel_path}: {exc}")


# ── Distribution docs + packaging layout ─────────────────────────────────────

def test_readme_uses_json_config_files() -> None:
    src = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "~/.trileaf/config.json" in src
    assert "~/.trileaf/rewrite_profiles.json" in src


def test_readme_does_not_advertise_env_as_primary_config() -> None:
    src = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert ".env.example" not in src
    assert "copy to .env" not in src


def test_download_scripts_are_packaged_modules() -> None:
    assert (PROJECT_ROOT / "scripts" / "download_scripts" / "__init__.py").exists()


def test_one_liner_installers_exist() -> None:
    for rel_path in ("install.sh", "install.ps1", "install.cmd"):
        assert (PROJECT_ROOT / rel_path).exists(), f"Missing installer: {rel_path}"


def test_optimizer_api_uses_packaged_static_dir() -> None:
    src = (PROJECT_ROOT / "api" / "optimizer_api.py").read_text(encoding="utf-8")
    assert 'Path(__file__).resolve().parent / "static"' in src


def test_pyproject_includes_packaged_static_assets() -> None:
    src = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'build-backend = "setuptools.build_meta"' in src
    assert 'api = ["static/*.html", "static/*.js", "static/*.css"]' in src


def test_readme_documents_one_liner_and_remove_flows() -> None:
    src = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "install.cmd" in src
    assert "trileaf remove" in src
    assert "--purge-source" in src


# ── Version consistency ───────────────────────────────────────────────────────

def test_version_importable() -> None:
    from scripts._version import __version__
    assert isinstance(__version__, str) and __version__, "Version string is empty"


def test_version_format() -> None:
    from scripts._version import __version__
    parts = __version__.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts), (
        f"Expected semver X.Y.Z, got {__version__!r}"
    )


# ── No stray openclaw user-facing strings ─────────────────────────────────────

WIZARD_FILES = [
    "scripts/onboarding.py",
    "scripts/rewrite_provider_cli.py",
]


@pytest.mark.parametrize("rel_path", WIZARD_FILES)
def test_no_openclaw_user_strings(rel_path: str) -> None:
    """User-visible wizard text must not reference openclaw."""
    tree = ast.parse((PROJECT_ROOT / rel_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "openclaw" not in node.value.lower(), (
                f"User-facing string in {rel_path} references openclaw: {node.value[:80]!r}"
            )


USER_FACING_FILES = [
    "README.md",
    "scripts/onboarding.py",
    "scripts/check_env.py",
    "run.py",
    "setup.sh",
    "setup.ps1",
    "api/static/index.html",
    "api/static/app.js",
]


@pytest.mark.parametrize("rel_path", USER_FACING_FILES)
def test_no_legacy_project_name_in_user_facing_files(rel_path: str) -> None:
    src = (PROJECT_ROOT / rel_path).read_text(encoding="utf-8").lower()
    assert "llm writing optimizer" not in src, f"Legacy project name found in {rel_path}"
