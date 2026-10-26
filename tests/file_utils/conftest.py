"""Fixtures for file_utils tests."""
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def sample_file_content() -> str:
    """Provide sample file content for testing."""
    return "This is test content for atomic file operations."


@pytest.fixture
def sample_binary_content() -> bytes:
    """Provide sample binary content for testing."""
    return b"Binary test content \x00\x01\x02"


@pytest.fixture
def existing_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a file that exists before test runs."""
    test_file = tmp_path / "existing.txt"
    test_file.write_text("original content")
    yield test_file


@pytest.fixture
def existing_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a directory that exists before test runs."""
    test_dir = tmp_path / "existing_dir"
    test_dir.mkdir()
    (test_dir / "file1.txt").write_text("file 1")
    (test_dir / "file2.txt").write_text("file 2")
    yield test_dir


@pytest.fixture
def dest_file_path(tmp_path: Path) -> Path:
    """Provide a destination file path (doesn't exist yet)."""
    return tmp_path / "destination.txt"


@pytest.fixture
def dest_dir_path(tmp_path: Path) -> Path:
    """Provide a destination directory path (doesn't exist yet)."""
    return tmp_path / "destination_dir"
