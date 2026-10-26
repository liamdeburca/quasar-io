"""Tests for AtomicDirectoryHelper context manager."""
from pathlib import Path

import pytest

from quasar_io.file_utils import AtomicDirectoryHelper


class TestAtomicDirectoryHelperBasicBehavior:
    """Test basic functionality of AtomicDirectoryHelper."""

    def test_context_manager_creates_temp_directory(
        self, dest_dir_path: Path
    ):
        """Verify that entering context creates a temporary directory."""
        assert not dest_dir_path.exists()

        with AtomicDirectoryHelper(dest_dir_path) as temp_dir:
            # temp_dir is the result of __enter__() which returns
            # self.src.__enter__() (a string path)
            assert temp_dir is not None
            assert Path(temp_dir).exists()
            assert Path(temp_dir).is_dir()
            # Destination should not exist yet
            assert not dest_dir_path.exists()

    def test_raises_file_exists_error_by_default(
        self, existing_dir: Path
    ):
        """Verify FileExistsError is raised if destination exists."""
        with pytest.raises(FileExistsError):
            AtomicDirectoryHelper(existing_dir)

    def test_exist_ok_true_allows_existing_dir(self, existing_dir: Path):
        """Verify exist_ok=True allows creating helper with existing dir."""
        # Should not raise
        helper = AtomicDirectoryHelper(existing_dir, exist_ok=True)
        assert helper.dst == existing_dir

    def test_can_create_files_in_temp_directory(
        self, dest_dir_path: Path
    ):
        """Verify that files can be created in the temp directory."""
        with AtomicDirectoryHelper(dest_dir_path) as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test content")
            assert test_file.exists()

    def test_can_create_subdirectories(self, dest_dir_path: Path):
        """Verify that subdirectories can be created in temp directory."""
        with AtomicDirectoryHelper(dest_dir_path) as temp_dir:
            subdir = Path(temp_dir) / "subdir"
            subdir.mkdir()
            assert subdir.exists()
            assert subdir.is_dir()

    def test_destination_directory_not_created_without_commit(
        self, dest_dir_path: Path
    ):
        """Verify destination directory is not created if commit() is not called."""
        with AtomicDirectoryHelper(dest_dir_path) as temp_dir:
            (Path(temp_dir) / "file.txt").write_text("content")

        # After context exit without commit, destination should not exist
        assert not dest_dir_path.exists()

    def test_path_property_raises_before_enter(self, dest_dir_path: Path):
        """Verify .path raises RuntimeError if accessed before __enter__."""
        helper = AtomicDirectoryHelper(dest_dir_path)
        with pytest.raises(
            RuntimeError,
            match="Temporary directory \\(src\\) does not exist",
        ):
            _ = helper.path

    def test_can_create_nested_directories(self, dest_dir_path: Path):
        """Verify nested directory structures can be created."""
        with AtomicDirectoryHelper(dest_dir_path) as temp_dir:
            temp_path = Path(temp_dir)
            nested = temp_path / "a" / "b" / "c"
            nested.mkdir(parents=True)
            assert nested.exists()

    def test_temp_directory_is_isolated(self, tmp_path: Path):
        """Verify each temp directory is isolated."""
        dest_dir_path1 = tmp_path / "dest1"
        dest_dir_path2 = tmp_path / "dest2"

        temp_dirs = []

        with AtomicDirectoryHelper(dest_dir_path1) as temp_dir1:
            temp_dirs.append(temp_dir1)
            (Path(temp_dir1) / "file1.txt").write_text("content1")

            with AtomicDirectoryHelper(dest_dir_path2) as temp_dir2:
                temp_dirs.append(temp_dir2)
                (Path(temp_dir2) / "file2.txt").write_text("content2")
                # Both should exist and be different
                assert temp_dir1 != temp_dir2
                assert (Path(temp_dir1) / "file1.txt").exists()
                assert (Path(temp_dir2) / "file2.txt").exists()
                assert not (Path(temp_dir2) / "file1.txt").exists()


class TestAtomicDirectoryHelperExceptionHandling:
    """Test exception handling in AtomicDirectoryHelper."""

    def test_exception_during_context_cleans_up_temp_dir(
        self, dest_dir_path: Path
    ):
        """Verify temp directory is cleaned up even if exception occurs."""
        with pytest.raises(ValueError):
            with AtomicDirectoryHelper(dest_dir_path) as temp_dir:
                (Path(temp_dir) / "file.txt").write_text("content")
                raise ValueError("Simulated error during write")

        # Destination should not exist (no commit)
        assert not dest_dir_path.exists()

    def test_exception_before_commit_leaves_destination_untouched(
        self, dest_dir_path: Path
    ):
        """Verify destination is not modified if exception before commit."""
        # Create existing destination directory
        dest_dir_path.mkdir()
        original_file = dest_dir_path / "original.txt"
        original_file.write_text("original content")

        with pytest.raises(ValueError):
            helper = AtomicDirectoryHelper(dest_dir_path, exist_ok=True)
            with helper as temp_dir:
                temp_path = Path(temp_dir)
                new_file = temp_path / "new.txt"
                new_file.write_text("new content")
                raise ValueError("Error before commit")

        # Destination should still have original content
        assert original_file.exists()
        assert original_file.read_text() == "original content"
        assert not (dest_dir_path / "new.txt").exists()

    def test_existing_directory_raises_without_exist_ok(
        self, tmp_path: Path
    ):
        """Verify FileExistsError is raised for existing dir without exist_ok."""
        dest_dir_path = tmp_path / "dir"
        dest_dir_path.mkdir()

        with pytest.raises(FileExistsError):
            AtomicDirectoryHelper(dest_dir_path)

    def test_init_validates_destination_exists_check(
        self, dest_dir_path: Path
    ):
        """Verify __init__ checks destination directory existence."""
        # First, directory doesn't exist - should work
        helper = AtomicDirectoryHelper(dest_dir_path)
        assert helper.src is None  # src not set until __enter__

        # Now create the directory and try again - should fail
        dest_dir_path.mkdir()
        with pytest.raises(FileExistsError) as exc_info:
            AtomicDirectoryHelper(dest_dir_path)
        assert "already exists" in str(exc_info.value)

    def test_context_manager_is_reentrant(self, dest_dir_path: Path):
        """Verify multiple sequential contexts work correctly."""
        # First context (without commit)
        with AtomicDirectoryHelper(dest_dir_path) as temp_dir:
            (Path(temp_dir) / "file1.txt").write_text("content1")
        assert not dest_dir_path.exists()

        # Second context (also without commit)
        with AtomicDirectoryHelper(dest_dir_path) as temp_dir:
            (Path(temp_dir) / "file2.txt").write_text("content2")
        assert not dest_dir_path.exists()
