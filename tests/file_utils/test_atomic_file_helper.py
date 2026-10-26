"""Tests for AtomicFileHelper context manager."""
from pathlib import Path

import pytest

from quasar_io.file_utils import AtomicFileHelper


class TestAtomicFileHelperBasicBehavior:
    """Test basic functionality of AtomicFileHelper."""

    def test_context_manager_creates_temp_file(self, dest_file_path: Path):
        """Verify that entering context creates a temporary file."""
        assert not dest_file_path.exists()

        with AtomicFileHelper(dest_file_path) as temp_file:
            # temp_file is the result of __enter__() which returns
            # self.src.__enter__()
            assert temp_file is not None
            # Destination should not exist yet
            assert not dest_file_path.exists()

    def test_raises_file_exists_error_by_default(
        self, existing_file: Path
    ):
        """Verify FileExistsError is raised if destination exists."""
        with pytest.raises(FileExistsError):
            AtomicFileHelper(existing_file)

    def test_exist_ok_true_allows_existing_file(
        self, existing_file: Path
    ):
        """Verify exist_ok=True allows creating helper with existing file."""
        # Should not raise
        helper = AtomicFileHelper(existing_file, exist_ok=True)
        assert helper.dst == existing_file

    def test_temp_file_has_correct_suffix(self, dest_file_path: Path):
        """Verify temporary file has the same suffix as destination."""
        dest_file_path = dest_file_path.with_suffix(".json")

        with AtomicFileHelper(dest_file_path) as temp_file:
            # Verify the temp file has the correct suffix
            assert hasattr(temp_file, 'name')
            assert Path(temp_file.name).suffix == ".json"

    def test_can_write_to_temp_file(
        self, dest_file_path: Path, sample_file_content: str
    ):
        """Verify that content can be written to the temp file."""
        with AtomicFileHelper(dest_file_path) as temp_file:
            temp_file.write(sample_file_content.encode())
            temp_file.flush()

    def test_destination_file_not_created_without_commit(
        self, dest_file_path: Path, sample_file_content: str
    ):
        """Verify destination file is not created if commit() is not called."""
        with AtomicFileHelper(dest_file_path) as temp_file:
            temp_file.write(sample_file_content.encode())
            temp_file.flush()

        # After context exit without commit, destination should not exist
        assert not dest_file_path.exists()

    def test_path_property_raises_before_enter(
        self, dest_file_path: Path
    ):
        """Verify .path raises RuntimeError if accessed before __enter__."""
        helper = AtomicFileHelper(dest_file_path)
        with pytest.raises(
            RuntimeError,
            match="Temporary file \\(src\\) does not exist",
        ):
            _ = helper.path

    def test_can_read_temp_file_content(
        self, dest_file_path: Path, sample_file_content: str
    ):
        """Verify content written to temp file can be read back."""
        with AtomicFileHelper(dest_file_path) as temp_file:
            temp_file.write(sample_file_content.encode())
            temp_file.flush()
            # Seek to beginning to read
            temp_file.seek(0)
            content = temp_file.read()
            assert content == sample_file_content.encode()


class TestAtomicFileHelperExceptionHandling:
    """Test exception handling in AtomicFileHelper."""

    def test_exception_during_context_cleans_up_temp_file(
        self, dest_file_path: Path
    ):
        """Verify temp file is cleaned up even if exception occurs."""
        with pytest.raises(ValueError):
            with AtomicFileHelper(dest_file_path) as temp_file:
                temp_file.write(b"some content")
                raise ValueError("Simulated error during write")

        # Destination should not exist (no commit)
        assert not dest_file_path.exists()

    def test_exception_before_commit_leaves_destination_untouched(
        self, dest_file_path: Path, sample_file_content: str
    ):
        """Verify destination is not modified if exception before commit."""
        # Create existing destination file
        original_content = "original content"
        dest_file_path.write_text(original_content)

        with pytest.raises(ValueError):
            helper = AtomicFileHelper(dest_file_path, exist_ok=True)
            with helper as temp_file:
                temp_file.write(sample_file_content.encode())
                raise ValueError("Error before commit")

        # Destination should still have original content
        assert dest_file_path.read_text() == original_content

    def test_existing_file_raises_without_exist_ok(
        self, tmp_path: Path
    ):
        """Verify FileExistsError is raised for existing file without exist_ok."""
        dest_file_path = tmp_path / "file.txt"
        dest_file_path.write_text("original")

        with pytest.raises(FileExistsError):
            AtomicFileHelper(dest_file_path)

    def test_init_validates_destination_exists_check(
        self, dest_file_path: Path
    ):
        """Verify __init__ checks destination file existence."""
        # First, file doesn't exist - should work
        helper = AtomicFileHelper(dest_file_path)
        assert helper.src is None  # src not set until __enter__

        # Now create the file and try again - should fail
        dest_file_path.write_text("content")
        with pytest.raises(FileExistsError) as exc_info:
            AtomicFileHelper(dest_file_path)
        assert "already exists" in str(exc_info.value)

    def test_context_manager_is_reentrant(self, dest_file_path: Path):
        """Verify multiple sequential contexts work correctly."""
        content1 = b"content1"
        content2 = b"content2"

        # First context (without commit)
        with AtomicFileHelper(dest_file_path) as temp_file:
            temp_file.write(content1)
        assert not dest_file_path.exists()

        # Second context (also without commit)
        with AtomicFileHelper(dest_file_path) as temp_file:
            temp_file.write(content2)
        assert not dest_file_path.exists()
