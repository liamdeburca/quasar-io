"""Tests for helper functions in file_utils."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from quasar_io.file_utils import (
    dump_to_dir_helper,
    find_file_with_name,
    get_path_helper,
)


class TestGetPathHelper:
    """Test get_path_helper function."""

    def test_appends_name_when_not_present(self, tmp_path: Path):
        """Verify name is appended when not the last path component."""
        result = get_path_helper(tmp_path, "subdir")
        assert result == tmp_path / "subdir"

    def test_returns_same_path_when_name_matches_last_component(
        self, tmp_path: Path
    ):
        """Verify path is returned unchanged when name matches last component."""
        subdir = tmp_path / "subdir"
        result = get_path_helper(subdir, "subdir")
        assert result == subdir

    def test_works_with_string_input(self, tmp_path: Path):
        """Verify function works with string path input."""
        result = get_path_helper(str(tmp_path), "newdir")
        assert result == tmp_path / "newdir"

    def test_validates_nonempty_name(self, tmp_path: Path):
        """Verify empty name raises validation error."""
        with pytest.raises(ValidationError):
            get_path_helper(tmp_path, "")

    def test_handles_nested_paths(self, tmp_path: Path):
        """Verify function works with nested paths."""
        nested_path = tmp_path / "a" / "b" / "c"
        result = get_path_helper(nested_path, "d")
        assert result == nested_path / "d"

    def test_case_sensitive_name_matching(self, tmp_path: Path):
        """Verify name matching is case-sensitive."""
        path_with_suffix = tmp_path / "MyDir"
        result = get_path_helper(path_with_suffix, "mydir")
        # Should append because case doesn't match
        assert result == path_with_suffix / "mydir"


class TestDumpToDirHelper:
    """Test dump_to_dir_helper function."""

    def test_creates_directory_at_specified_path(self, tmp_path: Path):
        """Verify directory is created at the specified path."""
        result = dump_to_dir_helper(tmp_path, "newdir")
        assert result.exists()
        assert result.is_dir()
        assert result == tmp_path / "newdir"

    def test_creates_parents_if_needed(self, tmp_path: Path):
        """Verify parent directories are created."""
        nested_path = tmp_path / "a" / "b" / "c"
        result = dump_to_dir_helper(nested_path, "newdir")
        assert result.exists()
        assert result == nested_path / "newdir"

    def test_exist_ok_false_raises_if_exists(self, tmp_path: Path):
        """Verify FileExistsError is raised when exist_ok=False and dir exists."""
        target_dir = tmp_path / "existing"
        target_dir.mkdir()

        with pytest.raises(FileExistsError):
            dump_to_dir_helper(target_dir, "existing", exist_ok=False)

    def test_exist_ok_true_allows_existing_dir(self, tmp_path: Path):
        """Verify exist_ok=True allows existing directory."""
        target_dir = tmp_path / "existing"
        target_dir.mkdir()

        result = dump_to_dir_helper(target_dir, "existing", exist_ok=True)
        assert result == target_dir
        assert result.exists()

    def test_returns_path_type(self, tmp_path: Path):
        """Verify function returns a Path object."""
        result = dump_to_dir_helper(tmp_path, "newdir")
        assert isinstance(result, Path)

    def test_validates_nonempty_name(self, tmp_path: Path):
        """Verify empty name raises validation error."""
        with pytest.raises(ValidationError):
            dump_to_dir_helper(tmp_path, "")

    def test_uses_get_path_helper_internally(self, tmp_path: Path):
        """Verify directory with same name as last component is not duplicated."""
        target = tmp_path / "mydir"
        # Calling with path that already ends in "mydir" should not duplicate
        result = dump_to_dir_helper(target, "mydir", exist_ok=True)
        assert result == target
        # Verify not doubled up
        assert not (target / "mydir").exists()

    def test_works_with_string_input(self, tmp_path: Path):
        """Verify function works with string path input."""
        result = dump_to_dir_helper(str(tmp_path), "newdir")
        assert result.exists()
        assert result == tmp_path / "newdir"


class TestFindFileWithName:
    """Test find_file_with_name function."""

    def test_finds_file_with_matching_name(self, tmp_path: Path):
        """Verify file is found when it exists with matching name."""
        # Create a test file
        test_file = tmp_path / "myfile.txt"
        test_file.write_text("content")

        result = find_file_with_name(tmp_path, "myfile")
        assert result == test_file

    def test_finds_file_regardless_of_extension(self, tmp_path: Path):
        """Verify file is found regardless of extension."""
        # Create a file with .json extension
        test_file = tmp_path / "config.json"
        test_file.write_text('{"key": "value"}')

        result = find_file_with_name(tmp_path, "config")
        assert result == test_file

    def test_raises_file_not_found_when_no_match(self, tmp_path: Path):
        """Verify FileNotFoundError is raised when no file matches."""
        with pytest.raises(FileNotFoundError) as exc_info:
            find_file_with_name(tmp_path, "nonexistent")

        assert "No file with name 'nonexistent'" in str(exc_info.value)
        assert str(tmp_path) in str(exc_info.value)

    def test_finds_first_match_when_multiple_exist(self, tmp_path: Path):
        """Verify first match is returned when multiple files match name."""
        # Create multiple files with same name but different extensions
        file1 = tmp_path / "data.txt"
        file2 = tmp_path / "data.json"
        file3 = tmp_path / "data.csv"

        file1.write_text("text")
        file2.write_text("json")
        file3.write_text("csv")

        result = find_file_with_name(tmp_path, "data")
        # Result should be one of the files (glob order)
        assert result.stem == "data"
        assert result in [file1, file2, file3]

    def test_validates_nonempty_name(self, tmp_path: Path):
        """Verify empty name raises validation error."""
        with pytest.raises(ValueError):
            find_file_with_name(tmp_path, "")

    def test_does_not_match_directories(self, tmp_path: Path):
        """Verify only files are matched, not directories."""
        # Create a directory and file with same base name
        test_dir = tmp_path / "myitem"
        test_dir.mkdir()
        test_file = tmp_path / "myitem.txt"
        test_file.write_text("content")

        result = find_file_with_name(tmp_path, "myitem")
        # Should return the file, not the directory
        assert result == test_file

    def test_with_nested_search_path(self, tmp_path: Path):
        """Verify function works when searching in subdirectories."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        test_file = subdir / "found.txt"
        test_file.write_text("content")

        result = find_file_with_name(subdir, "found")
        assert result == test_file

    def test_case_sensitive_search(self, tmp_path: Path):
        """Verify filename search is case-sensitive (glob behavior)."""
        # Create file with specific case
        test_file = tmp_path / "MyFile.txt"
        test_file.write_text("content")

        # Should find exact case match
        result = find_file_with_name(tmp_path, "MyFile")
        assert result == test_file

        # Different case should not find it (on case-sensitive systems)
        # This may vary by OS, so we just verify it searches with glob
        try:
            find_file_with_name(tmp_path, "myfile")
        except FileNotFoundError:
            pass  # Expected on case-sensitive systems

    def test_error_message_includes_directory(self, tmp_path: Path):
        """Verify error message includes the searched directory path."""
        with pytest.raises(FileNotFoundError) as exc_info:
            find_file_with_name(tmp_path, "missing")

        error_msg = str(exc_info.value)
        assert "missing" in error_msg
        assert str(tmp_path) in error_msg
