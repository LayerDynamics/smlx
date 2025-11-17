"""Tests for smlx.utils.io module."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from smlx.utils.io import (
    append_csv,
    append_jsonl,
    ensure_dir,
    load_csv,
    load_json,
    load_jsonl,
    save_csv,
    save_json,
    save_jsonl,
)


@dataclass
class SampleData:
    """Sample dataclass for testing."""

    name: str
    value: int


class TestJSON:
    """Test JSON save/load functions."""

    def test_save_load_json_dict(self):
        """Test saving and loading dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            data = {"name": "Alice", "age": 30, "city": "NYC"}

            save_json(data, filepath)
            loaded = load_json(filepath)

            assert loaded == data

    def test_save_load_json_list(self):
        """Test saving and loading list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            data = [1, 2, 3, 4, 5]

            save_json(data, filepath)
            loaded = load_json(filepath)

            assert loaded == data

    def test_save_json_dataclass(self):
        """Test saving dataclass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            data = SampleData(name="test", value=42)

            save_json(data, filepath)
            loaded = load_json(filepath)

            assert loaded["name"] == "test"
            assert loaded["value"] == 42

    def test_save_json_list_of_dataclasses(self):
        """Test saving list of dataclasses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            data = [
                SampleData(name="a", value=1),
                SampleData(name="b", value=2),
            ]

            save_json(data, filepath)
            loaded = load_json(filepath)

            assert len(loaded) == 2
            assert loaded[0]["name"] == "a"

    def test_save_json_creates_dir(self):
        """Test that save_json creates parent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "test.json"
            data = {"key": "value"}

            save_json(data, filepath)

            assert filepath.exists()
            assert filepath.parent.exists()

    def test_save_json_custom_indent(self):
        """Test saving JSON with custom indentation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            data = {"a": 1, "b": 2}

            save_json(data, filepath, indent=4)

            # Check that file is indented
            content = filepath.read_text()
            assert "    " in content  # 4 spaces


class TestCSV:
    """Test CSV save/load functions."""

    def test_save_load_csv_basic(self):
        """Test saving and loading CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            data = [
                {"name": "Alice", "age": "30"},
                {"name": "Bob", "age": "25"},
            ]

            save_csv(data, filepath)
            loaded = load_csv(filepath)

            assert len(loaded) == 2
            assert loaded[0]["name"] == "Alice"
            assert loaded[1]["name"] == "Bob"

    def test_save_csv_empty(self):
        """Test saving empty CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"

            save_csv([], filepath)

            # File should not be created for empty data
            assert not filepath.exists()

    def test_save_csv_dataclass(self):
        """Test saving dataclasses to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            data = [
                SampleData(name="a", value=1),
                SampleData(name="b", value=2),
            ]

            save_csv(data, filepath)
            loaded = load_csv(filepath)

            assert len(loaded) == 2
            assert loaded[0]["name"] == "a"

    def test_save_csv_custom_fieldnames(self):
        """Test saving CSV with custom field names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            data = [
                {"name": "Alice", "age": "30", "city": "NYC"},
            ]

            # Only save specific fields
            save_csv(data, filepath, fieldnames=["name", "age"])
            loaded = load_csv(filepath)

            assert "name" in loaded[0]
            assert "age" in loaded[0]
            # city should not be in CSV

    def test_append_csv(self):
        """Test appending to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"

            # Append first row
            append_csv({"name": "Alice", "age": "30"}, filepath)

            # Append second row
            append_csv({"name": "Bob", "age": "25"}, filepath)

            loaded = load_csv(filepath)

            assert len(loaded) == 2
            assert loaded[0]["name"] == "Alice"
            assert loaded[1]["name"] == "Bob"

    def test_append_csv_dataclass(self):
        """Test appending dataclass to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"

            append_csv(SampleData(name="a", value=1), filepath)
            append_csv(SampleData(name="b", value=2), filepath)

            loaded = load_csv(filepath)

            assert len(loaded) == 2


class TestJSONL:
    """Test JSON Lines save/load functions."""

    def test_save_load_jsonl(self):
        """Test saving and loading JSON Lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.jsonl"
            data = [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ]

            save_jsonl(data, filepath)
            loaded = load_jsonl(filepath)

            assert len(loaded) == 2
            assert loaded[0]["name"] == "Alice"
            assert loaded[1]["name"] == "Bob"

    def test_save_jsonl_dataclass(self):
        """Test saving dataclasses to JSON Lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.jsonl"
            data = [
                SampleData(name="a", value=1),
                SampleData(name="b", value=2),
            ]

            save_jsonl(data, filepath)
            loaded = load_jsonl(filepath)

            assert len(loaded) == 2
            assert loaded[0]["name"] == "a"

    def test_append_jsonl(self):
        """Test appending to JSON Lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.jsonl"

            # Append first item
            append_jsonl({"name": "Alice"}, filepath)

            # Append second item
            append_jsonl({"name": "Bob"}, filepath)

            loaded = load_jsonl(filepath)

            assert len(loaded) == 2
            assert loaded[0]["name"] == "Alice"
            assert loaded[1]["name"] == "Bob"

    def test_append_jsonl_dataclass(self):
        """Test appending dataclass to JSON Lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.jsonl"

            append_jsonl(SampleData(name="a", value=1), filepath)
            append_jsonl(SampleData(name="b", value=2), filepath)

            loaded = load_jsonl(filepath)

            assert len(loaded) == 2

    def test_load_jsonl_empty_lines(self):
        """Test loading JSON Lines with empty lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.jsonl"

            # Write file with empty lines
            with open(filepath, "w") as f:
                f.write('{"a": 1}\n')
                f.write('\n')  # Empty line
                f.write('{"b": 2}\n')

            loaded = load_jsonl(filepath)

            # Should skip empty lines
            assert len(loaded) == 2


class TestEnsureDir:
    """Test ensure_dir function."""

    def test_ensure_dir_file_path(self):
        """Test ensuring directory for file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "file.txt"

            result = ensure_dir(filepath)

            assert result == filepath
            assert filepath.parent.exists()

    def test_ensure_dir_directory_path(self):
        """Test ensuring directory path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dirpath = Path(tmpdir) / "subdir" / "subsubdir"

            result = ensure_dir(dirpath)

            assert result == dirpath
            assert dirpath.exists()
            assert dirpath.is_dir()

    def test_ensure_dir_already_exists(self):
        """Test ensure_dir with existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dirpath = Path(tmpdir) / "existing"
            dirpath.mkdir()

            result = ensure_dir(dirpath)

            assert result == dirpath
            assert dirpath.exists()

    def test_ensure_dir_nested(self):
        """Test ensure_dir with deeply nested path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "a" / "b" / "c" / "d" / "file.txt"

            ensure_dir(filepath)

            assert filepath.parent.exists()


class TestIOEdgeCases:
    """Test edge cases and error handling."""

    def test_load_json_missing_file(self):
        """Test loading non-existent JSON file."""
        with pytest.raises(FileNotFoundError):
            load_json("/nonexistent/path/file.json")

    def test_load_csv_missing_file(self):
        """Test loading non-existent CSV file."""
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path/file.csv")

    def test_load_jsonl_missing_file(self):
        """Test loading non-existent JSONL file."""
        with pytest.raises(FileNotFoundError):
            load_jsonl("/nonexistent/path/file.jsonl")

    def test_save_json_invalid_data(self):
        """Test saving non-serializable data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"

            # Functions are not JSON serializable
            with pytest.raises((TypeError, ValueError)):
                save_json({"func": lambda x: x}, filepath)


class TestPathTypes:
    """Test that functions accept both str and Path."""

    def test_save_json_str_path(self):
        """Test save_json with string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "test.json")
            data = {"key": "value"}

            save_json(data, filepath)

            assert Path(filepath).exists()

    def test_save_csv_str_path(self):
        """Test save_csv with string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "test.csv")
            data = [{"name": "test"}]

            save_csv(data, filepath)

            assert Path(filepath).exists()

    def test_ensure_dir_str_path(self):
        """Test ensure_dir with string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dirpath = str(Path(tmpdir) / "subdir")

            result = ensure_dir(dirpath)

            assert isinstance(result, Path)
            assert result.exists()
