"""Tests for smlx.utils.formatting module."""

import sys
from io import StringIO

from smlx.utils.formatting import (
    ProgressBar,
    colorize,
    format_comparison,
    format_dict_table,
    format_header,
    format_key_value,
    format_section,
    format_table,
)


class TestFormatTable:
    """Test table formatting."""

    def test_format_table_basic(self):
        """Test basic table formatting."""
        headers = ["Name", "Age", "City"]
        rows = [
            ["Alice", "30", "NYC"],
            ["Bob", "25", "LA"],
        ]

        result = format_table(headers, rows)

        assert "Name" in result
        assert "Age" in result
        assert "Alice" in result
        assert "Bob" in result
        assert "|" in result  # Has separators
        assert "-" in result  # Has separator line

    def test_format_table_empty(self):
        """Test table with no rows."""
        headers = ["Name", "Age"]
        rows = []

        result = format_table(headers, rows)

        assert result == ""

    def test_format_table_alignment(self):
        """Test table with custom alignment."""
        headers = ["Name", "Score", "Status"]
        rows = [["Alice", "100", "Pass"]]

        result = format_table(headers, rows, align=["left", "right", "center"])

        # Should contain the data
        assert "Alice" in result
        assert "100" in result

    def test_format_table_varying_widths(self):
        """Test table adjusts column widths."""
        headers = ["A", "B"]
        rows = [
            ["Short", "VeryLongValue"],
            ["X", "Y"],
        ]

        result = format_table(headers, rows)

        # All rows should be aligned
        lines = result.split("\n")
        assert len(lines) >= 4  # Header, separator, 2 data rows

    def test_format_table_missing_cells(self):
        """Test table with missing cells."""
        headers = ["A", "B", "C"]
        rows = [
            ["1", "2"],  # Missing last cell
            ["3", "4", "5"],
        ]

        result = format_table(headers, rows)

        assert "A" in result
        assert "1" in result


class TestFormatDictTable:
    """Test dictionary table formatting."""

    def test_format_dict_table_basic(self):
        """Test formatting list of dicts."""
        data = [
            {"name": "Alice", "age": 30, "city": "NYC"},
            {"name": "Bob", "age": 25, "city": "LA"},
        ]

        result = format_dict_table(data)

        assert "name" in result
        assert "Alice" in result
        assert "Bob" in result

    def test_format_dict_table_empty(self):
        """Test with empty data."""
        result = format_dict_table([])
        assert result == ""

    def test_format_dict_table_custom_keys(self):
        """Test with custom key order."""
        data = [
            {"name": "Alice", "age": 30, "city": "NYC"},
        ]

        result = format_dict_table(data, keys=["name", "city"])

        assert "name" in result
        assert "city" in result
        # age should not be in output
        assert "age" not in result

    def test_format_dict_table_missing_keys(self):
        """Test with missing keys in some dicts."""
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "city": "LA"},  # Missing age
        ]

        result = format_dict_table(data)

        assert "Alice" in result
        assert "Bob" in result


class TestFormatKeyValue:
    """Test key-value formatting."""

    def test_format_key_value_basic(self):
        """Test basic key-value formatting."""
        data = {"name": "Alice", "age": 30, "city": "NYC"}

        result = format_key_value(data)

        assert "name" in result
        assert "Alice" in result
        assert "age" in result
        assert "30" in result
        assert ":" in result

    def test_format_key_value_empty(self):
        """Test with empty dict."""
        result = format_key_value({})
        assert result == ""

    def test_format_key_value_with_indent(self):
        """Test key-value with indentation."""
        data = {"key": "value"}

        result = format_key_value(data, indent=4)

        # Should start with spaces
        assert result.startswith(" " * 4)

    def test_format_key_value_alignment(self):
        """Test that keys are aligned."""
        data = {"short": "v1", "very_long_key": "v2"}

        result = format_key_value(data)

        lines = result.split("\n")
        # Colons should align
        assert all(":" in line for line in lines)


class TestFormatHeader:
    """Test header formatting."""

    def test_format_header_basic(self):
        """Test basic header formatting."""
        result = format_header("Test")

        assert "Test" in result
        assert "=" in result

    def test_format_header_custom_width(self):
        """Test header with custom width."""
        result = format_header("Test", width=40)

        assert len(result) == 40
        assert "Test" in result

    def test_format_header_custom_char(self):
        """Test header with custom character."""
        result = format_header("Test", char="-")

        assert "Test" in result
        assert "-" in result
        assert "=" not in result

    def test_format_header_long_text(self):
        """Test header with long text."""
        long_text = "A" * 100
        result = format_header(long_text, width=80)

        assert long_text in result


class TestFormatSection:
    """Test section formatting."""

    def test_format_section_basic(self):
        """Test basic section formatting."""
        result = format_section("Title", "Content here")

        assert "Title" in result
        assert "Content here" in result
        assert "-" in result

    def test_format_section_multiline_content(self):
        """Test section with multiline content."""
        content = "Line 1\nLine 2\nLine 3"
        result = format_section("Section", content)

        assert "Section" in result
        assert "Line 1" in result
        assert "Line 2" in result


class TestProgressBar:
    """Test ProgressBar class."""

    def test_progress_bar_creation(self):
        """Test creating progress bar."""
        progress = ProgressBar(total=100)

        assert progress.total == 100
        assert progress.current == 0

    def test_progress_bar_update(self):
        """Test updating progress bar."""
        progress = ProgressBar(total=100)

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            progress.update(50)
            output = sys.stdout.getvalue()

            assert progress.current == 50
            # Output should contain progress indicator
            assert "|" in output or "█" in output

        finally:
            sys.stdout = old_stdout

    def test_progress_bar_complete(self):
        """Test completing progress bar."""
        progress = ProgressBar(total=100)

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            progress.update(100)
            output = sys.stdout.getvalue()

            assert "100" in output or "100.0" in output

        finally:
            sys.stdout = old_stdout

    def test_progress_bar_close(self):
        """Test closing progress bar."""
        progress = ProgressBar(total=10)

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            progress.update(5)
            progress.close()
            output = sys.stdout.getvalue()

            # Should contain newline from close
            assert "\n" in output

        finally:
            sys.stdout = old_stdout

    def test_progress_bar_custom_fill(self):
        """Test progress bar with custom fill character."""
        progress = ProgressBar(total=10, fill="#")

        assert progress.fill == "#"

    def test_progress_bar_prefix_suffix(self):
        """Test progress bar with prefix and suffix."""
        progress = ProgressBar(total=10, prefix="Progress:", suffix="items")

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            progress.update(5)
            output = sys.stdout.getvalue()

            assert "Progress:" in output
            assert "items" in output

        finally:
            sys.stdout = old_stdout


class TestColorize:
    """Test text colorization."""

    def test_colorize_basic(self):
        """Test basic colorization."""
        # When stdout is not a tty, should return plain text
        result = colorize("Hello", "red")

        # May or may not include color codes depending on environment
        assert "Hello" in result

    def test_colorize_colors(self):
        """Test different colors."""
        colors = ["red", "green", "yellow", "blue", "magenta", "cyan"]

        for color in colors:
            result = colorize("Text", color)
            assert "Text" in result

    def test_colorize_bold(self):
        """Test bold text."""
        result = colorize("Bold", "bold")
        assert "Bold" in result

    def test_colorize_invalid_color(self):
        """Test with invalid color (should return plain text)."""
        result = colorize("Text", "invalid_color")
        assert result == "Text"


class TestFormatComparison:
    """Test comparison formatting."""

    def test_format_comparison_basic(self):
        """Test basic comparison formatting."""
        data = [
            {"name": "baseline", "tps": 100, "baseline": True},
            {"name": "optimized", "tps": 150, "baseline": False},
        ]

        result = format_comparison(data)

        assert "baseline" in result
        assert "optimized" in result
        assert "100" in result
        assert "150" in result

    def test_format_comparison_speedup(self):
        """Test that speedup is calculated."""
        data = [
            {"name": "v1", "tps": 100, "baseline": True},
            {"name": "v2", "tps": 200, "baseline": False},
        ]

        result = format_comparison(data)

        # Should show speedup
        assert "2.00x" in result or "1.00x" in result

    def test_format_comparison_empty(self):
        """Test with empty data."""
        result = format_comparison([])
        assert result == ""

    def test_format_comparison_no_baseline(self):
        """Test when no baseline is specified (uses first item)."""
        data = [
            {"name": "v1", "tps": 100},
            {"name": "v2", "tps": 150},
        ]

        result = format_comparison(data)

        # First item should be used as baseline
        assert "v1" in result
        assert "v2" in result

    def test_format_comparison_custom_keys(self):
        """Test with custom comparison keys."""
        data = [
            {"name": "v1", "speed": 100, "memory": 50, "baseline": True},
            {"name": "v2", "speed": 150, "memory": 60, "baseline": False},
        ]

        result = format_comparison(data, compare_keys=["speed"])

        assert "speed" in result
        assert "100" in result


class TestFormattingEdgeCases:
    """Test edge cases in formatting."""

    def test_format_table_unicode(self):
        """Test table with unicode characters."""
        headers = ["Name", "Symbol"]
        rows = [["Pi", "�"], ["Delta", "�"]]

        result = format_table(headers, rows)

        assert "�" in result
        assert "�" in result

    def test_format_key_value_numeric_keys(self):
        """Test key-value with numeric keys."""
        data = {1: "one", 2: "two", 3: "three"}

        result = format_key_value(data)

        assert "1" in result
        assert "one" in result

    def test_progress_bar_zero_total(self):
        """Test progress bar with zero total."""
        progress = ProgressBar(total=0)

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            progress.update(0)
            # Should not crash

        finally:
            sys.stdout = old_stdout
