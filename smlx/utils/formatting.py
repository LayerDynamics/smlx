"""
Formatting utilities for pretty console output.

Provides table formatting, progress bars, and color utilities.
"""

import sys
from typing import Any, Optional


def format_table(
    headers: list[str],
    rows: list[list[Any]],
    align: Optional[list[str]] = None,
) -> str:
    """
    Format data as a simple ASCII table.

    Args:
        headers: Column headers
        rows: Data rows
        align: Alignment for each column ('left', 'right', 'center')
               Defaults to 'left' for all columns

    Returns:
        Formatted table string

    Example:
        >>> headers = ["Model", "TPS", "Memory"]
        >>> rows = [["SmolLM2", "123.45", "2.3 GB"]]
        >>> print(format_table(headers, rows))
    """
    if not rows:
        return ""

    if align is None:
        align = ["left"] * len(headers)

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Create format strings for each column
    def format_cell(text: str, width: int, alignment: str) -> str:
        text = str(text)
        if alignment == "right":
            return text.rjust(width)
        elif alignment == "center":
            return text.center(width)
        else:
            return text.ljust(width)

    # Build table
    lines = []

    # Header
    header_line = " | ".join(format_cell(h, col_widths[i], align[i]) for i, h in enumerate(headers))
    lines.append(header_line)

    # Separator
    separator = "-+-".join("-" * w for w in col_widths)
    lines.append(separator)

    # Rows
    for row in rows:
        row_line = " | ".join(
            format_cell(row[i] if i < len(row) else "", col_widths[i], align[i])
            for i in range(len(headers))
        )
        lines.append(row_line)

    return "\n".join(lines)


def format_dict_table(data: list[dict[str, Any]], keys: Optional[list[str]] = None) -> str:
    """
    Format a list of dictionaries as a table.

    Args:
        data: list of dictionaries
        keys: Keys to include (in order). If None, uses keys from first dict.

    Returns:
        Formatted table string

    Example:
        >>> data = [{"name": "model1", "tps": 100}, {"name": "model2", "tps": 150}]
        >>> print(format_dict_table(data))
    """
    if not data:
        return ""

    if keys is None:
        keys = list(data[0].keys())

    headers = keys
    rows = [[str(d.get(k, "")) for k in keys] for d in data]

    return format_table(headers, rows)


def format_key_value(data: dict[Any, Any], indent: int = 0) -> str:
    """
    Format dictionary as key-value pairs.

    Args:
        data: dictionary to format (keys will be converted to strings)
        indent: Indentation level

    Returns:
        Formatted string

    Example:
        >>> data = {"model": "SmolLM2", "tps": 123.45}
        >>> print(format_key_value(data))
        model: SmolLM2
        tps: 123.45
    """
    if not data:
        return ""

    indent_str = " " * indent
    max_key_len = max(len(str(k)) for k in data.keys())

    lines = []
    for key, value in data.items():
        key_str = str(key).ljust(max_key_len)
        lines.append(f"{indent_str}{key_str}: {value}")

    return "\n".join(lines)


def format_header(text: str, width: int = 80, char: str = "=") -> str:
    """
    Format a header with surrounding characters.

    Args:
        text: Header text
        width: Total width
        char: Character to use for border

    Returns:
        Formatted header

    Example:
        >>> print(format_header("Benchmark Results"))
        ==================== Benchmark Results ====================
    """
    text = f" {text} "
    padding = width - len(text)
    left_pad = padding // 2
    right_pad = padding - left_pad
    return char * left_pad + text + char * right_pad


def format_section(title: str, content: str) -> str:
    """
    Format a section with title and content.

    Args:
        title: Section title
        content: Section content

    Returns:
        Formatted section

    Example:
        >>> print(format_section("Results", "TPS: 123.45"))
    """
    lines = [
        "",
        format_header(title, width=60, char="-"),
        content,
        "",
    ]
    return "\n".join(lines)


class ProgressBar:
    """
    Simple progress bar for console output.

    Example:
        >>> progress = ProgressBar(total=100)
        >>> for i in range(100):
        ...     progress.update(i + 1)
        >>> progress.close()
    """

    def __init__(
        self,
        total: int,
        prefix: str = "",
        suffix: str = "",
        length: int = 50,
        fill: str = "█",
    ):
        """
        Initialize progress bar.

        Args:
            total: Total iterations
            prefix: Prefix string
            suffix: Suffix string
            length: Bar length in characters
            fill: Fill character
        """
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.length = length
        self.fill = fill
        self.current = 0

    def update(self, current: int):
        """Update progress bar to current value."""
        self.current = current
        self._print()

    def _print(self):
        """Print the progress bar."""
        if self.total == 0:
            percent = 100.0
            filled_length = self.length  # Full bar when total is 0
        else:
            percent = 100 * (self.current / float(self.total))
            filled_length = int(self.length * self.current // self.total)

        bar = self.fill * filled_length + "-" * (self.length - filled_length)

        sys.stdout.write(f"\r{self.prefix} |{bar}| {percent:.1f}% {self.suffix}")
        sys.stdout.flush()

    def close(self):
        """Close the progress bar (print newline)."""
        sys.stdout.write("\n")
        sys.stdout.flush()


def colorize(text: str, color: str) -> str:
    """
    Add ANSI color codes to text.

    Args:
        text: Text to colorize
        color: Color name ('red', 'green', 'yellow', 'blue', 'magenta', 'cyan')

    Returns:
        Colorized text (if terminal supports it)

    Example:
        >>> print(colorize("Success!", "green"))
    """
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "bold": "\033[1m",
    }
    reset = "\033[0m"

    # Check if stdout is a terminal
    if not sys.stdout.isatty():
        return text

    color_code = colors.get(color.lower(), "")
    if color_code:
        return f"{color_code}{text}{reset}"
    return text


def format_comparison(
    data: list[dict[str, Any]],
    baseline_key: str = "baseline",
    compare_keys: Optional[list[str]] = None,
) -> str:
    """
    Format comparison data showing relative differences.

    Args:
        data: list of benchmark results
        baseline_key: Key in dict identifying baseline
        compare_keys: Keys to compare

    Returns:
        Formatted comparison table

    Example:
        >>> data = [
        ...     {"name": "float16", "tps": 100, "baseline": True},
        ...     {"name": "4-bit", "tps": 150, "baseline": False}
        ... ]
        >>> print(format_comparison(data))
    """
    if not data:
        return ""

    # Find baseline
    baseline = None
    for item in data:
        if item.get(baseline_key):
            baseline = item
            break

    if baseline is None:
        # No baseline specified, use first item
        baseline = data[0]

    if compare_keys is None:
        # Use numeric keys
        compare_keys = [k for k, v in baseline.items() if isinstance(v, (int, float))]

    # Build comparison table
    headers = ["Name"] + compare_keys + ["vs Baseline"]
    rows = []

    for item in data:
        row = [item.get("name", "")]

        for key in compare_keys:
            value = item.get(key, 0)
            row.append(f"{value:.2f}" if isinstance(value, float) else str(value))

        # Calculate speedup
        if item == baseline:
            row.append("1.00x (baseline)")
        else:
            # Use first compare key for speedup calculation
            baseline_val = baseline.get(compare_keys[0], 1)
            item_val = item.get(compare_keys[0], 1)
            if baseline_val > 0:
                speedup = item_val / baseline_val
                row.append(f"{speedup:.2f}x")
            else:
                row.append("N/A")

        rows.append(row)

    return format_table(headers, rows)
