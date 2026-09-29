"""Regenerate all figures and check them against the inventory.

Fails if the figure file names differ from those listed in
docs/figure_inventory.md (the thesis references them by name), or if
raw identifiers leak into the rendered figure text.

Usage:
    python -m scripts.check_figures
"""

import re
import subprocess
import sys
from pathlib import Path

from pdfminer.high_level import extract_text


FIGURES_DIR = Path("figures")
PDF_DIR = FIGURES_DIR / "pdf"
INVENTORY_PATH = Path("docs/figure_inventory.md")

# Raw identifiers and unformatted numbers that must never appear in a
# figure (the label helpers in plotstyle.py exist to prevent this).
# Captions live in LaTeX, so only the drawn text is checked.
FORBIDDEN_SUBSTRINGS = [
    "lora_target", "learning_rate", "rank_alpha", "subset_size",
    "training_regime", "iso_epoch", "iso_step", "is_collapsed",
    "src/analysis",
    "0.0001", "0.0003", "0.0005",
    "1e-05", "3e-05", "5e-05",
]

# Word-boundary match, so words such as "symmetric" do not trigger it.
FORBIDDEN_WORD_PATTERN = re.compile(r"\bmetric\b", re.IGNORECASE)


def _expected_filenames():
    """Return the figure file names listed in the inventory table.

    Reads the markdown directly, so the document itself is checked
    rather than plots.py's FIGURES registry.
    """
    text = INVENTORY_PATH.read_text()

    filenames = set(re.findall(r"\b(fig_[A-Za-z0-9_]+\.pdf)\b", text))

    assert filenames, f"no filenames found in {INVENTORY_PATH} -- check its table format"

    return filenames


def _actual_filenames():
    return {p.name for p in PDF_DIR.glob("*.pdf")}


def _check_text_leaks():
    """Return (filename, text) pairs for forbidden text in the PDFs."""
    violations = []

    for path in sorted(PDF_DIR.glob("*.pdf")):

        text = extract_text(path)

        for forbidden in FORBIDDEN_SUBSTRINGS:
            if forbidden in text:
                violations.append((path.name, forbidden))

        if FORBIDDEN_WORD_PATTERN.search(text):
            violations.append((path.name, "metric"))

    return violations


def main():

    print("Regenerating every figure...")

    result = subprocess.run(
        [sys.executable, "-m", "src.analysis.plots", "--all"],
        check=False,
    )

    if result.returncode != 0:
        print("FAIL -- figure regeneration itself failed, see output above.")
        sys.exit(1)

    expected = _expected_filenames()
    actual = _actual_filenames()

    added = actual - expected
    removed = expected - actual

    if added or removed:

        print("FAIL -- figures/ does not match docs/figure_inventory.md:")

        if added:
            print(f"  added (not in inventory): {sorted(added)}")

        if removed:
            print(f"  missing (in inventory, not regenerated): {sorted(removed)}")

        sys.exit(1)

    print("Checking for label leaks (raw identifiers, hand-formatted numbers)...")

    violations = _check_text_leaks()

    if violations:

        print("FAIL -- forbidden text leaked into rendered figures:")

        for filename, offending in violations:
            print(f"  {filename}: {offending!r}")

        sys.exit(1)

    print(f"OK -- all {len(expected)} figure filenames match docs/figure_inventory.md exactly.")
    print("OK -- no label leaks found in any figure.")


if __name__ == "__main__":
    main()
