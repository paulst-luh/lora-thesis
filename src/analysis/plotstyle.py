"""Shared matplotlib styling for all thesis figures.

Every figure function calls apply_style() before plotting and
save_figure() to write the figure.

Size contract: every figure is created at figsize=(TEXTWIDTH_IN, h)
with h <= MAX_FIG_HEIGHT_IN and saved without a cropping bbox, so the
PDF is exactly TEXTWIDTH_IN wide. Include it in LaTeX at natural size
(no width, scale or height option) so the fonts keep their size.
"""

import colorsys
import math
import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


# The only place paths and thesis layout numbers may be hard-coded.

FIGURES_DIR = Path("figures")
# PDFs for LaTeX and PNGs for quick review, in separate subfolders.
PDF_DIR = FIGURES_DIR / "pdf"
PNG_DIR = FIGURES_DIR / "png"

# \textwidth and \textheight of the thesis document, in TeX points
# (72.27 per inch).
_TEXTWIDTH_PT = 390.0
_TEXTHEIGHT_PT = 592.0
_PT_PER_IN = 72.27
TEXTWIDTH_IN = _TEXTWIDTH_PT / _PT_PER_IN
TEXTHEIGHT_IN = _TEXTHEIGHT_PT / _PT_PER_IN

# Height reserved for the caption and float spacing, so a figure never
# pushes its caption off the page.
_CAPTION_AND_FLOAT_SPACE_IN = 1.2
MAX_FIG_HEIGHT_IN = TEXTHEIGHT_IN - _CAPTION_AND_FLOAT_SPACE_IN

# The PDF MediaBox uses PostScript points (72 per inch), not TeX
# points; save_figure() checks against this.
_PS_PT_PER_IN = 72.0

BASE_FONT_SIZE_PT = 9

# Okabe-Ito colorblind-safe palette, fixed for the whole study.
STRATEGY_COLORS = {
    "zeroshot": "#000000",   # black
    "fft": "#D55E00",        # vermillion
    "qv": "#0072B2",         # blue
    "attention": "#009E73",  # bluish green
    "full": "#CC79A7",       # reddish purple
}

# Sequential colormap for the LoRA ranks (dark = low, light = high).
RANK_VALUES = [4, 8, 16, 32, 64]
RANK_CMAP = plt.get_cmap("viridis")

# Stop at 0.9: the top of viridis is a near-white yellow that is hard
# to read on a white background.
_RANK_CMAP_MAX_POSITION = 0.9


def rank_color(rank):
    """Return the color of `rank`, spaced evenly over RANK_VALUES.

    The ranks double at each step, so the index (not the value) sets
    the position, capped at _RANK_CMAP_MAX_POSITION.
    """
    if rank not in RANK_VALUES:
        raise ValueError(
            f"rank {rank!r} is not one of the study's rank values {RANK_VALUES}"
        )

    position = RANK_VALUES.index(rank) / (len(RANK_VALUES) - 1) * _RANK_CMAP_MAX_POSITION

    return RANK_CMAP(position)


def adjust_lightness(color, amount):
    """Return `color` with its lightness scaled by `amount`.

    Values below 1 darken, above 1 lighten. Hue and saturation stay, so
    two lines of the same rank can differ by shade only.
    """
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l * amount))

    return colorsys.hls_to_rgb(h, l, s)


# Label contract: no figure formats a reader-facing label by hand.
# Identifiers, learning rates, subset sizes and metric names all go
# through PRETTY/pretty(), fmt_lr(), fmt_subset() or METRIC_NAMES.
# scripts/check_figures.py fails if raw identifiers reach a figure.

PRETTY = {

    "lora_target": "Target modules",
    "learning_rate": "Learning rate",
    "rank_alpha": r"($r$, $\alpha$)",
    "subset_size": "Training-set size",
    "training_regime": "Training schedule",
    "iso_epoch": "Iso-epoch",
    "iso_step": "Iso-step",
    # Terms fixed by the thesis terminology; "FFT" and "Full LoRA"
    # share no word, so they cannot be mixed up.
    "qv": "Query-Value LoRA",
    "attention": "Attention LoRA",
    "full": "Full LoRA",
    "fft": "FFT",
    "zeroshot": "Zero-shot",
    "rank": "Rank r",
    "scaling_ratio": r"Scaling ratio $\alpha$/$r$",

    # Cost columns (fig_6_2c) and sub-metric keys (fig_A2).
    "runtime_s": "Runtime",
    "peak_mem_gb": "Peak memory",
    "trainable_params": "Trainable params",
    "eval_accuracy": "Accuracy",
    "eval_precision": "Precision",
    "eval_recall": "Recall",
    "eval_f1": "F1",
    "eval_exact_match": "Exact match",

    # Outlook models, used only by outlook_plots.py.
    "llama-3.1-8b-instruct": "Llama 3.1 8B Instruct",
    "deepseek-r1-distill-qwen-7b": "DeepSeek-R1-Distill-Qwen-7B",
    "gpt-oss-20b": "GPT-OSS-20B",
    "gemma-4-e4b-it": "Gemma 4 E4B",
    "mistral-7b-instruct-v0.3": "Mistral 7B Instruct v0.3",

}


def pretty(identifier):
    """Return the reader-facing label of an internal identifier.

    Looks up PRETTY first; anything else (e.g. diagnostic block names)
    falls back to a title-cased version with spaces.
    """
    if identifier in PRETTY:
        return PRETTY[identifier]

    return str(identifier).replace("_", " ").replace(".", " ").title()


# Each dataset has its own primary metric, so an axis is never labeled
# with a bare "Metric".
METRIC_NAMES = {
    "mnli": "Macro-F1",
    "squad": "F1",
    "conll2003": "F1",
    "gsm8k": "Exact match",
}


def fmt_lr(value):
    """Format a learning rate as mathtext (mantissa times 10^exponent).

    Raises ValueError unless the value is a single-digit mantissa times
    a negative power of ten, instead of printing e.g. "9.999e-05". The
    split uses Python's exact decimal formatting, since log10() of a
    float such as 1e-4 is not exactly -4.
    """
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"learning rate must be a positive number, got {value!r}")

    mantissa_str, exponent_str = f"{value:.6e}".split("e")
    exponent = int(exponent_str)
    mantissa = float(mantissa_str)
    mantissa_int = round(mantissa)

    if not (1 <= mantissa_int <= 9) or not math.isclose(mantissa, mantissa_int, abs_tol=1e-6):
        raise ValueError(
            f"{value!r} does not reduce to a single-digit integer mantissa times "
            f"a power of ten (got mantissa={mantissa_str}, exponent={exponent})"
        )

    if exponent >= 0:
        raise ValueError(
            f"fmt_lr expects a learning rate < 1 (negative exponent), got "
            f"exponent={exponent} for value={value!r}"
        )

    return f"${mantissa_int} \\times 10^{{{exponent}}}$"


# Numerical zero of the HyperSHAP values, as in
# scripts/export_hypershap_values.py (shapiq's least-squares residuals).
NUMERICAL_ZERO = 1e-8


def fmt_e3(value):
    """Format a value in units of 10^-3 with three decimals.

    |value| <= NUMERICAL_ZERO gives "0"; a value that rounds to 0.000
    gives "<0.001". "-0.000" is never returned.
    """
    if abs(value) <= NUMERICAL_ZERO:
        return "0"
    text = f"{value * 1e3:.3f}"
    if float(text) == 0:
        return "<0.001"
    return text


def fmt_subset(n):
    """Return a compact subset-size label such as "5k".

    Full numbers overlapped when rotated at the figure font size.
    """
    n = int(n)

    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"

    return str(n)


# Palette (tab10) for all non-strategy categories, e.g. Shapley players,
# cost columns or ladder learning rates. STRATEGY_COLORS is reserved
# for the five strategies and shares no color with this palette.
CATEGORICAL_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def categorical_color(identifiers):
    """Map identifiers to CATEGORICAL_PALETTE colors in list order.

    Pass the same ordered list everywhere, so an identifier keeps its
    color across figures.
    """
    if len(identifiers) > len(CATEGORICAL_PALETTE):
        raise ValueError(
            f"{len(identifiers)} categories requested but CATEGORICAL_PALETTE "
            f"only has {len(CATEGORICAL_PALETTE)} colors"
        )

    return dict(zip(identifiers, CATEGORICAL_PALETTE))


def text_color_for_rgba(rgba):
    """Return "white" or "black", whichever is legible on `rgba`.

    For cells whose color does not come from a single (cmap, norm)
    pair; otherwise use annot_color().
    """
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]

    return "white" if luminance < 0.5 else "black"


def annot_color(cell_value, cmap, norm):
    """Return a legible annotation color for a heatmap cell.

    Based on the luminance of each cell's own color, not one fixed
    choice per figure.
    """
    return text_color_for_rgba(cmap(norm(cell_value)))


def categorical_axis(ax, values, formatter, axis="x"):
    """Place discrete `values` at positions 0..n-1 with custom labels.

    For values a continuous or log axis shows badly, e.g. learning
    rates within one decade. Returns {value: position}; plot against
    positions[value] instead of the raw value.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    positions = {value: i for i, value in enumerate(values)}
    labels = [formatter(value) for value in values]

    if axis == "x":
        ax.set_xticks(list(positions.values()))
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.5, len(values) - 0.5)
    else:
        ax.set_yticks(list(positions.values()))
        ax.set_yticklabels(labels)
        ax.set_ylim(-0.5, len(values) - 0.5)

    return positions


def apply_style():
    """Set the rcParams shared by all thesis figures.

    Call once before creating a figure; calling it again is harmless.
    Figures must not set titles, since captions live in LaTeX.
    """
    plt.rcParams.update({

        # Computer Modern (cmr10, shipped with matplotlib) for text and
        # math, so labels match the thesis font and fmt_lr() output.
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        # cmr10 has no glyph for the Unicode minus sign.
        "axes.unicode_minus": False,
        "font.size": BASE_FONT_SIZE_PT,

        "axes.labelsize": BASE_FONT_SIZE_PT,
        "xtick.labelsize": BASE_FONT_SIZE_PT - 1,
        "ytick.labelsize": BASE_FONT_SIZE_PT - 1,
        "legend.fontsize": BASE_FONT_SIZE_PT - 1,

        # Titles are not used, but keep their size consistent anyway.
        "axes.titlesize": BASE_FONT_SIZE_PT,

        # constrained_layout keeps the given figsize exactly, whereas a
        # tight bbox would crop it. Figure legends therefore use
        # loc="outside upper center" to get their space reserved.
        "figure.constrained_layout.use": True,

        "savefig.dpi": 300,

    })


def _pdf_mediabox_size_in(pdf_path):
    """Return (width, height) in inches from the PDF's MediaBox.

    Read from the saved file, not fig.get_size_inches(), so the check
    in save_figure() tests what was actually written.
    """
    data = Path(pdf_path).read_bytes()
    match = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", data)

    assert match, f"no /MediaBox found in {pdf_path}"

    x0, y0, x1, y1 = (float(v) for v in match.groups())

    return (x1 - x0) / _PS_PT_PER_IN, (y1 - y0) / _PS_PT_PER_IN


def save_figure(fig, figure_id, slug):
    """Save `fig` as PDF (for LaTeX) and PNG (for quick review).

    Writes figures/pdf/<figure_id>_<slug>.pdf and the matching PNG
    without cropping, then checks the saved PDF: its width must equal
    TEXTWIDTH_IN and its height must not exceed MAX_FIG_HEIGHT_IN.

    Returns (pdf_path, png_path).
    """
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    basename = f"{figure_id}_{slug}"

    pdf_path = PDF_DIR / f"{basename}.pdf"
    png_path = PNG_DIR / f"{basename}.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path)

    width_in, height_in = _pdf_mediabox_size_in(pdf_path)
    tolerance_in = 0.5 / _PS_PT_PER_IN

    assert abs(width_in - TEXTWIDTH_IN) <= tolerance_in, (
        f"{figure_id}: saved PDF width {width_in:.4f}in does not match "
        f"TEXTWIDTH_IN {TEXTWIDTH_IN:.4f}in (tolerance {tolerance_in:.4f}in) -- "
        f"check this figure's figsize= and that nothing overrides "
        f"constrained layout with a tight/cropping bbox"
    )
    assert height_in <= MAX_FIG_HEIGHT_IN + tolerance_in, (
        f"{figure_id}: saved PDF height {height_in:.4f}in exceeds "
        f"MAX_FIG_HEIGHT_IN {MAX_FIG_HEIGHT_IN:.4f}in -- this figure needs "
        f"a shorter figsize or a layout redesign (fewer panels/rows), not "
        f"a rescale"
    )

    return pdf_path, png_path
