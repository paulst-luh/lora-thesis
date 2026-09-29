"""Generate figures/figures.tex with one LaTeX figure block per figure.

Each block has the graphic, a short caption, optional TODO comments
and a label of the form fig:<id> without the "fig_" prefix (e.g.
fig_6_3a -> fig:6_3a).

Caption policy: one sentence on what is plotted, one on the takeaway,
then only the caveats needed to read the figure. Longer notes go into
CAPTION_TODOS, written as LaTeX comments (invisible in the PDF) that
name the thesis section they belong to.

Captions are hard-coded, so re-running is safe; only captions citing
live run counts can go stale.

Usage:
    python -m src.analysis.figures_tex
"""

import datetime
import glob
import os
from pathlib import Path


FIGURES_DIR = Path("figures")
PDF_DIR = FIGURES_DIR / "pdf"
OUTPUT_PATH = FIGURES_DIR / "figures.tex"

# (figure_id, chapter heading) in thesis order. Headings are written
# as LaTeX comments only, to keep the generated file navigable.
FIGURE_ORDER = [
    ("fig_5_1", "Chapter 5 -- Implementation"),

    ("fig_6_1a", "Chapter 6.1 -- Strategy Comparison"),
    ("fig_6_1e", None),
    ("fig_6_1b", None),
    ("fig_6_1c", None),
    ("__TABLE_6_1d__", None),

    ("fig_6_2a", "Chapter 6.2 -- Efficiency"),
    ("fig_6_2b", None),
    ("fig_6_2c", None),

    ("fig_6_3a", "Chapter 6.3 -- Dataset Properties and Configuration (RQ1)"),
    ("fig_6_3a_lr3e4", None),
    ("fig_6_3c", None),
    ("fig_6_3d", None),
    ("fig_6_3e", None),

    ("fig_6_4", "Chapter 6.4 -- Configuration Transfer (RQ2)"),
    ("fig_6_4d", None),
    ("fig_6_4e", None),
    ("fig_6_4f", None),

    ("fig_6_5a_mnli", "Chapter 6.5 -- Optimization Settings (RQ3)"),
    ("fig_6_5a_squad", None),
    ("fig_6_5a_conll2003", None),
    ("fig_6_5a_gsm8k", None),
    ("fig_6_5b", None),

    ("fig_6_6a", "Chapter 6.6 -- Hyperparameter Importance"),
    ("fig_6_6b", None),
    ("fig_6_6c", None),
    ("fig_6_6d", None),

    ("fig_7_1a", "Chapter 7.1 -- Practical Guidelines"),
    ("fig_7_1b", None),

    ("fig_7_2a", "Chapter 7.2 -- Relation to Prior Work"),
    ("fig_7_2b", None),

    ("fig_7_3a", "Chapter 7.3 -- Limitations"),
    ("fig_7_3b", None),

    ("fig_A1_mnli_qv", "Appendix A1 -- Mean-metric heatmaps (all 12)"),
    ("fig_A1_mnli_attention", None),
    ("fig_A1_mnli_full", None),
    ("fig_A1_squad_qv", None),
    ("fig_A1_squad_attention", None),
    ("fig_A1_squad_full", None),
    ("fig_A1_conll2003_qv", None),
    ("fig_A1_conll2003_attention", None),
    ("fig_A1_conll2003_full", None),
    ("fig_A1_gsm8k_qv", None),
    ("fig_A1_gsm8k_attention", None),
    ("fig_A1_gsm8k_full", None),

    ("fig_A2", "Appendix A2 -- Sub-metrics"),
    ("fig_A3", "Appendix A3 -- Training loss curves"),
    ("fig_A3_squad", None),
    ("fig_A3_conll2003", None),
    ("fig_A3_gsm8k", None),
    ("fig_A5", "Appendix A5 -- Design coverage"),
]

CAPTIONS = {

    "fig_5_1":
        "Realized optimizer steps against subset size for the two ladder training "
        "schedules, MNLI and SQuAD. \\texttt{iso\\_step} holds steps flat at 938 "
        "regardless of subset size, while \\texttt{iso\\_epoch} scales steps linearly "
        "with it (189 at 1k to 9375 at 50k) -- this is what lets later dataset-size "
        "effects be attributed to data volume or step count separately, not both at "
        "once.",

    "fig_6_1a":
        "Peak and typical performance per adaptation strategy, one panel per dataset: "
        "light dot = grid mean, star = the best-performing configuration's own mean "
        "with a $\\pm1$ seed-SD error bar; the zero-shot mean is stated as text in "
        "whichever corner is clear of the data, and the $y$-axis is scaled to the "
        "plotted strategies' own range rather than stretched down to zero-shot, so "
        "the four strategies stay directly distinguishable. There is "
        "no consistent peak-performance winner across datasets, but \\texttt{qv}'s mean "
        "sits closest to its own peak everywhere -- its advantage is consistency, not "
        "capability. FFT's grid is 3 configurations (one per learning rate) against 30 "
        "per LoRA target, a smaller search budget behind its own star.",

    "fig_6_1e":
        "Score distribution per strategy per dataset: each point is one "
        "configuration's mean over its 3 seeds (n=30 per LoRA target, n=3 for FFT, "
        "hatched), median and IQR overlaid, dashed line = zero-shot mean. Panels with "
        "a wide gap between the healthy and collapsed clusters (MNLI, SQuAD, "
        "CoNLL-2003) use an explicit broken axis rather than clipping, so the low tails "
        "-- MNLI's degenerate-predictor floor at 0.1714, SQuAD \\texttt{full} reaching "
        "0.0000 -- stay visible rather than compressed against a healthy-only range.",

    "fig_6_1b":
        "Full-fine-tuning stability across seeds, one panel per dataset, using the "
        "pooled 10-seed diagnostic sample at each learning rate; dashed line = "
        "zero-shot mean. On MNLI, $10^{-5}$ is "
        "10/10 healthy (mean 0.879, SD 0.008), while $3\\times10^{-5}$ and "
        "$5\\times10^{-5}$ collapse 1/10 and 2/10 seeds respectively (marked "
        "\\texttimes) -- a failure rate a 3-seed sample could not have revealed.",

    "fig_6_1c":
        "Share of collapsed seeds by strategy and dataset, with 95\\% Wilson-score "
        "confidence intervals. FFT is shown at both 3 and 10 seeds per learning-rate "
        "cell: a 0-share estimate from 3 seeds and one from 10 seeds are not the same "
        "claim about reliability, even at equal height, which the very different CI "
        "widths make visible -- qv/attention/full remain at their core 3-seeds-per-cell "
        "grid.",

    "fig_6_2a":
        "Trainable parameters (log scale) against mean per-configuration metric, "
        "colored by target, FFT as a star, with the Pareto frontier (best metric at or "
        "under a given parameter budget) as a step line and dashed line = zero-shot "
        "mean. Across roughly four orders of "
        "magnitude (qv rank 4's 1.26M parameters to FFT's 7.6B), the achievable metric "
        "barely moves -- what grows with parameter count is the spread, not the "
        "ceiling.",

    "fig_6_2b":
        "Runtime and peak GPU memory by strategy, one column per dataset; the y-axis is "
        "shared within each row so costs are directly comparable across datasets. "
        "Zero-shot is omitted from the runtime row (it never trains, so runtime is "
        "undefined, not zero) but included in the peak-memory row, where it has a "
        "real recorded value -- the eval-only forward pass still loads the model, "
        "around 14.2--14.3GB across all four datasets, essentially the base model's "
        "own footprint before any optimizer or gradient state is added. "
        "Wall-clock time is hardware-dependent -- comparable within this study only, "
        "not as a general LoRA/FFT claim.",

    "fig_6_2c":
        "Runtime, peak memory and trainable-parameter cost of each LoRA target relative "
        "to FFT $=1.0$, averaged across datasets (top: runtime/memory; bottom: "
        "trainable parameters on their own $\\sim$0--1\\% scale, plotted separately "
        "since they are 60--900$\\times$ smaller and invisible on the top panel's "
        "axis). Attention costs roughly 65\\% of FFT's runtime and 60\\% of its memory "
        "for a mean result that matches or beats FFT's on two of four datasets "
        "(MNLI, GSM8K); the parameter saving is far larger still.",

    "fig_6_3a":
        "Metric against training-set size (log scale), one line per rank, at learning "
        "rate $10^{-4}$, faceted by dataset (rows) and training regime (columns; "
        "shared $y$-axis within each row); the zero-shot mean is stated as text in "
        "each panel's corner rather than drawn as a line, so the $y$-axis stays "
        "scaled to the curves' own range. At this "
        "learning rate, rank 64 does not "
        "degrade with subset size -- it is flat to mildly improving; the rank-64 "
        "instability only appears once $3\\times10^{-4}$ is included "
        "(Figure~\\ref{fig:6_3a_lr3e4}), i.e.\\ it is a learning-rate$\\times$rank "
        "interaction, not a pure rank effect.",

    "fig_6_3a_lr3e4":
        "Same as Figure~\\ref{fig:6_3a}, at learning rate $3\\times10^{-4}$ -- the "
        "condition under which rank 64's instability actually appears, driven by "
        "discrete seed collapses to the degenerate one-label-predictor value 0.171372 "
        "rather than a smooth capacity effect (seed-level detail in "
        "Figure~\\ref{fig:6_3d}).",

    "fig_6_3c":
        "Best-performing rank per rung: for each (dataset, regime, learning rate) row "
        "and subset-size column, the winning rank is annotated and cell color encodes "
        "its gap to the runner-up. Hatched cells mark where that margin is smaller "
        "than the winning rank's own seed-to-seed SD -- across most of the grid, "
        "``the best rank'' is not a claim the 3-seed data can support.",

    "fig_6_3d":
        "Individual seed points (no averaging) at rank 64 across the four ladder "
        "rungs, colored by learning rate, faceted by dataset and regime; dashed line "
        "= zero-shot mean. Confirms that "
        "SQuAD's dip-and-recovery pattern under iso-epoch is a single seed (44) at "
        "$3\\times10^{-4}$ partially collapsing at 5k/20k and recovering at 50k -- not "
        "a smooth effect present in every seed or an artifact of averaging.",

    "fig_6_3e":
        "The full-target ladder overlaid on the original attention-target ladder "
        "(color = rank, solid = attention, dashed = full), same 2x2 grid and "
        "learning\\_rate$=10^{-4}$ restriction as Figure~\\ref{fig:6_3a}, zero-shot "
        "mean stated as corner text the same way. Neither "
        "target collapses at this learning rate in either regime, and full tracks "
        "attention closely rank-for-rank -- the collapse gap between the two "
        "targets that context doc \\S8.2 reports is concentrated at "
        "$3\\times10^{-4}$, outside this restricted view.",

    "fig_6_4":
        "Transfer regret between the four datasets' best configurations: (a) full "
        "configuration transferred as-is; (b) learning rate re-selected on the "
        "target; (c) a$-$b, the loss attributable to learning rate alone. Color is a "
        "single scale shared across all three panels, directly proportional to "
        "regret -- a darker cell always means a larger regret, regardless of which "
        "target column it sits in. Regret values are each dataset's own bounded-score "
        "points, not the same underlying quantity across columns, so this is a rough "
        "visual proxy rather than a strict statistical comparison; the annotated "
        "numbers carry the exact values regardless of color. Panel c's MNLI column is "
        "exactly zero for three of four sources -- re-tuning the learning rate only "
        "helps when transferring GSM8K's winner into MNLI (0.015).",

    "fig_6_4d":
        "Percentile rank of each source dataset's best configuration within the "
        "target dataset's own 90-configuration distribution -- metric-scale "
        "independent, so (unlike Figure~\\ref{fig:6_4}) comparable across the whole "
        "grid, not just within one column. CoNLL-2003 is the hardest transfer target "
        "(7th--39th percentile depending on source); every dataset transfers into "
        "itself at the 100th percentile by construction.",

    "fig_6_4e":
        "Each dataset's own best configuration on a shared [0,1] score axis (a "
        "bounded proportion for all four metrics, though not the same quantity), "
        "annotated with the exact (target, rank, $\\alpha$, learning rate) tuple; "
        "dashed vertical line = zero-shot mean. No "
        "two datasets share the same full configuration, directly answering RQ2 -- "
        "though individual factors do recur (\\texttt{full} wins twice, as do the "
        "(rank, $\\alpha$) pairs (4, 4) and (8, 16); all four winners share learning "
        "rate $1\\times10^{-4}$).",

    "fig_6_4f":
        "Transfer regret against a leave-one-out global default, one panel per "
        "target dataset (own y-axis -- regret is in the target's own metric, not "
        "comparable across panels). Circles are the three cross-dataset transfer "
        "regrets (each source's own winning configuration, transferred unchanged), "
        "colored by source dataset with the same palette as "
        "Figure~\\ref{fig:6_2a}; the grey diamond is the configuration minimizing "
        "mean regret over the other three datasets, selected without using the "
        "target's own data at all. Shaded band = 0 to the target's own winner's "
        "seed SD (the noise floor a smaller regret can't be distinguished from); "
        "dashed line = the median regret across the full 90-configuration grid. "
        "$y$-axis clipped at 0.035: the full grid's regret tail runs far higher (up "
        "to 0.72 on MNLI, driven by collapsed configurations), but no point "
        "plotted here exceeds 0.030.",

    "fig_6_5a_mnli":
        "MNLI: metric against learning rate, one line per (rank, $\\alpha$) pair "
        "(color = rank, dashed = ratio 1, solid = ratio 2), faceted by target "
        "module, grey dashed line = zero-shot mean; each panel's inset zooms to "
        "its own non-collapsed range (zero-shot mean omitted there, well below "
        "the zoomed range), since "
        "the full [0,1] axis needed to show collapse flattens the healthy band to "
        "a hairline. Line crossings inside the insets -- most visible in the "
        "attention and full panels -- show that rank ordering is not stable "
        "across learning rates, a genuine RQ3 learning-rate$\\times$rank "
        "interaction. Figures~\\ref{fig:6_5a_squad}--\\ref{fig:6_5a_gsm8k} show "
        "the same view for the other three datasets.",

    "fig_6_5a_squad":
        "SQuAD: same view as Figure~\\ref{fig:6_5a_mnli}.",

    "fig_6_5a_conll2003":
        "CoNLL-2003: same view as Figure~\\ref{fig:6_5a_mnli}.",

    "fig_6_5a_gsm8k":
        "GSM8K: same view as Figure~\\ref{fig:6_5a_mnli}.",

    "fig_6_5b":
        "Fraction of seeds collapsed for every (rank, $\\alpha$) $\\times$ "
        "learning-rate cell, one panel per target, columns grouped by dataset then "
        "learning rate; zero-collapse cells are left blank so the discrete 4-level "
        "legend and the risk pattern stay legible. Collapse concentrates in the "
        "high-rank, high-learning-rate corner of \\texttt{full} and, to a lesser "
        "extent, \\texttt{attention} -- GSM8K is clean everywhere, consistent with its "
        "markedly lower sensitivity (Figure~\\ref{fig:6_6a}).",

    "fig_6_6a":
        "Exact Shapley-value attribution for the main-sweep grid (contributors: "
        "target, learning rate, (rank,$\\alpha$)), one grouped bar per dataset. "
        "Learning rate and (rank,$\\alpha$) dominate on MNLI/SQuAD/GSM8K; target "
        "choice contributes comparatively little on its own everywhere, consistent "
        "with Figure~\\ref{fig:6_5b}'s finding that it matters mainly in combination "
        "with rank and learning rate.",

    "fig_6_6b":
        "Pairwise Grabisch-Roubens interaction indices for the same three-contributor "
        "game, one $3\\times3$ symmetric panel per dataset (diagonal = main effect, "
        "off-diagonal = interaction, values $\\times10^{-3}$), color normalized "
        "independently per panel. The (learning rate, rank$\\alpha$) pair is the "
        "largest-magnitude off-diagonal term on MNLI, SQuAD and CoNLL-2003.",

    "fig_6_6c":
        "Same Shapley attribution, dataset-size ladder (MNLI/SQuAD only), "
        "contributors: rank, subset size, learning rate, training regime. Subset "
        "size dominates on both datasets by a wide margin -- formal confirmation that "
        "dataset size is the primary driver of ladder outcomes.",

    "fig_6_6d":
        "Rank and scaling ratio as separate Shapley contributors, over all five ranks "
        "$\\{4,8,16,32,64\\}$, each run at both scaling ratios 1 and 2, so rank and "
        "ratio vary independently; target and learning rate are held fixed at qv, "
        "$3\\times10^{-4}$. SQuAD's rank contribution is exactly zero (annotated) -- "
        "verified directly: within this slice the fixed point's rank 8 is already "
        "SQuAD's best rank at either ratio, so optimizing rank never improves on it.",

    "fig_7_1a":
        "Expected best result under a search budget of $k$ configurations, "
        "bootstrapped from the already-recorded main-sweep grid (1000 draws without "
        "replacement per $k$, no new training), one line per target, faceted by "
        "dataset, band = interquartile range; the zero-shot mean is stated as text "
        "in each panel's bottom-right corner rather than drawn as a line. On "
        "MNLI, SQuAD and CoNLL-2003 all three "
        "targets saturate within 2--3 draws; on GSM8K, qv reaches a durably higher "
        "ceiling than attention or full even at the full budget of 30. Even a "
        "single random draw already clears the zero-shot mean by a wide margin "
        "on every dataset.",

    "fig_7_1b":
        "Grid mean against grid best metric, one point per (target, dataset), "
        "identity line for reference. Points well below the diagonal have a high "
        "ceiling but an unreliable typical outcome -- \\texttt{full} sits furthest "
        "below the diagonal on MNLI and GSM8K, the clearest single-picture summary of "
        "the reliability-versus-capability trade-off.",

    "fig_7_2a":
        "Best learning rate per strategy and dataset (log scale; strategies dodged "
        "horizontally so coincident values stay visible), with a shaded band from "
        "FFT's own best LR up to 10$\\times$ that value. LoRA's best learning rates "
        "fall inside or just above this band on most datasets, broadly consistent "
        "with Schulman's reported $\\sim$9.8$\\times$ factor -- though both ranges are "
        "only three-valued and disjoint by design, so this shows consistency with a "
        "$\\sim$10$\\times$ factor, not an estimate of it.",

    "fig_7_2b":
        "Metric against rank for the qv target (this study's most reliable target, "
        "uncontaminated by the collapse noise affecting attention/full at high rank), "
        "two lines: fixed learning rate ($3\\times10^{-4}$) versus the per-rank best "
        "learning rate, faceted by dataset. Re-tuning the learning rate flattens but "
        "does not eliminate the apparent rank effect -- partial support for why rank "
        "effects appear in some studies and vanish in others.",

    "fig_7_3a":
        "95\\% CI width as a function of seed count $k=1..10$, bootstrapped from "
        "MNLI's full-fine-tuning 10-seed sample at its safe learning rate "
        "($10^{-5}$). At $k=3$ -- this thesis's main-sweep seed count -- the CI width "
        "is 0.0174 ($\\pm1.7$ points of Macro-F1 even at FFT's most reliable "
        "setting), dropping to 0.0095 by $k=10$: quantitative justification for why "
        "the 3-seed main-sweep cells carry limited statistical weight.",

    "fig_7_3b":
        "Within-cell standard deviation over seeds, one point per core-grid cell, "
        "median as a bar, grouped by dataset; collapsed runs excluded so the figure "
        "measures run-to-run noise rather than collapse rate. MNLI's scoring is fully "
        "deterministic given fixed weights, so its spread reflects seed variability "
        "alone; the three generative datasets' spread additionally includes "
        "generation-time nondeterminism. CoNLL-2003 is, contrary to a naive "
        "expectation, the cleanest dataset in the study, and GSM8K the noisiest.",

    "fig_A1_mnli_qv":
        "Mean metric, MNLI, qv target: learning rate $\\times$ (rank,$\\alpha$), full "
        "resolution, one of twelve heatmaps (dataset $\\times$ target) making up "
        "Appendix~A1. Green marks a high (good) score, red a low (bad) score, "
        "yellow/orange between -- a deliberate exception to this study's usual "
        "colorblind-safe palette, so the exact value is also printed in every cell "
        "regardless of color. Color is shared across this dataset's three target "
        "panels (qv/attention/full), not normalized separately per panel, so the "
        "same color means the same score everywhere in the three; floored at the "
        "5th percentile of all three pooled together rather than the true min, so "
        "the healthy range stays legible. Hatched cells (with a colorbar arrow) are "
        "collapsed outliers below that pooled floor. See "
        "Figure~\\ref{fig:6_5b} for the corresponding collapse-fraction view of the "
        "same cells.",
    "fig_A1_mnli_attention": "Mean metric, MNLI, attention target -- see Figure~\\ref{fig:A1_mnli_qv}.",
    "fig_A1_mnli_full": "Mean metric, MNLI, full target -- see Figure~\\ref{fig:A1_mnli_qv}.",
    "fig_A1_squad_qv": "Mean metric, SQuAD, qv target -- see Figure~\\ref{fig:A1_mnli_qv}.",
    "fig_A1_squad_attention": "Mean metric, SQuAD, attention target -- see Figure~\\ref{fig:A1_mnli_qv}.",
    "fig_A1_squad_full": "Mean metric, SQuAD, full target -- see Figure~\\ref{fig:A1_mnli_qv}.",
    "fig_A1_conll2003_qv": "Mean metric, CoNLL-2003, qv target -- see Figure~\\ref{fig:A1_mnli_qv}.",
    "fig_A1_conll2003_attention": "Mean metric, CoNLL-2003, attention target -- see Figure~\\ref{fig:A1_mnli_qv}.",
    "fig_A1_conll2003_full": "Mean metric, CoNLL-2003, full target -- see Figure~\\ref{fig:A1_mnli_qv}.",
    "fig_A1_gsm8k_qv": "Mean metric, GSM8K, qv target -- see Figure~\\ref{fig:A1_mnli_qv}.",
    "fig_A1_gsm8k_attention": "Mean metric, GSM8K, attention target -- see Figure~\\ref{fig:A1_mnli_qv}.",
    "fig_A1_gsm8k_full": "Mean metric, GSM8K, full target -- see Figure~\\ref{fig:A1_mnli_qv}.",

    "fig_A2":
        "Sub-metrics not present in the aggregated CSV schema (read directly from "
        "each result JSON's raw metrics): precision/recall/F1 (+accuracy for MNLI) "
        "for MNLI and CoNLL-2003, exact-match versus F1 for SQuAD, mean over each "
        "strategy's core grid. GSM8K has no sub-metric concept beyond exact-match "
        "accuracy and is absent from this figure for that reason, not by omission.",

    "fig_A3":
        "Representative training-loss curves, MNLI, one of four datasets making up "
        "Appendix~A3, one panel per block with real "
        "training (FFT at $10^{-5}$; qv/attention/full at rank 8, $\\alpha=16$, "
        "learning rate $3\\times10^{-4}$), one line per seed. All four blocks "
        "converge to a similarly small residual loss well before training ends, with "
        "no visible divergence in optimization behavior across strategies.",
    "fig_A3_squad": "Training-loss curves, SQuAD -- see Figure~\\ref{fig:A3}. Same pattern: all four blocks converge cleanly, no visible divergence.",
    "fig_A3_conll2003": "Training-loss curves, CoNLL-2003 -- see Figure~\\ref{fig:A3}. Same pattern: all four blocks converge cleanly, no visible divergence.",
    "fig_A3_gsm8k": "Training-loss curves, GSM8K -- see Figure~\\ref{fig:A3}. Same pattern: all four blocks converge cleanly, no visible divergence.",

    "fig_A5":
        "Design coverage: run count per block and dataset, full results tree. The "
        "core seven-block design (zeroshot/fft/main\\_qv/main\\_attention/main\\_full/"
        "ladder\\_isoepoch/ladder\\_isostep) has no missing cell (1{,}608 runs total), "
        "and every diagnostic follow-up round is complete as well, including the "
        "risky-seed reruns (70 of 70 runs) and the full-target ladder (480 of 480).",

}

# figure_id -> TODO lines, each written as a "% TODO ..." comment under
# the figure's caption.
CAPTION_TODOS = {

    "fig_6_1b": [
        "TODO(\\S5.3 or wherever the diagnostic seed-extension rounds are "
        "described): the pooled 10-seed sample here combines the "
        "fft\\_wd0.1 and fft\\_wd0.1\\_moreseeds blocks (both "
        "weight\\_decay$=0.1$) -- state that provenance in the methods "
        "text, not this caption.",
    ],

    "fig_6_1c": [
        "TODO(\\S7.3 Limitations): note that the risky-seed diagnostic "
        "extending selected attention/full cells to 10 seeds is complete "
        "(70 of 70 runs), and describe it there rather than in this "
        "caption.",
    ],

    "fig_6_3c": [
        "TODO: color currently shares one raw-gap scale across MNLI "
        "(rows 1--4, accuracy) and SQuAD (rows 5--8, F1) deltas -- "
        "flagged as comparing two different quantities on one colorbar. "
        "Update this caption once the pending per-dataset-norm vs "
        "SD-normalized-color decision is implemented.",
    ],

    "fig_6_6a": [
        "TODO(\\S5.3 Reproducibility): move here -- exact Shapley/"
        "interaction values (src/analysis/shapley.py) were "
        "cross-validated against the independent shapiq library to "
        "floating-point precision for all three games (main sweep, "
        "ladder, rank/ratio) before any figure in this section was "
        "built.",
        "TODO: pending decision on normalizing each dataset's bars to "
        "its own total Shapley mass (sum$=1$, raw values moved to the "
        "A4 appendix table) and on adding bootstrap CIs (1000 "
        "seed-resampled draws). Update this caption once that lands -- "
        "bars shown now are raw, un-normalized, no CI.",
    ],

    "fig_6_6b": [
        "TODO: bootstrap CIs (1000 seed-resampled draws, reported in "
        "chat 2026-08-05) show only MNLI's (learning\\_rate, "
        "rank\\_alpha) interaction excludes zero at 95\\%; SQuAD's and "
        "CoNLL-2003's largest-magnitude terms named above do NOT survive "
        "bootstrap. Add CI bars/shading per the pending decision and "
        "revise this caption's claim accordingly -- do not reintroduce "
        "``negative in all three'' as a finding without the CI attached. "
        "Likely \\S7.3 content: the main-sweep interaction estimates are "
        "underpowered at $n=3$ seeds.",
    ],

    "fig_6_6c": [
        "TODO: bootstrap CIs (reported in chat 2026-08-05) confirm all "
        "eight bars here exclude zero at 95\\%, unlike "
        "Figure~\\ref{fig:6_6b} -- add CI bars per the pending decision "
        "for consistency with 6.19/6.20; the ranking claim above already "
        "holds up without them.",
    ],

    "fig_6_6d": [
        "TODO(\\S4.5): Chapter 4 does not currently name a default "
        "configuration -- this caption states the held-fixed values "
        "(qv, $3\\times10^{-4}$) directly rather than calling them "
        "``the default,'' since Figure~\\ref{fig:6_4e} shows all four "
        "dataset winners at $1\\times10^{-4}$, not $3\\times10^{-4}$. If "
        "\\S4.5 later defines a named default, this caption (and "
        "Figure~\\ref{fig:A3}'s) can reference it by name instead.",
    ],

    "fig_A3": [
        "TODO(\\S4.5): see Figure~\\ref{fig:6_6d}'s TODO -- same "
        "undefined-``default configuration'' issue; this caption states "
        "the rank/$\\alpha$/LR values directly instead.",
        "TODO(\\S5.3 Reproducibility): loss-curve files are matched to "
        "their metrics file by closest file-modification time, not a "
        "shared run ID, because save\\_loss\\_history() and "
        "save\\_results() (src/train.py) each independently call the "
        "same time\\_ns()+pid suffix generator a few milliseconds "
        "apart. This is fixable properly, not just documentable: both "
        "calls already run in the same function/scope with the same "
        "config, so a real join needs only one run\\_id computed once "
        "and threaded into both save calls "
        "($\\sim$10--15 lines in src/utils/results.py + src/train.py) "
        "-- assessed as a SMALL change, not involved. The "
        "mtime-matching fallback would still need to stay permanently "
        "for the $\\sim$2{,}670 already-saved legacy file pairs, which "
        "have no ID to retroactively embed.",
    ],

}


_ALL_FIGURE_IDS = [fid for fid, _ in FIGURE_ORDER if not fid.startswith("__TABLE_")]

# Draft figures that have a PDF but are not in FIGURE_ORDER. _find_pdf
# needs them so that e.g. fig_7_2b_v2_*.pdf is not taken for a file of
# fig_7_2b.
_NON_THESIS_FIGURE_IDS = ["fig_7_2b_v2"]


def _find_pdf(figure_id):
    """Return the PDF file name of `figure_id`.

    Some ids are prefixes of others (fig_6_3a, fig_6_3a_lr3e4), so each
    file is assigned to its longest matching id.
    """
    candidates = sorted(glob.glob(str(PDF_DIR / f"{figure_id}_*.pdf")))
    assert candidates, f"no PDF found for {figure_id!r} in {PDF_DIR} -- generate it first"

    resolved = []

    for path in candidates:

        name = os.path.basename(path)
        owning_id = max(
            (fid for fid in _ALL_FIGURE_IDS + _NON_THESIS_FIGURE_IDS if name.startswith(fid + "_")),
            key=len,
        )

        if owning_id == figure_id:
            resolved.append(path)

    assert resolved, f"no PDF resolves to {figure_id!r} specifically (all matches belong to a more specific id)"
    assert len(resolved) == 1, f"ambiguous match for {figure_id!r}: {resolved}"

    return os.path.basename(resolved[0])


def render_figure_block(figure_id):

    pdf_name = _find_pdf(figure_id)
    label = figure_id.removeprefix("fig_")
    caption = CAPTIONS[figure_id]
    todos = CAPTION_TODOS.get(figure_id, [])

    # No width option: save_figure() makes every PDF exactly text width,
    # so the fonts keep their intended size.
    lines = [
        "\\begin{figure}[htbp]",
        "\\centering",
        f"\\includegraphics{{figures/pdf/{pdf_name}}}",
        f"\\caption{{{caption}}}",
    ]

    # TODO lines are LaTeX comments, so they are hidden in the PDF.
    for todo in todos:
        lines.append(f"% {todo}")

    lines.append(f"\\label{{fig:{label}}}")
    lines.append("\\end{figure}\n")

    return "\n".join(lines)


def main():

    missing = [
        fid for fid, _ in FIGURE_ORDER
        if not fid.startswith("__TABLE_") and fid not in CAPTIONS
    ]
    assert not missing, f"no caption written for: {missing}"

    blocks = [
        f"% Generated by src/analysis/figures_tex.py on {datetime.date.today().isoformat()}.\n"
        "% results/ is a live tree (diagnostic SLURM jobs complete over time), but\n"
        "% captions and TODOs below are hardcoded editorial content in this script,\n"
        "% not recomputed from results/ -- re-running this generator is safe and\n"
        "% will NOT overwrite hand-considered wording. The one exception: any\n"
        "% caption that cites an exact run-completion count from a live results/\n"
        "% tree goes stale as more runs land; those have been moved out of\n"
        "% captions and into TODO comments pointing at \\S7.3 instead, precisely\n"
        "% so a stale number can't sit in reader-facing text unnoticed.\n"
    ]

    for figure_id, heading in FIGURE_ORDER:

        if heading:
            blocks.append(f"\n% --- {heading} ---\n")

        # Slot of the former fig_6_1d, now a table from tables.py.
        if figure_id == "__TABLE_6_1d__":
            blocks.append(
                "% Generated separately by src/analysis/tables.py (a table, not a figure).\n"
                "\\input{figures/table_6_1d_parse_failure.tex}\n"
            )
            continue

        blocks.append(render_figure_block(figure_id))

    # The A4 table is generated by tables.py and only included here.
    blocks.append(
        "\n% --- Appendix A4 -- Runtime and memory table ---\n"
        "% Generated separately by src/analysis/tables.py (a table, not a figure).\n"
        "\\input{figures/table_A4_runtime_memory.tex}\n"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(blocks))

    print(f"wrote {OUTPUT_PATH} ({len(_ALL_FIGURE_IDS)} figures)")


if __name__ == "__main__":
    main()
