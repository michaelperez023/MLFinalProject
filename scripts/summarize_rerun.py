"""
Collects the outputs of scripts/desktop_rerun.sh into WORK_DIR/results/rerun_2026.json and
draws the figures in WORK_DIR/results/figures.

    python scripts/summarize_rerun.py --work-dir WORK_DIR --repo REPO_DIR

Every input is optional, so it also summarizes a run that stopped part of the way.
"""

import argparse
import csv
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

# Colors and ink of the figures
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

PY38_PACKAGES = ["numpy", "scipy", "scikit-learn", "pillow", "pandas", "openpyxl"]
PY312_PACKAGES = [
    "numpy",
    "scipy",
    "scikit-learn",
    "pillow",
    "pandas",
    "openpyxl",
    "h5py",
    "keras",
    "torch",
    "matplotlib",
]
SUMMARY_METRICS = ["accuracy", "sensitivity", "specificity", "roc_auc"]


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def read_timings(path):
    """Seconds of the last completed run of each step of the runbook."""
    timings = {}
    path = Path(path)
    if not path.exists():
        return timings
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 5 and parts[4] == "ok":
            timings[parts[0]] = int(parts[3])
    return timings


def read_runs(runs_dir, settings=None):
    """The retrain runs; with the runbook's settings, only those it asked for."""
    runs = []
    for metrics_path in sorted(Path(runs_dir).glob("cnn*_seed*/metrics.json")):
        run = json.loads(metrics_path.read_text())
        if settings and (
            run["conv_blocks"] not in settings["conv_blocks"]
            or run["seed"] not in settings["seeds"]
            or run["image_size"] != settings["image_size"]
            or run["settings"]["max_epochs"] != settings["max_epochs"]
        ):
            continue
        scores_path = metrics_path.with_name("test_scores.csv")
        if scores_path.exists():
            with open(scores_path, newline="") as f:
                rows = list(csv.DictReader(f))
            run["_labels"] = [int(row["label"]) for row in rows]
            run["_normal_probability"] = [float(row["normal_probability"]) for row in rows]
        runs.append(run)
    return sorted(runs, key=lambda run: (run["conv_blocks"], run["seed"]))


def describe(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def by_depth(runs):
    summary = {}
    for depth in sorted({run["conv_blocks"] for run in runs}):
        selected = [run for run in runs if run["conv_blocks"] == depth]
        summary[str(depth)] = {
            "runs": len(selected),
            "seeds": [run["seed"] for run in selected],
            **{name: describe([run["test"][name] for run in selected]) for name in SUMMARY_METRICS},
        }
    return summary


def select_versions(env, packages):
    if not env:
        return None
    found = {"python": env.get("python")}
    found.update({name: env["packages"].get(name) for name in packages if name in env["packages"]})
    for key in ("torch_cuda", "cuda_available", "gpu", "platform"):
        if key in env:
            found[key] = env[key]
    return found


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_figure(width, height, **kwargs):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(width, height), facecolor=SURFACE, **kwargs)
    return figure, axes


def save(figure, path):
    figure.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)


def accuracy_figure(rows, baseline, path):
    """Horizontal bars of test accuracy, 2019 models and 2026 CNNs, with the baseline."""
    height = 0.32 * len(rows) + 1.4
    figure, ax = new_figure(7.5, height)
    style(ax)
    ax.grid(axis="y", visible=False)
    labels = [row["label"] for row in rows][::-1]
    values = [100 * row["accuracy"] for row in rows][::-1]
    colors = [SERIES[0] if row["group"] == "2019" else SERIES[1] for row in rows][::-1]
    positions = range(len(rows))
    ax.barh(positions, values, height=0.6, color=colors)
    for position, value in zip(positions, values, strict=True):
        ax.text(
            value + 0.4, position, f"{value:.1f}%", va="center", fontsize=8, color=INK_SECONDARY
        )
    ax.set_yticks(list(positions), labels, fontsize=8.5, color=INK)
    ax.axvline(100 * baseline, color=INK_SECONDARY, linewidth=1)
    ax.text(
        100 * baseline,
        len(rows) - 0.3,
        f" always pneumonia: {100 * baseline:.1f}%",
        fontsize=8,
        color=INK_SECONDARY,
        va="bottom",
    )
    ax.set_xlim(50, 102)
    ax.set_xlabel("Accuracy on the test split (%)", color=INK_SECONDARY, fontsize=9)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=SERIES[0], label="2019 models"),
            Patch(color=SERIES[1], label="2026 CNN retrain (mean over seeds)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07 * 7 / height),
        ncol=2,
        frameon=False,
        fontsize=8,
        labelcolor=INK,
    )
    save(figure, path)


def roc_figure(runs, path):
    from sklearn.metrics import roc_curve

    figure, ax = new_figure(5.2, 5)
    style(ax)
    depths = sorted({run["conv_blocks"] for run in runs})
    for index, depth in enumerate(depths):
        selected = [run for run in runs if run["conv_blocks"] == depth and "_labels" in run]
        for number, run in enumerate(selected):
            pneumonia = [label == 0 for label in run["_labels"]]
            score = [1.0 - p for p in run["_normal_probability"]]
            fpr, tpr, _ = roc_curve(pneumonia, score)
            mean_auc = statistics.fmean(r["test"]["roc_auc"] for r in selected)
            ax.plot(
                fpr,
                tpr,
                color=SERIES[index % len(SERIES)],
                linewidth=2,
                alpha=0.85,
                solid_capstyle="round",
                label=f"{depth} blocks (mean AUC {mean_auc:.3f})" if number == 0 else None,
            )
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel(
        "1 - specificity (normal images labelled pneumonia)", color=INK_SECONDARY, fontsize=9
    )
    ax.set_ylabel(
        "Sensitivity (pneumonia images labelled pneumonia)", color=INK_SECONDARY, fontsize=9
    )
    ax.set_title(
        "2026 CNN retrain: ROC curves on the test split", color=INK, fontsize=10, loc="left"
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    save(figure, path)


def confusion_figure(runs, path):
    depths = sorted({run["conv_blocks"] for run in runs})
    shown = [next(run for run in runs if run["conv_blocks"] == depth) for depth in depths]
    figure, axes = new_figure(
        2.9 * len(shown) + 0.8, 3.3, ncols=len(shown), squeeze=False, layout="constrained"
    )
    from matplotlib.colors import LinearSegmentedColormap

    colormap = LinearSegmentedColormap.from_list("blues", BLUES)
    for ax, run in zip(axes[0], shown, strict=True):
        matrix = run["test"]["confusion_matrix"]
        cells = [
            [
                matrix["true_pneumonia"]["predicted_pneumonia"],
                matrix["true_pneumonia"]["predicted_normal"],
            ],
            [
                matrix["true_normal"]["predicted_pneumonia"],
                matrix["true_normal"]["predicted_normal"],
            ],
        ]
        ax.imshow([[c / sum(row) for c in row] for row in cells], cmap=colormap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                share = cells[i][j] / sum(cells[i])
                ax.text(
                    j,
                    i,
                    f"{cells[i][j]}\n{share:.0%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#ffffff" if share > 0.5 else INK,
                )
        ax.set_xticks([0, 1], ["pneumonia", "normal"], fontsize=8, color=INK)
        ax.set_xlabel("Predicted", color=INK_SECONDARY, fontsize=8.5)
        if ax is axes[0][0]:
            ax.set_yticks([0, 1], ["pneumonia", "normal"], fontsize=8, color=INK)
            ax.set_ylabel("True", color=INK_SECONDARY, fontsize=8.5)
        else:
            ax.set_yticks([])
        ax.set_title(f"{run['conv_blocks']} blocks, seed {run['seed']}", color=INK, fontsize=9)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    figure.suptitle("2026 CNN retrain: test-split confusion matrices", color=INK, fontsize=10)
    save(figure, path)


def training_figure(runs, path):
    figure, axes = new_figure(9, 3.4, ncols=2)
    depths = sorted({run["conv_blocks"] for run in runs})
    for ax, key, title in (
        (axes[0], "val_auc", "Validation ROC AUC"),
        (axes[1], "val_loss", "Validation loss"),
    ):
        style(ax)
        for index, depth in enumerate(depths):
            for number, run in enumerate(r for r in runs if r["conv_blocks"] == depth):
                values = run["history"].get(key, [])
                ax.plot(
                    range(1, len(values) + 1),
                    values,
                    color=SERIES[index % len(SERIES)],
                    linewidth=2,
                    alpha=0.85,
                    label=f"{depth} blocks" if number == 0 else None,
                )
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Epoch", color=INK_SECONDARY, fontsize=9)
    axes[0].legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    save(figure, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize a desktop_rerun.sh run.")
    parser.add_argument("--work-dir", required=True, help="the WORK_DIR of desktop_rerun.sh")
    parser.add_argument("--repo", required=True, help="the repository folder")
    args = parser.parse_args(argv)
    work, repo = Path(args.work_dir), Path(args.repo)
    results = work / "results"
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    check = read_json(results / "check_2019.json")
    benchmark = {}
    for name in ("benchmark_2019_sklearn.json", "benchmark_2019_cnn.json"):
        benchmark.update(read_json(results / name) or {})
    reproduction = None
    if check:
        reproduction = {
            "all_reproduced": check["all_reproduced"],
            "tolerance": check["tolerance"],
            "models": {},
        }
        for model, comparison in check["models"].items():
            entry = dict(comparison)
            measured = benchmark.get(model, {})
            for key in ("accuracy", "sensitivity", "specificity", "roc_auc", "confusion_matrix"):
                if key in measured:
                    entry[key] = measured[key]
            reproduction["models"][model] = entry

    settings = read_json(work / "logs" / "settings.json")
    runs = read_runs(work / "runs", settings)
    dataset = read_json(work / "data" / "chest_xray.json")
    test_counts = (dataset or {}).get("counts", {})
    baseline = None
    if test_counts.get("test/PNEUMONIA") and test_counts.get("test/NORMAL"):
        pneumonia, normal = test_counts["test/PNEUMONIA"], test_counts["test/NORMAL"]
        baseline = pneumonia / (pneumonia + normal)

    summary = {
        "description": "Re-run of Project Marrow (CAP 4612, fall 2019): the saved 2019 models "
        "scored again on the official test split, and the 2019 CNN architecture retrained in 2026.",
        "repository_commit": read_text(work / "logs" / "repo_commit.txt"),
        "settings": settings,
        "machine": {
            "gpu": read_text(work / "logs" / "gpu.txt"),
            "cpu": read_text(work / "logs" / "cpu.txt"),
        },
        "dataset": dataset,
        "baseline_always_pneumonia_accuracy": baseline,
        "reproduction_2019": reproduction,
        "cnn_2026": {
            "runs": [
                {
                    key: value
                    for key, value in run.items()
                    if not key.startswith("_") and key != "history"
                }
                for run in runs
            ],
            "by_conv_blocks": by_depth(runs),
        }
        if runs
        else None,
        "timings_seconds": read_timings(work / "logs" / "timings.tsv"),
        "timings_note": "wall-clock seconds of the invocation that finished each step; a step "
        "resumed after an interruption counts only its last part (each run's own times are in "
        "cnn_2026.runs)",
        "versions": {
            "python38_env": select_versions(
                read_json(work / "logs" / "env_py38.json"), PY38_PACKAGES
            ),
            "python312_env": select_versions(
                read_json(work / "logs" / "env_py312.json"), PY312_PACKAGES
            ),
        },
        "figures": [],
    }

    if baseline is not None and (reproduction or runs):
        rows = []
        if reproduction:
            # The 2019 results as recorded in Results/Test_Results.xlsx
            for model in sorted(reproduction["models"], key=model_order):
                accuracy = reproduction["models"][model]["expected_avg"]
                rows.append({"label": model, "accuracy": accuracy, "group": "2019"})
        for depth, entry in by_depth(runs).items():
            rows.append(
                {
                    "label": f"CNN, {depth} blocks, retrained",
                    "accuracy": entry["accuracy"]["mean"],
                    "group": "2026",
                }
            )
        accuracy_figure(rows, baseline, figures / "test_accuracy.png")
        summary["figures"].append("figures/test_accuracy.png")
    if runs and all("_labels" in run for run in runs):
        roc_figure(runs, figures / "roc_2026.png")
        confusion_figure(runs, figures / "confusion_2026.png")
        training_figure(runs, figures / "training_2026.png")
        summary["figures"] += [
            "figures/roc_2026.png",
            "figures/confusion_2026.png",
            "figures/training_2026.png",
        ]

    temporary = results / "rerun_2026.json.tmp"
    temporary.write_text(json.dumps(summary, indent=1) + "\n")
    os.replace(temporary, results / "rerun_2026.json")
    print(f"Wrote {results / 'rerun_2026.json'} and {len(summary['figures'])} figures")
    if repo.exists():
        # Results/, not results/, which would be the same folder on macOS and Windows
        print(
            f"Copy {results}/rerun_2026.json and {figures} into {repo / 'Results' / '2026'} "
            "to publish them"
        )


def model_order(name):
    """NB, LOG, PER, SVM, CNN, then by epochs or layers (NBClassifier_1, CNN_2layers_1epoch)."""
    family = next(
        (
            i
            for i, prefix in enumerate(("NB", "LOG", "PER", "SVM", "CNN"))
            if name.startswith(prefix)
        ),
        9,
    )
    numbers = [
        int(part)
        for part in name.replace("layers", "_").replace("epoch", "").split("_")
        if part.isdigit()
    ]
    return family, numbers, name


def read_text(path):
    path = Path(path)
    return path.read_text().strip() if path.exists() else None


if __name__ == "__main__":
    main()
