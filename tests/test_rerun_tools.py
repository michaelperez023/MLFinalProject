"""The helpers of scripts/desktop_rerun.sh, on a fake dataset archive and fake run outputs."""

import hashlib
import http.server
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import zipfile

import pandas as pd
import pytest
from helpers import REPO, write_jpeg_dataset

pytestmark = pytest.mark.skipif(sys.version_info < (3, 10), reason="the runbook uses Python 3.12")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = load_script("fetch_dataset")
check = load_script("check_2019_results")
summarize = load_script("summarize_rerun")

COUNTS = {"train/NORMAL": 3, "train/PNEUMONIA": 5, "test/NORMAL": 2, "test/PNEUMONIA": 2}


@pytest.fixture
def archive(tmp_path):
    """A small zip laid out like ChestXRay2017.zip, with its macOS metadata entries."""
    images = write_jpeg_dataset(tmp_path / "src" / "chest_xray", COUNTS)
    path = tmp_path / "ChestXRay2017.zip"
    with zipfile.ZipFile(path, "w") as z:
        for file in sorted(images.rglob("*.jpeg")):
            name = "chest_xray/" + file.relative_to(images).as_posix()
            z.write(file, name)
            z.writestr("__MACOSX/" + name.replace(file.name, "._" + file.name), b"metadata")
        z.writestr("chest_xray/.DS_Store", b"x")
        z.writestr("chest_xray/train/", b"")
    return path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """
    Serves files and honours "Range: bytes=N-", like the S3 bucket behind Mendeley Data. With
    server.limit set, each response announces the rest of the file but sends at most that many
    bytes before the connection closes, without an error.
    """

    def do_GET(self):
        data = (self.server.root / self.path.lstrip("/")).read_bytes()
        start = int(self.headers.get("Range", "bytes=0-")[6:].split("-")[0] or 0)
        self.server.ranges.append(self.headers.get("Range"))
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        limit = self.server.limit
        self.wfile.write(data[start:] if limit is None else data[start : start + limit])

    def log_message(self, *args):
        pass


@pytest.fixture
def server(archive):
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    httpd.root = archive.parent
    httpd.limit = None
    httpd.ranges = []
    httpd.url = f"http://127.0.0.1:{httpd.server_address[1]}/{archive.name}"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def test_download_checks_and_resumes(archive, server, tmp_path):
    target = tmp_path / "downloads" / "ChestXRay2017.zip"
    target.parent.mkdir()
    # An interrupted earlier download: the first half of the file
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(archive.read_bytes()[: archive.stat().st_size // 2])
    fetch.download(server.url, target, archive.stat().st_size, sha256(archive))
    assert target.read_bytes() == archive.read_bytes()
    assert not partial.exists()
    # A finished download is not repeated
    stamp = target.stat().st_mtime_ns
    fetch.download(server.url, target, archive.stat().st_size, sha256(archive))
    assert target.stat().st_mtime_ns == stamp


def test_download_continues_after_the_connection_closes_early(archive, server, tmp_path):
    # Without an error: read() just returns no more data
    size = archive.stat().st_size
    server.limit = size // 3 + 1
    target = tmp_path / "downloads" / "ChestXRay2017.zip"
    fetch.download(server.url, target, size, sha256(archive), retry_wait=0)
    assert target.read_bytes() == archive.read_bytes()
    assert server.ranges == [None, f"bytes={server.limit}-", f"bytes={2 * server.limit}-"]


def test_download_that_stops_adding_data_keeps_the_partial_file(archive, server, tmp_path):
    size = archive.stat().st_size
    target = tmp_path / "downloads" / "ChestXRay2017.zip"
    target.parent.mkdir()
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(archive.read_bytes()[: size // 2])
    server.limit = 0
    with pytest.raises(SystemExit, match="3 attempts in a row added nothing"):
        fetch.download(server.url, target, size, sha256(archive), attempts=3, retry_wait=0)
    assert partial.stat().st_size == size // 2 and len(server.ranges) == 3
    # The next run resumes it
    server.limit = None
    fetch.download(server.url, target, size, sha256(archive), retry_wait=0)
    assert target.read_bytes() == archive.read_bytes() and not partial.exists()


def test_download_with_the_wrong_checksum_fails_and_cleans_up(archive, server, tmp_path):
    target = tmp_path / "downloads" / "ChestXRay2017.zip"
    with pytest.raises(SystemExit, match="does not match"):
        fetch.download(server.url, target, archive.stat().st_size, "0" * 64)
    assert not target.exists() and not target.with_name(target.name + ".part").exists()
    # A download longer than expected is deleted too
    with pytest.raises(SystemExit, match="does not match"):
        fetch.download(server.url, target, archive.stat().st_size - 1, sha256(archive))
    assert not target.exists() and not target.with_name(target.name + ".part").exists()


def test_extract_skips_macos_metadata_and_counts_images(archive, tmp_path):
    info = fetch.extract(archive, tmp_path / "data", "abc", COUNTS, {"doi": fetch.DOI})
    images = tmp_path / "data" / "chest_xray"
    assert sorted(
        p.relative_to(images).as_posix() for p in images.rglob("*") if p.is_file()
    ) == sorted(
        p.relative_to(tmp_path / "src" / "chest_xray").as_posix()
        for p in (tmp_path / "src" / "chest_xray").rglob("*.jpeg")
    )
    assert (
        info["counts"] == COUNTS
        and info["modes"]["RGB"] >= 1
        and info["doi"] == "10.17632/rscbjbr9sj.2"
    )
    assert json.loads((tmp_path / "data" / "chest_xray.json").read_text()) == info
    # A finished extraction is not repeated
    (images / "marker").write_text("kept")
    fetch.extract(archive, tmp_path / "data", "abc", COUNTS)
    assert (images / "marker").exists()


def test_extract_stops_on_unexpected_counts(archive, tmp_path):
    with pytest.raises(SystemExit, match="Unexpected image counts"):
        fetch.extract(archive, tmp_path / "data", "abc", dict(COUNTS, **{"test/NORMAL": 234}))
    assert not (tmp_path / "data" / "chest_xray").exists()


def test_extract_rejects_unsafe_paths(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("chest_xray/../../escape.jpeg", b"x")
    with pytest.raises(SystemExit, match="Unsafe path"):
        fetch.extract(path, tmp_path / "data", "abc", {})


def test_fetch_main_end_to_end(archive, server, tmp_path):
    fetch.main(
        [
            "--downloads",
            str(tmp_path / "dl"),
            "--data-dir",
            str(tmp_path / "data"),
            "--url",
            server.url,
            "--size",
            str(archive.stat().st_size),
            "--sha256",
            sha256(archive),
            "--expected-counts",
            json.dumps(COUNTS),
        ]
    )
    info = json.loads((tmp_path / "data" / "chest_xray.json").read_text())
    assert info["archive_sha256"] == sha256(archive) and info["file"] == "ChestXRay2017.zip"


def write_results_xlsx(path, columns):
    table = pd.DataFrame(
        {name: values + [sum(values) / len(values)] for name, values in columns.items()},
        index=list(range(1, len(next(iter(columns.values()))) + 1)) + ["AVG"],
    )
    table.to_excel(path, sheet_name="Results", index_label="Test_Batch")


def test_check_2019_results(tmp_path):
    write_results_xlsx(tmp_path / "expected.xlsx", {"A": [0.5, 1.0], "B": [0.25, 0.25]})
    reproduced = {
        "A": {"test_images": 8, "correct": 6, "avg_partition_accuracy": 0.75},
        "B": {"test_images": 8, "correct": 3, "avg_partition_accuracy": 0.375},
    }
    (tmp_path / "rerun.json").write_text(json.dumps(reproduced))
    check.main(
        [
            "--expected",
            str(tmp_path / "expected.xlsx"),
            "--reproduced",
            str(tmp_path / "rerun.json"),
            "--out",
            str(tmp_path / "check.json"),
        ]
    )
    result = json.loads((tmp_path / "check.json").read_text())
    assert result["models"]["A"]["reproduced"] and result["models"]["A"]["expected_correct"] == 6
    assert not result["models"]["B"]["reproduced"] and not result["all_reproduced"]
    with pytest.raises(SystemExit):
        check.main(
            [
                "--expected",
                str(tmp_path / "expected.xlsx"),
                "--reproduced",
                str(tmp_path / "rerun.json"),
                "--out",
                str(tmp_path / "check.json"),
                "--strict",
            ]
        )


def test_the_real_2019_results_are_read():
    expected = check.read_avg_row(REPO / "Results" / "Test_Results.xlsx")
    assert len(expected) == 16
    assert round(expected["SVMClassifier_1"] * 624) == 498
    assert all(expected[f"CNN_{n}layers_1epoch"] == 0.625 for n in (2, 3, 4))


def fake_run(runs, conv_blocks, seed, labels, normal_probability):
    from retrain_2026 import binary_metrics

    folder = runs / f"cnn{conv_blocks}_seed{seed}"
    folder.mkdir(parents=True)
    metrics = {
        "conv_blocks": conv_blocks,
        "seed": seed,
        "image_size": 256,
        "settings": {"max_epochs": 40},
        "epochs_run": 3,
        "best_epoch": 2,
        "test": binary_metrics(labels, normal_probability),
        "history": {"val_auc": [0.8, 0.9, 0.85], "val_loss": [0.6, 0.4, 0.5]},
    }
    (folder / "metrics.json").write_text(json.dumps(metrics))
    rows = ["file,label,normal_probability"] + [
        f"x{i}.jpeg,{y},{p}"
        for i, (y, p) in enumerate(zip(labels, normal_probability, strict=True))
    ]
    (folder / "test_scores.csv").write_text("\n".join(rows) + "\n")


def test_summary_and_figures(tmp_path):
    pytest.importorskip("matplotlib")
    work = tmp_path / "work"
    (work / "results").mkdir(parents=True)
    (work / "logs").mkdir()
    (work / "data").mkdir()
    (work / "data" / "chest_xray.json").write_text(
        json.dumps(
            {
                "counts": {
                    "train/NORMAL": 1349,
                    "train/PNEUMONIA": 3883,
                    "test/NORMAL": 234,
                    "test/PNEUMONIA": 390,
                }
            }
        )
    )
    (work / "results" / "check_2019.json").write_text(
        json.dumps(
            {
                "all_reproduced": True,
                "tolerance": 1e-6,
                "models": {
                    "SVMClassifier_1": {
                        "expected_avg": 0.798,
                        "reproduced_avg": 0.798,
                        "reproduced": True,
                    }
                },
            }
        )
    )
    (work / "results" / "benchmark_2019_sklearn.json").write_text(
        json.dumps({"SVMClassifier_1": {"accuracy": 0.798, "sensitivity": 0.9, "specificity": 0.6}})
    )
    (work / "logs" / "settings.json").write_text(
        json.dumps({"seeds": [1, 2], "conv_blocks": [2, 4], "max_epochs": 40, "image_size": 256})
    )
    (work / "logs" / "timings.tsv").write_text(
        "dataset\t0\t60\t60\tok\npreprocess\t0\t0\t0\tskipped\n"
    )
    labels = [0, 0, 0, 1, 1, 1]
    for conv_blocks in (2, 4):
        for seed in (1, 2, 3):  # seed 3 is not in the settings, so it is left out
            fake_run(work / "runs", conv_blocks, seed, labels, [0.1, 0.4, 0.6, 0.7, 0.8, 0.3])

    summarize.main(["--work-dir", str(work), "--repo", str(REPO)])
    summary = json.loads((work / "results" / "rerun_2026.json").read_text())
    assert summary["baseline_always_pneumonia_accuracy"] == pytest.approx(0.625)
    assert summary["reproduction_2019"]["models"]["SVMClassifier_1"]["sensitivity"] == 0.9
    assert summary["timings_seconds"] == {"dataset": 60}
    by_depth = summary["cnn_2026"]["by_conv_blocks"]
    assert sorted(by_depth) == ["2", "4"] and by_depth["2"]["runs"] == 2
    assert by_depth["2"]["accuracy"]["mean"] == pytest.approx(4 / 6)
    assert len(summary["cnn_2026"]["runs"]) == 4
    for figure in summary["figures"]:
        assert (work / "results" / figure).stat().st_size > 10_000


def test_runbook_is_valid_bash():
    subprocess.run(["bash", "-n", str(REPO / "scripts" / "desktop_rerun.sh")], check=True)
    if shutil.which("shellcheck"):
        subprocess.run(
            ["shellcheck", "--severity=warning", str(REPO / "scripts" / "desktop_rerun.sh")],
            check=True,
        )


def run_runbook(work_dir, **env):
    return subprocess.run(
        ["bash", str(REPO / "scripts" / "desktop_rerun.sh"), str(work_dir)],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )


def test_runbook_refuses_a_work_dir_inside_the_repository(tmp_path):
    result = run_runbook(REPO / "work")
    assert result.returncode == 2 and "must not be inside the repository" in result.stderr
    assert not (REPO / "work").exists()
    # Also through a symbolic link, without creating the folder first
    (tmp_path / "link").symlink_to(REPO)
    created = REPO / "rerun-through-link"
    try:
        result = run_runbook(tmp_path / "link" / created.name)
        assert result.returncode == 2 and "must not be inside the repository" in result.stderr
        assert not created.exists()
    finally:
        if created.is_dir():
            created.rmdir()


def test_runbook_needs_a_new_or_empty_work_dir(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "notes.txt").write_text("kept")
    result = run_runbook(home)
    assert result.returncode == 2 and "must be a new or empty folder" in result.stderr
    assert [p.name for p in home.iterdir()] == ["notes.txt"]

    # A new folder is marked as the script's own and accepted again later; the run then stops
    # here because there is no conda at CONDA_EXE
    work = tmp_path / "new" / "work"
    for _ in range(2):
        result = run_runbook(work, CONDA_EXE=str(tmp_path / "no-conda"))
        assert result.returncode == 1 and "conda not found" in result.stderr
        assert (work / ".project-marrow-rerun").is_file()
