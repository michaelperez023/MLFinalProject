"""
Downloads the chest X-ray images from Mendeley Data, checks the archive's size and SHA-256,
and extracts the chest_xray folder.

    python scripts/fetch_dataset.py --downloads DIR --data-dir DIR

Kermany, Zhang and Goldbaum (2018), "Labeled Optical Coherence Tomography (OCT) and Chest
X-Ray Images for Classification", Mendeley Data, V2, doi:10.17632/rscbjbr9sj.2, CC BY 4.0.
The checksum and size are the ones Mendeley Data publishes for ChestXRay2017.zip.

An interrupted download resumes where it stopped, and a finished, verified download or
extraction is not repeated. Writes DATA_DIR/chest_xray (the images) and
DATA_DIR/chest_xray.json (image counts, modes and sizes).
"""

import argparse
import hashlib
import http.client
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

DOI = "10.17632/rscbjbr9sj.2"
URL = (
    "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/"
    "f12eaf6d-6023-432f-acc9-80c9d7393433/file_downloaded"
)
FILE_NAME = "ChestXRay2017.zip"
SIZE = 1_235_512_464
SHA256 = "13efc055629733dbab07877f8b3c9f81097840dbcdaa326a8542322c2281ce36"
EXPECTED_COUNTS = {
    "train/NORMAL": 1349,
    "train/PNEUMONIA": 3883,
    "test/NORMAL": 234,
    "test/PNEUMONIA": 390,
}
CHUNK = 1 << 20


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, target, size, sha256, attempts=5, retry_wait=10):
    """
    Downloads url to target and checks its size and SHA-256.

    A download that breaks off, with an error or without one (a connection that closes early),
    continues from where it stopped. After `attempts` tries in a row that add nothing, it stops
    and keeps the partial file, so that the next run resumes it.
    """
    target = Path(target)
    if target.exists():
        if target.stat().st_size == size and sha256_of(target) == sha256:
            print(f"{target} is already downloaded and verified")
            return
        print(f"{target} does not match the expected size and SHA-256; downloading it again")
        target.unlink()
    partial = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    largest = partial.stat().st_size if partial.exists() else 0
    failures = 0
    while True:
        done = partial.stat().st_size if partial.exists() else 0
        if done >= size:
            break  # complete, or too long: the checks below decide
        # Mendeley Data answers 403 to Python's default User-Agent header
        request = urllib.request.Request(url, headers={"User-Agent": "project-marrow-rerun"})
        if done:
            request.add_header("Range", f"bytes={done}-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                resumed = done and getattr(response, "status", None) == 206
                mode = "ab" if resumed else "wb"
                done = done if resumed else 0
                print(f"Downloading {url} to {partial} (from byte {done:,} of {size:,})")
                start, last_report = time.time(), 0.0
                with open(partial, mode) as out:
                    # read() returns b"" when the connection closes early, without an error
                    for block in iter(lambda: response.read(CHUNK), b""):
                        out.write(block)
                        done += len(block)
                        if time.time() - last_report > 30:
                            last_report = time.time()
                            speed = done / 2**20 / (time.time() - start + 1e-9)
                            print(f"  {done / size:6.1%}  {speed:.1f} MiB/s")
            if done >= size:
                break
            problem = f"the connection closed at byte {done:,} of {size:,}"
        except (urllib.error.URLError, http.client.HTTPException, OSError) as err:
            problem = str(err)
        current = partial.stat().st_size if partial.exists() else 0
        failures = 0 if current > largest else failures + 1
        largest = max(largest, current)
        if failures >= attempts:
            raise SystemExit(
                f"Download failed: {attempts} attempts in a row added nothing ({problem}). "
                f"{partial} keeps the {current:,} bytes downloaded; run the script again to resume."
            )
        wait = retry_wait * max(failures, 1)
        print(f"Download interrupted ({problem}); continuing in {wait} s")
        time.sleep(wait)
    actual_size = partial.stat().st_size
    actual_sha256 = sha256_of(partial)
    if actual_size != size or actual_sha256 != sha256:
        partial.unlink()
        raise SystemExit(
            f"Downloaded file does not match: {actual_size:,} bytes, SHA-256 {actual_sha256}; "
            f"expected {size:,} bytes, SHA-256 {sha256}. It was deleted; run the script again."
        )
    os.replace(partial, target)
    print(f"Verified {target}: {size:,} bytes, SHA-256 {sha256}")


def safe_members(archive):
    """The archive's chest_xray/ entries, without macOS metadata files, checked for unsafe paths."""
    members = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
            raise SystemExit(f"Unsafe path in the archive: {info.filename!r}")
        if not path.parts or path.parts[0] != "chest_xray":
            continue  # __MACOSX/ holds macOS metadata
        if path.name.startswith("._") or path.name == ".DS_Store" or info.is_dir():
            continue
        members.append(info)
    return members


def extract(archive_path, data_dir, sha256, expected_counts, metadata=None):
    """Extracts chest_xray/ into data_dir and writes data_dir/chest_xray.json."""
    data_dir = Path(data_dir)
    images_dir = data_dir / "chest_xray"
    info_path = data_dir / "chest_xray.json"
    if images_dir.is_dir() and info_path.exists():
        info = json.loads(info_path.read_text())
        if info.get("archive_sha256") == sha256:
            print(f"{images_dir} is already extracted")
            return info
    temporary = data_dir / "chest_xray.tmp"
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.rmtree(images_dir, ignore_errors=True)
    temporary.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = safe_members(archive)
        for member in members:
            relative = PurePosixPath(member.filename).relative_to("chest_xray")
            destination = temporary.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, open(destination, "wb") as out:
                shutil.copyfileobj(source, out, CHUNK)
    info = dict(metadata or {})
    info.update(describe_images(temporary, expected_counts))
    info["archive_sha256"] = sha256
    os.replace(temporary, images_dir)
    temporary_info = info_path.with_name(info_path.name + ".tmp")
    temporary_info.write_text(json.dumps(info, indent=1) + "\n")
    os.replace(temporary_info, info_path)
    print(f"Extracted {sum(info['counts'].values())} images to {images_dir}")
    return info


def describe_images(images_dir, expected_counts):
    """Counts, color modes and sizes of the images; stops if the counts are unexpected."""
    from collections import Counter

    try:
        from PIL import Image
    except ImportError:
        Image = None
    counts, modes, widths, heights = {}, Counter(), [], []
    for folder in sorted(expected_counts):
        files = sorted((images_dir / folder).glob("*.jpeg"))
        counts[folder] = len(files)
        if Image is not None:
            for file in files:
                with Image.open(file) as image:  # reads only the header
                    modes[image.mode] += 1
                    widths.append(image.width)
                    heights.append(image.height)
    if expected_counts and counts != expected_counts:
        raise SystemExit(f"Unexpected image counts {counts}; expected {expected_counts}")
    info = {"counts": counts}
    if widths:
        info["modes"] = dict(sorted(modes.items()))
        info["width"] = {"min": min(widths), "max": max(widths)}
        info["height"] = {"min": min(heights), "max": max(heights)}
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--downloads", required=True, help="folder for the downloaded archive")
    parser.add_argument("--data-dir", required=True, help="folder to extract chest_xray/ into")
    parser.add_argument("--url", default=URL, help="download URL (default: Mendeley Data)")
    parser.add_argument(
        "--file-name", default=FILE_NAME, help=f"archive name (default: {FILE_NAME})"
    )
    parser.add_argument("--size", type=int, default=SIZE, help="expected archive size in bytes")
    parser.add_argument("--sha256", default=SHA256, help="expected SHA-256 of the archive")
    parser.add_argument(
        "--expected-counts",
        default=json.dumps(EXPECTED_COUNTS),
        help='JSON object of image counts per folder, or "{}" to skip the check',
    )
    args = parser.parse_args(argv)

    archive = Path(args.downloads) / args.file_name
    download(args.url, archive, args.size, args.sha256)
    metadata = {"source": args.url, "doi": DOI, "file": args.file_name, "archive_size": args.size}
    info = extract(archive, args.data_dir, args.sha256, json.loads(args.expected_counts), metadata)
    print(json.dumps(info, indent=1))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
