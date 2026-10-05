#!/usr/bin/env python3
"""Pin the upstream checkpoint: record the SHA-256 and byte size of ``raftstereo-middlebury.pth``.

The checkpoint is not on a model hub. Upstream publishes it inside ``models.zip``, linked from
``download_models.sh`` (a Dropbox URL, no credential). The committed manifest names that archive URL and the member
with no digest and no size (``"revision": "unpinned"``, ``"sha256": null``, ``"bytes": null``). Until this tool has
run, the package refuses to stage, verify or load the checkpoint.

What it does, in order:

1. downloads the manifest's archive URL into a staging directory (or uses ``--archive PATH``, a ``models.zip`` you
   downloaded yourself from the same link);
2. extracts the single member named ``raftstereo-middlebury.pth`` (nothing else) and computes its SHA-256 and size;
3. builds the carried upstream RAFT-Stereo architecture (``MODEL_ARGS``), loads the file with
   ``torch.load(weights_only=True)`` and ``load_state_dict(strict=True)``, and stops if any tensor is missing,
   unexpected or mis-shaped, or if the parameter count differs from ``EXPECTED_PARAMETERS``;
4. moves the checkpoint into ``weights/<key>/``, writes the manifest (revision = the checkpoint's full SHA-256, bytes,
   sha256, totalBytes, and the archive's own size and digest at pin time for the record) and replaces
   ``MODEL_REVISION = "unpinned"`` in ``src/<package>/pipeline.py`` with the digest.

A URL-hosted file has no commit, so the pinned revision is the SHA-256 of the checkpoint's bytes (not of the archive,
which the host may re-pack). The tool then prints what is left to do by hand: regenerate the notebook, revise the
prose that says the checkpoint is not yet pinned, and run the validator. It needs network access to the archive host
(or ``--archive``) and the pinned torch, numpy, scipy and opt-einsum.

Usage (from the repository root):
    python tools/pin_snapshot.py                      # download, extract, hash, load-check and pin
    python tools/pin_snapshot.py --archive models.zip # same, from a local copy of the archive
    python tools/pin_snapshot.py --dry-run            # everything except writing
"""

# ruff: noqa: E501  -- printed guidance is kept on one line per message
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _template(root: Path) -> dict:
    spec = importlib.util.spec_from_file_location("notebook_template", root / "tools" / "notebook_template.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TEMPLATE


def _package(root: Path, name: str):
    """The repository's package from ``src/`` (the architecture lives in its carried upstream source)."""
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module(f"{name}.pipeline")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_check(pipeline, path: Path) -> str:
    """Strict-load the checkpoint into the carried architecture; return a one-line summary."""
    model = pipeline.build_model()
    state = pipeline.load_checkpoint_state(path)
    model.load_state_dict(state, strict=True)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != pipeline.EXPECTED_PARAMETERS:
        raise ValueError(f"{parameters:,} parameters != EXPECTED_PARAMETERS {pipeline.EXPECTED_PARAMETERS:,}")
    if len(state) != pipeline.EXPECTED_STATE_TENSORS:
        raise ValueError(f"{len(state)} state tensors != EXPECTED_STATE_TENSORS {pipeline.EXPECTED_STATE_TENSORS}")
    return f"{len(state)} tensors, {parameters:,} parameters, strict load ok"


def pin(
    root: Path = ROOT,
    *,
    dry_run: bool = False,
    archive: Path | None = None,
    download: Callable[[str, Path], int] | None = None,
    load_check: Callable[[object, Path], str] | None = None,
) -> int:
    """Pin the checkpoint under ``root``; ``download``/``load_check`` default to the package's downloader and loader."""
    template = _template(root)
    pipeline = _package(root, template["package"])
    download = download or pipeline.download_archive
    load_check = load_check or _load_check
    key = template["weights_key"]
    weights_dir = root / "weights" / key
    manifest_path = weights_dir / "dimer-base-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest["files"]) != 1:
        print(f"expected one checkpoint file in {manifest_path}, found {len(manifest['files'])}", file=sys.stderr)
        return 1
    [entry] = manifest["files"]
    if entry["url"] != pipeline.WEIGHTS_ARCHIVE_URL or entry["path"] != pipeline.WEIGHTS_FILE:
        print(f"manifest entry {entry['path']} @ {entry['url']} is not the package's checkpoint ({pipeline.WEIGHTS_FILE} @ {pipeline.WEIGHTS_ARCHIVE_URL})", file=sys.stderr)
        return 1

    staging = root / "outputs" / "pin-staging" / key
    staging.mkdir(parents=True, exist_ok=True)
    staged_archive = staging / "models.zip"
    if archive is not None:
        shutil.copyfile(archive, staged_archive)
        print(f"using local archive {archive}")
    else:
        print(f"downloading {entry['url']}")
        download(entry["url"], staged_archive)
    archive_bytes = staged_archive.stat().st_size
    archive_digest = _sha256(staged_archive)
    staged = staging / entry["path"]
    staged.unlink(missing_ok=True)
    try:
        member = pipeline.extract_member(staged_archive, entry.get("archiveMember", entry["path"]), staged)
    except Exception as exc:  # noqa: BLE001 -- any archive defect means nothing is pinned
        print(f"models.zip: {exc}; nothing written", file=sys.stderr)
        return 1
    size = staged.stat().st_size
    digest = _sha256(staged)
    try:
        summary = load_check(pipeline, staged)
    except Exception as exc:  # noqa: BLE001 -- any load failure means the file must not be pinned
        print(f"{entry['path']}: strict load into the carried RAFT-Stereo architecture failed ({exc}); nothing written", file=sys.stderr)
        return 1
    print(f"  archive: {archive_bytes:,} bytes  sha256 {archive_digest}")
    print(f"  {member}: {size:,} bytes  sha256 {digest}  ({summary})")

    pinned = {
        **manifest,
        "revision": digest,
        "files": [{**entry, "archiveMemberPath": member, "bytes": size, "sha256": digest}],
        "totalBytes": size,
        "archiveAtPin": {"url": entry["url"], "bytes": archive_bytes, "sha256": archive_digest, "note": "informational: the host may re-pack the archive; only the member's digest is verified"},
    }
    if dry_run:
        print(json.dumps(pinned, indent=2))
        return 0

    module_path = root / "src" / template["package"] / "pipeline.py"
    text = module_path.read_text(encoding="utf-8")
    new_text, n = re.subn(r'^MODEL_REVISION = "[^"]*"$', f'MODEL_REVISION = "{digest}"', text, count=1, flags=re.M)
    if n != 1:
        print(f"{module_path}: MODEL_REVISION constant not found; nothing written", file=sys.stderr)
        return 1
    shutil.move(str(staged), str(weights_dir / entry["path"]))
    staged_archive.unlink(missing_ok=True)
    manifest_path.write_text(json.dumps(pinned, indent=2) + "\n", encoding="utf-8", newline="\n")
    module_path.write_text(new_text, encoding="utf-8", newline="\n")
    print(f"wrote {manifest_path.relative_to(root)} and MODEL_REVISION in {module_path.relative_to(root)}")

    leftovers = [
        name
        for name in ("README.md", "MODEL_CARD.md", "STATUS.md", "docs/WEIGHTS.md", "tutorials/README.md", "docs/release-verification.md")
        if (root / name).exists() and "not yet pinned" in (root / name).read_text(encoding="utf-8")
    ]
    print("next:")
    print("  1. commit, then run `python tools/build_notebook.py` and commit the regenerated notebook")
    if leftovers:
        print(f"  2. replace the 'not yet pinned' statements in: {', '.join(leftovers)} (cite the digest and byte size)")
    print("  3. run `python tools/validate_release_assets.py` and `pytest`")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="download, extract, hash and load-check only; write nothing")
    parser.add_argument("--archive", type=Path, default=None, help="a local models.zip downloaded from the manifest URL")
    args = parser.parse_args(argv)
    return pin(ROOT, dry_run=args.dry_run, archive=args.archive)


if __name__ == "__main__":
    raise SystemExit(main())
