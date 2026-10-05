"""Static release-asset validation for the RAFT-Stereo dense stereo disparity DIMER pipeline.

Checks the STANDALONE tutorial notebook (DIMER Notebook Specification 2.2 §4, isolated hash-locked environment of
§25.13), the carried stage runner and upstream source, the tutorial registry, model card (DIMER Model Card
Specification 1.2), README, STATUS.md and weight documentation for source conformance and cross-document identity
consistency, the checkpoint pin state, and runs the generator parity checks (PAR1-PAR3).

This is source validation only. A PASS here is NOT clean-runtime execution evidence;
the release gate is defined in docs/release-verification.md.
"""
# ruff: noqa: E501  -- rule messages name the file and requirement in full; they are kept on one line
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "raft_stereo_pipeline"
REPO_NAME = "raft-stereo-pipeline"
NOTEBOOK_NAME = "raft_stereo_colab.ipynb"
EXPECTED_PROFILE = "E2E"
EXPECTED_MODEL_ID = "princeton-vl/RAFT-Stereo/raftstereo-middlebury"
WEIGHTS_KEY = "raftstereo-middlebury"
STAGE_RUNNER = ROOT / "tools" / "tutorial_stages.py"
UNPINNED = "unpinned"
UNPINNED_PHRASE = "not yet pinned"
# The upstream source commit the package carries; documents may cite it beside the checkpoint revision.
UPSTREAM_COMMIT = "6e93ed2169bd858dbb43033988563f3b0bb49506"
KNOWN_SHAS: frozenset[str] = frozenset({UPSTREAM_COMMIT})
BYOD_GATES = ("USE_BYOD_PAIR", "USE_BYOD_DATASET")
# EXE1/EXE2 form fields, exactly as an executor edits them in a run copy.
BYOD_FIELD_LINES = (
    'USE_BYOD_PAIR = False  # @param {type:"boolean"}',
    "BYOD_LEFT_PATH = ''  # @param {type:\"string\"}",
    "BYOD_RIGHT_PATH = ''  # @param {type:\"string\"}",
    "BYOD_DISPARITY_PATH = ''  # @param {type:\"string\"}",
    'USE_BYOD_DATASET = False  # @param {type:"boolean"}',
    "BYOD_DATASET_DIR = ''  # @param {type:\"string\"}",
)
# NOTEBOOK_SPEC 2.2 §3.5 guided layer (GDL1–GDL15): learner-facing elements that must survive regeneration.
GUIDED_MARKDOWN_MARKERS = (
    "## How to use this notebook",
    "**Who this notebook is for.**",
    "**Where the code runs.**",
    "**Form controls.**",
    "**Two kinds of cell.**",
    "**Section tags.**",
    "## The task: Input → Model/System → Output",
    "## Roadmap",
    "**Fast path.**",
    "<summary><strong>Glossary</strong>",
    "> **Infrastructure.**",
    "## 14. Your turn — change one thing: unfreeze the encoders",
    "**Predict → Change one thing → Run → Observe → Explain**",
    "Runtime → Run after",
    "## Troubleshooting",
    "## Conclusion (your notes)",
)
GUIDED_MIN_COUNTS = {
    "**Predict before running:**": 7,
    "**What to notice": 8,
    "**Expected result:**": 4,
    "<summary>Check your reasoning": 7,
}
GUIDED_PREDICT_SECTIONS = (4, 5, 7, 8, 9, 10)
GLOSSARY_TERMS = ("Rectified stereo pair", "Disparity", "Occlusion and the valid mask", "Correlation volume", "Refinement iterations", "EPE (end-point error)", "Bad-pixel rate", "SGBM", "Held-out split", "Adapter / reload equivalence")
SECTION_TAGS = ("[Concept]", "[Evaluation practice]", "[Engineering]")
INFRASTRUCTURE_TITLES = {
    "check": "# @title Infrastructure: check the runtime, GPU and disk; create a fresh run directory",
    "carrier": "# @title Infrastructure: write and verify the carried package, stage runner, lock and manifests",
    "install": "# @title Infrastructure: install the locked runtime into an isolated environment and define the stage runner",
    "weights": "# @title Infrastructure: verify the carried upstream source; stage and digest-verify the pinned checkpoint",
}
# The learner cells run the stages in this order (RUN1: the default path is one pass from the top).
STAGE_CALLS = (
    "run_stage('weights')",
    "run_stage('demo', '--iters', VALID_ITERS)",
    "run_stage('probes', '--iters', VALID_ITERS)",
    "run_stage('prepare', '--n-pairs', N_PAIRS, '--dataset-seed', DATASET_SEED, '--holdout', HOLDOUT, '--seed', SEED, '--epochs', EPOCHS)",
    "run_stage('baselines', '--iters', VALID_ITERS)",
    "run_stage('adapt', '--epochs', EPOCHS, '--batch-size', BATCH_SIZE, '--lr', LEARNING_RATE, '--train-iters', TRAIN_ITERS,",
    "run_stage('evaluate', '--iters', VALID_ITERS)",
    "run_stage('infer', '--new-seed', NEW_DATA_SEED, '--iters', VALID_ITERS)",
    "run_stage('reload')",
    "run_stage('byod_pair', *pair_options)",
)
# Kernel cells may import only the standard library, IPython's display helpers and (for the BYOD upload) google.colab.
ALLOWED_KERNEL_IMPORTS = frozenset(
    {"hashlib", "io", "json", "os", "pathlib", "platform", "shutil", "subprocess", "tempfile", "time", "urllib.error", "urllib.request", "uuid", "zipfile", "IPython.display", "google.colab"}
)
KERNEL_CODE_MARKERS = (
    "CARRIED_FILES = {",
    "CARRIED_HASHES = {",
    "for name, text in CARRIED_FILES.items():",
    "if hashlib.sha256(path.read_bytes()).hexdigest() != CARRIED_HASHES[name]:",
    "NOTEBOOK_SOURCE = json.loads(",
    "if len(wheel) != UV_BYTES or hashlib.sha256(wheel).hexdigest() != UV_SHA256:",
    "'venv', '--managed-python', '--python', '3.12.12'",
    "'pip', 'install', '--python', str(PYTHON), '--require-hashes', '--only-binary', ':all:'",
    "MPLBACKEND='Agg'",
    "for name in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'PYTHONPATH', 'PYTHONHOME'):",
    "def run_stage(stage, *options):",
    "detail = error['type'] + ': ' + error['message']",
    "raise RuntimeError(f'Stage {stage!r} failed (exit {process.returncode}): {detail}')",
    "if not RUNTIME['cuda']:",
    "from google.colab import files",
    "files.upload()",
    "if name.startswith('/') or ':' in name or '..' in parts:",
    "shutil.rmtree(target, ignore_errors=True)",
    "run_stage(stage, '--namespace', 'byod', *options)",
    "('prepare', ['--byod', dataset_dir,",
    "load_record('{stem}_result.json')".replace("{stem}", "raft_stereo"),
)
# What the carried stage runner must do (checked on tools/tutorial_stages.py, which the carrier holds byte for byte).
RUNNER_MARKERS = (
    "RaftStereoPipeline.from_pretrained(device=device(), weights_dir=run.snapshot())",
    "RaftStereoPipeline.load_artifact(artifact, weights_dir=run.snapshot(), device=device())",
    "upstream = verify_upstream()",
    "fetched = stage_missing_files(target, allow_download=True)",
    "verified = verify_snapshot(target)",
    "manifest = validate_inputs(pair[\"left\"], pair[\"right\"], iters=iters, name=pair[\"id\"])",
    "sgbm = sgbm_disparity(",
    "evaluation_report(result, pair[\"disparity\"], pair[\"valid\"], baselines={\"sgbm\": sgbm[\"disparity\"]}",
    "random_dot_pair(shift)",
    "records = read_stereo_records(byod_path)",
    "manifest = validate_dataset(records, epochs=opts.epochs)",
    "train, held = split_dataset(records, holdout=opts.holdout, seed=opts.seed)",
    "raise RuntimeError(f\"the split leaked records into both sides",
    "if records_digest(records) != data[\"dataset_digest\"]:",
    "median = median_disparity(train)",
    "search = sgbm_num_disparities(train)",
    "pretrained = pipe.evaluate(held, iters=iters)",
    "run_record = pipe.finetune(",
    "freeze_encoders=bool(opts.freeze_encoders)",
    "descriptor = pipe.save_artifact(artifact,",
    "pipe = load_adapted(run, Path(adapted_state[\"artifact\"][\"path\"]))",
    "adapted = pipe.evaluate(held, iters=iters)",
    "if baselines[\"iters\"] != iters:",
    "history.append(",
    "write_disparity_png16(",
    "reloaded = load_adapted(run, artifact)",
    "raise AssertionError(f\"the reloaded adapter does not reproduce the in-memory model within tolerance",
    "\"remote_code_fetched\": False",
    "\"evidence_class\": \"tutorial/sanity evidence on the stated sample; not a benchmark\"",
    "error_file.write_text(json.dumps(",
    'print(f"STAGE FAILED ({options.stage}): {type(exc).__name__}: {message}", flush=True)',
)
EXPECTED_OUTPUTS = (
    "_input_manifest.json",
    "_evaluation_report.json",
    "probes.json",
    "_dataset.json",
    "_baselines.json",
    "_adaptation.json",
    "_held_out_evaluation.json",
    "_new_pairs.csv",
    "_new_pairs.json",
    "_adapter.safetensors",
    "_result.json",
    "byod_pair_report.json",
)
MARKDOWN_MARKERS = (
    "**Capability:** dense disparity estimation for rectified stereo pairs with RAFT-Stereo",
    "**The default path really adapts the model.**",
    "corr_implementation = 'reg'",
    "**Read the loss as optimisation evidence only:**",
    "**Every run starts from the checkpoint.**",
    "**Sample provenance and licence:**",
    "**Pretraining overlap:**",
    "sample-sanity",
    "weights_only=True",
    "Nothing is installed into the notebook kernel",
    "--require-hashes",
    "MIT",
    "non-occluded",
    "**Pin state of this revision:",
)
# Direct model-library use that must stay inside the carried files (G2): the kernel imports no model library at all.
FORBIDDEN_IN_KERNEL = (
    "huggingface_hub",
    "hf_hub_download(",
    "safetensors",
    "pickle",
    "sys.executable",
    "importlib",
    "pip install",
    "'-m', 'pip'",
    "import torch",
    "import cv2",
    "import numpy",
)

NOTEBOOK_SPEC = "2.2"
ALLOWED_PROFILES = {"E2E", "ARTIFACT-INFERENCE", "TASK-INFERENCE", "MULTI-CAPABILITY", "SMOKE"}
STATUS_TOKENS = ("Candidate", "Release-grade")
PLACEHOLDER = re.compile(r"\b(TODO|TBD|FIXME)\b|Insert text here|Tooltip:", re.I)
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")
IDENTITY_NAMES = ("MODEL_ID", "MODEL_REVISION", "MODEL_LICENSE", "MODEL_KEY")
MODEL_CARD_SPEC = "1.2"
UNSUPPORTED_CLAIMS = re.compile(
    r"\b(production[- ]ready|battle[- ]tested|state[- ]of[- ]the[- ]art results (were|are) reproduced"
    r"|benchmark superiority (is|was) (shown|established)|is release-grade|now release-grade)\b",
    re.I,
)
# MODEL_CARD_SPEC 1.2 G14/G15/G17: a public card names no non-public capability, private source or internal build
# environment, and avoids fleet-maintenance vocabulary. These patterns catch the known forms; they do not replace §6.
CARD_NON_PUBLIC = re.compile(r"ml-worker|workbench|internal (?:enterprise|tooling|platform)|\.venv\b|Windows venv|holdout surface", re.I)
CARD_FLEET_VOCABULARY = re.compile(r"\bfleet\b|\bwave\b|build queue|matrix row|inventory row", re.I)
REQUIRED_CARD_HEADINGS = [
    (4, "Description"),
    (4, "Intended Use and Limitations"),
    (6, "Primary Intended Uses"),
    (6, "Primary Intended Users"),
    (6, "Out-of-scope use cases"),
    (4, "Factors"),
    (6, "Groups"),
    (6, "Instrumentation"),
    (6, "Environment"),
    (4, "Metrics"),
    (6, "Performance Measures"),
    (6, "Decision thresholds"),
    (6, "Approaches to uncertainty and variability"),
    (4, "Ethical considerations and biases"),
    (6, "Data"),
    (6, "Human Life"),
    (6, "Mitigations"),
    (6, "Risks and harms"),
    (6, "Use cases"),
]
COMMON_MARKDOWN_MARKERS = (
    f"**Notebook specification:** DIMER Notebook Specification {NOTEBOOK_SPEC} — **standalone** (§4)",
    "**Mode:** `",
    "**Run all:**",
    "**Bring Your Own Data:**",
    "**This notebook is standalone.**",
    "**Learning objectives:**",
    "## Prerequisites",
    "Do not upload confidential or restricted",
    "- **External access:** Dropbox, to fetch upstream's `models.zip`",
    "## 1. Check the runtime",
    "## 2. Carry the code and install the locked runtime",
    "## 3. Pin, stage and verify the model",
    "## Interpretation and limits",
    "Successful execution proves that the recorded repository revision",
    "without the repository being",
    "It does **not** establish benchmark superiority",
    "## References",
    f"- Repository model card: https://github.com/kurtvalcorza/{REPO_NAME}/blob/main/MODEL_CARD.md",
)
FORBIDDEN_PATTERNS = (
    ("credential in clone URL", re.compile(r"https://[^/'\"\s]*@github\.com/|x-access-token:")),
    ("repository clone (ST1)", re.compile(r"\bgit\b[^\n]*\bclone\b|github\.com")),
    ("editable self-install", re.compile(r"""['"](?:-e|--editable)['"]|pip install (?:-e|--editable)\b""")),
    ("repository package import in the kernel (ST1)", re.compile(rf"^\s*(?:from|import)\s+{PACKAGE}\b", re.M)),
    ("mutable model reference (MOD14)", re.compile(r"revision\s*=\s*['\"](?:main|latest)['\"]")),
    ("trust_remote_code enabled", re.compile(r"trust_remote_code\s*[=:]\s*True")),
    (
        "unsafe deserialization",
        re.compile(
            r"\bpickle\.load"
            r"|\btorch\.load\s*\((?![^)]*weights_only\s*=\s*True)"
            r"|weights_only\s*=\s*False"
            r"|getattr\(\s*torch\s*,\s*['\"]load['\"]"
        ),
    ),
    ("archive extractall", re.compile(r"\.extractall\s*\(")),
    ("notebook magic or shell escape", re.compile(r"(?m)^\s*[%!]|get_ipython\(\)")),
    ("unhashed or source install", re.compile(r"--no-binary|--no-build-isolation|--trusted-host|--extra-index-url")),
)
# Checked on the carried files (package, runner, lock): everything above except the kernel-only package-import rule.
CARRIED_FORBIDDEN = tuple(item for item in FORBIDDEN_PATTERNS if not item[0].startswith("repository package import"))


class ValidationError(AssertionError):
    """Raised for any release-asset defect; the message names the file and rule."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _cell_source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def _strip_comments(source: str) -> str:
    out: list[str] = []
    last_row, last_col = 1, 0
    lines = source.splitlines(keepends=True)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return source
    for token in tokens:
        (srow, scol), (erow, ecol) = token.start, token.end
        if srow > last_row:
            out.append(lines[last_row - 1][last_col:] if last_row - 1 < len(lines) else "")
            for row in range(last_row, srow - 1):
                out.append(lines[row])
            last_row, last_col = srow, 0
        if srow - 1 < len(lines):
            out.append(lines[srow - 1][last_col:scol])
        if token.type != tokenize.COMMENT:
            out.append(token.string)
        last_row, last_col = erow, ecol
    return "".join(out)


def _assignment_targets(node: ast.AST):
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign | ast.AugAssign | ast.NamedExpr | ast.For | ast.comprehension):
        targets = [node.target]
    elif isinstance(node, ast.withitem) and node.optional_vars is not None:
        targets = [node.optional_vars]
    else:
        return []
    names = []
    for target in targets:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name):
                names.append(sub.id)
    return names


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    _check(spec is not None and spec.loader is not None, f"tools/{name}.py is required")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _package_identity() -> tuple[str, str]:
    """MODEL_ID / MODEL_REVISION from the package source (without importing torch)."""
    text = _read(ROOT / "src" / PACKAGE / "pipeline.py")
    model_id = re.search(r'^MODEL_ID = "([^"]+)"$', text, re.M)
    revision = re.search(r'^MODEL_REVISION = "([^"]+)"$', text, re.M)
    _check(model_id is not None and revision is not None, "pipeline.py must define MODEL_ID and MODEL_REVISION")
    _check(
        SHA64.match(revision.group(1)) is not None or revision.group(1) == UNPINNED,
        f"MODEL_REVISION must be the checkpoint's 64-hex SHA-256 or the {UNPINNED!r} sentinel",
    )
    _check(model_id.group(1) == EXPECTED_MODEL_ID, f"MODEL_ID drifted from {EXPECTED_MODEL_ID}")
    return model_id.group(1), revision.group(1)


def _front_matter(text: str) -> dict[str, str]:
    front = text.split("---", 2)[1]
    fields: dict[str, str] = {}
    for line in front.splitlines():
        match = re.match(r"^([A-Za-z_]+):\s*(.*)$", line)
        if match:
            fields[match.group(1)] = match.group(2).strip().strip('"')
    return fields


def validate_model_card() -> None:
    path = ROOT / "MODEL_CARD.md"
    text = _read(path)
    _check(text.startswith("---\n"), "MODEL_CARD.md must start with YAML front matter (G1)")
    fields = _front_matter(text)
    for key in ("license", "model_card_spec", "pipeline_tag", "base_model", "date_published"):
        _check(key in fields, f"MODEL_CARD.md missing front-matter field: {key} (G1)")
    _check(fields["model_card_spec"] == MODEL_CARD_SPEC, f"MODEL_CARD.md model_card_spec must be {MODEL_CARD_SPEC!r}")
    _check(fields["base_model"] == EXPECTED_MODEL_ID, "MODEL_CARD.md base_model must equal MODEL_ID")
    _check(re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?|null", fields["date_published"]) is not None, "MODEL_CARD.md date_published must be YYYY, YYYY-MM, YYYY-MM-DD or null (G2)")
    _check(not PLACEHOLDER.search(text), "MODEL_CARD.md contains placeholder/scaffolding text (G9, G11)")
    _check(not UNSUPPORTED_CLAIMS.search(text), "MODEL_CARD.md makes an unsupported release/benchmark claim (G12)")
    _check(not CARD_NON_PUBLIC.search(text), f"MODEL_CARD.md names a non-public capability or source (G14, G15): {CARD_NON_PUBLIC.search(text)}")
    _check(not CARD_FLEET_VOCABULARY.search(text), f"MODEL_CARD.md uses fleet-maintenance vocabulary (G17): {CARD_FLEET_VOCABULARY.search(text)}")
    preamble = text.split("#### Description", 1)[0]
    badge_targets = re.findall(r"\[!\[[^\]]*\]\([^)]*\)\]\(([^)]+)\)", preamble)
    _check(not [t for t in badge_targets if t.rstrip("/") == f"https://github.com/kurtvalcorza/{REPO_NAME}"], "MODEL_CARD.md badge row must not link back to this repository (G8)")
    h1 = re.findall(r"(?m)^# (?!#)(.+)$", text)
    _check(len(h1) == 1, f"MODEL_CARD.md must contain exactly one H1, got {len(h1)} (G3)")
    found = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            found.append((len(match.group(1)), match.group(2).strip()))
    positions = []
    for heading in REQUIRED_CARD_HEADINGS:
        matches = [index for index, item in enumerate(found) if item[0] == heading[0] and item[1].casefold() == heading[1].casefold()]
        _check(len(matches) == 1, f"required model-card heading missing/duplicated: {heading} (G4, G5)")
        positions.append(matches[0])
    _check(positions == sorted(positions), "required model-card headings are out of order (G6)")
    _check("## Immutable provenance" in text, "MODEL_CARD.md must carry an '## Immutable provenance' section")


def _manifest() -> dict:
    return json.loads(_read(ROOT / "weights" / WEIGHTS_KEY / "dimer-base-manifest.json"))


def validate_pin_state() -> None:
    """The package, the manifest and the documents agree on whether the checkpoint is pinned."""
    model_id, revision = _package_identity()
    manifest = _manifest()
    _check(manifest.get("modelId") == model_id, "manifest modelId != MODEL_ID")
    _check(manifest.get("revision") == revision, f"manifest revision {manifest.get('revision')!r} != MODEL_REVISION {revision!r}")
    digests = [entry.get("sha256") for entry in manifest["files"]]
    sizes = [entry.get("bytes") for entry in manifest["files"]]
    docs = ("README.md", "MODEL_CARD.md", "docs/WEIGHTS.md", "STATUS.md")
    if revision == UNPINNED:
        _check(all(d is None for d in digests) and all(b is None for b in sizes) and manifest.get("totalBytes") is None, "an unpinned manifest must not carry digests or sizes; run tools/pin_snapshot.py to pin the checkpoint")
        for name in docs + ("docs/release-verification.md",):
            _check(UNPINNED_PHRASE in _read(ROOT / name), f"{name} must state that the checkpoint is {UNPINNED_PHRASE} while MODEL_REVISION is {UNPINNED!r}")
        _check("Current status: **Candidate" in _read(ROOT / "STATUS.md"), "an unpinned checkpoint can only be Candidate")
    else:
        _check(digests == [revision], "a pinned manifest records the checkpoint's SHA-256, which is MODEL_REVISION")
        _check(all(isinstance(b, int) and b > 0 for b in sizes) and manifest.get("totalBytes") == sum(sizes), "a pinned manifest records the byte size and totalBytes")
        for name in docs + ("tutorials/README.md", "docs/release-verification.md"):
            _check(UNPINNED_PHRASE not in _read(ROOT / name), f"{name} still says the checkpoint is {UNPINNED_PHRASE}; revise it for the pinned digest")


def validate_upstream_source() -> None:
    """The carried upstream files match UPSTREAM_SHA256 and the commit the documents cite (MOD6/RC3)."""
    text = _read(ROOT / "src" / PACKAGE / "pipeline.py")
    block = re.search(r"^UPSTREAM_SHA256 = (\{.*?^\})$", text, re.M | re.S)
    _check(block is not None, "pipeline.py must define UPSTREAM_SHA256 as a literal")
    expected = ast.literal_eval(block.group(1))
    commit = re.search(r'^UPSTREAM_COMMIT = "([0-9a-f]{40})"$', text, re.M)
    _check(commit is not None and commit.group(1) == UPSTREAM_COMMIT, "pipeline.py UPSTREAM_COMMIT drifted from the validator's")
    root = ROOT / "src" / PACKAGE / "third_party" / "raft_stereo"
    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    _check(on_disk == set(expected), f"third_party/raft_stereo holds {sorted(on_disk ^ set(expected))} beyond/short of UPSTREAM_SHA256")
    for name, digest in expected.items():
        _check(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, f"carried upstream file {name} does not match UPSTREAM_SHA256")
    template = _load_tool("notebook_template").TEMPLATE
    carried = [dest for dest in template["carried"] if "/third_party/raft_stereo/" in dest]
    _check(sorted(d.split("/third_party/raft_stereo/", 1)[1] for d in carried) == sorted(expected), "the notebook template must carry every upstream file")
    for name in ("README.md", "MODEL_CARD.md", "docs/WEIGHTS.md"):
        _check(UPSTREAM_COMMIT in _read(ROOT / name), f"{name} must cite the carried upstream commit {UPSTREAM_COMMIT}")


def validate_identity_consistency() -> None:
    model_id, revision = _package_identity()
    for name in ("README.md", "MODEL_CARD.md", "docs/WEIGHTS.md"):
        text = _read(ROOT / name)
        _check(model_id in text, f"{name} must name the upstream model `{model_id}`")
        _check(revision in text, f"{name} must cite the checkpoint revision {revision}")
        other = re.findall(r"\b[0-9a-f]{40}\b", text)
        stray = sorted({sha for sha in other if sha not in KNOWN_SHAS})
        _check(not stray, f"{name} cites an unexpected 40-hex revision: {stray}")


# --- weight-facts check (fleet rollout 2026-09-24) ---
# Every SHA-256 digest and byte count quoted in the weight prose must come from a committed
# weights/*/dimer-base-manifest.json, or be declared below with a label saying what it describes
# (dataset files, upstream files that are not staged, origin checkpoints, totals). Declared entries
# that no document cites any more are rejected, so the allowlist cannot go stale.
WEIGHT_DOCS = ("README.md", "MODEL_CARD.md", "docs/WEIGHTS.md")
EXTERNAL_WEIGHT_BYTES: dict[int, str] = {
    44605701: "expected size of raftstereo-middlebury.pth computed from the carried architecture (torch.save of the DataParallel state_dict); an estimate, not a measurement of the upstream file",
}
EXTERNAL_WEIGHT_DIGESTS: dict[str, str] = {}
_DIGEST = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{64}(?![0-9a-fA-F])")
_GROUPED = r"(\d{1,3}(?:[,\u202f\u00a0 ]\d{3})+|\d+)"
_BYTE_COUNT = re.compile(r"(?<![\d,\-])" + _GROUPED + r"\s*bytes\b|totalBytes`?\s*" + _GROUPED)


def _manifest_facts(root: Path = ROOT) -> tuple[set[str], set[int]]:
    digests: set[str] = set()
    sizes: set[int] = set()
    for path in sorted(root.glob("weights/*/dimer-base-manifest.json")):
        manifest = json.loads(_read(path))
        if isinstance(manifest.get("totalBytes"), int):
            sizes.add(manifest["totalBytes"])
        for entry in manifest["files"]:
            if entry.get("sha256"):
                digests.add(entry["sha256"])
            if isinstance(entry.get("bytes"), int):
                sizes.add(entry["bytes"])
    return digests, sizes


def validate_weight_facts(root: Path = ROOT) -> None:
    """Every SHA-256 and byte count quoted in the weight prose must come from a manifest or a labelled allowlist entry."""
    digests, sizes = _manifest_facts(root)
    _check(any(root.glob("weights/*/dimer-base-manifest.json")), "no weights/*/dimer-base-manifest.json found to check weight facts against")
    cited_digests: set[str] = set()
    cited_sizes: set[int] = set()
    for name in WEIGHT_DOCS:
        path = root / name
        if not path.exists():
            continue
        text = _read(path)
        found_digests = set(_DIGEST.findall(text))
        found_sizes = {int(re.sub(r"[,\u202f\u00a0 ]", "", m.group(1) or m.group(2))) for m in _BYTE_COUNT.finditer(text)}
        cited_digests |= found_digests
        cited_sizes |= found_sizes
        bad_digests = sorted(found_digests - digests - set(EXTERNAL_WEIGHT_DIGESTS))
        _check(not bad_digests, f"{name} cites SHA-256 digests absent from every manifest and from EXTERNAL_WEIGHT_DIGESTS: {bad_digests}")
        bad_sizes = sorted(found_sizes - sizes - set(EXTERNAL_WEIGHT_BYTES))
        _check(not bad_sizes, f"{name} cites byte counts absent from every manifest and from EXTERNAL_WEIGHT_BYTES: {bad_sizes}")
    stale = sorted(set(EXTERNAL_WEIGHT_BYTES) - cited_sizes) + sorted(set(EXTERNAL_WEIGHT_DIGESTS) - cited_digests)
    _check(not stale, f"EXTERNAL_WEIGHT_* entries no weight document cites any more: {stale}")


# --- end weight-facts check ---

def validate_release_status() -> None:
    status = _read(ROOT / "STATUS.md")
    match = re.search(r"Current status: \*\*(Candidate|Release-grade)\b", status)
    _check(match is not None, "STATUS.md must declare 'Current status: **Candidate**' or '**Release-grade**'")
    token = match.group(1)
    readme = _read(ROOT / "README.md")
    _check("## Release status" in readme, "README.md must have a '## Release status' section")
    section = readme.split("## Release status", 1)[1]
    _check(section.lstrip().startswith(f"**{token}"), f"README.md release status must open with **{token}**")
    registry = _read(ROOT / "tutorials" / "README.md").replace("**", "")
    _check(f"| {token}" in registry, f"tutorials/README.md must record the {token} status")
    other = [t for t in STATUS_TOKENS if t != token]
    for name, text in (("README.md", section.replace("**", "")), ("tutorials/README.md", registry)):
        for stale in other:
            _check(f"| {stale}" not in text, f"{name} carries a conflicting status token")
    if token == "Candidate":
        _check(
            "docs/release-verification.md" in registry or "release-verification" in registry,
            "tutorials/README.md must point Candidate notebooks at docs/release-verification.md",
        )
    for name in ("README.md", "STATUS.md", "tutorials/README.md", "docs/release-verification.md"):
        text = _read(ROOT / name)
        _check(not PLACEHOLDER.search(text), f"{name} contains placeholder text")
        _check(not UNSUPPORTED_CLAIMS.search(text), f"{name} makes an unsupported release/benchmark claim")
    verification = _read(ROOT / "docs" / "release-verification.md")
    _check(
        "## Recorded executions" in verification,
        "docs/release-verification.md must have '## Recorded executions'",
    )




def _validate_notebook_structure(path: Path, notebook: dict) -> tuple[list[tuple[int, str, ast.Module]], str]:
    _check(notebook.get("nbformat") == 4, f"{path.name}: nbformat must be 4")
    dimer = notebook.get("metadata", {}).get("dimer")
    _check(isinstance(dimer, dict), f"{path.name}: metadata.dimer block is required")
    profile = dimer.get("notebook_profile")
    _check(profile in ALLOWED_PROFILES, f"{path.name}: invalid metadata.dimer.notebook_profile {profile!r}")
    _check(profile == EXPECTED_PROFILE, f"{path.name}: profile {profile!r} != declared {EXPECTED_PROFILE!r}")
    spec = dimer.get("notebook_spec", dimer.get("notebook_spec_version"))
    _check(spec == NOTEBOOK_SPEC, f"{path.name}: metadata.dimer must declare notebook spec version '{NOTEBOOK_SPEC}'")
    _check(dimer.get("notebook_mode") in ("REFERENCE", "GUIDED", "WORKSHOP"), f"{path.name}: metadata.dimer.notebook_mode must declare a §3.3 pedagogical mode")
    _check(dimer.get("standalone") is True, f"{path.name}: metadata.dimer.standalone must be true (ST6)")
    _check(dimer.get("requires_dimer_worker") is False, f"{path.name}: metadata.dimer.requires_dimer_worker must be false")
    generated = dimer.get("generated_from")
    _check(isinstance(generated, dict), f"{path.name}: metadata.dimer.generated_from is required (ST5)")
    _check(generated.get("repository") == REPO_NAME, f"{path.name}: generated_from.repository must be {REPO_NAME}")
    _check(generated.get("module") == f"src/{PACKAGE}/pipeline.py", f"{path.name}: generated_from.module must name the package entry module")
    _check(bool(generated.get("generator")), f"{path.name}: generated_from.generator is required")
    cells = notebook.get("cells", [])
    _check(bool(cells) and cells[0].get("cell_type") == "markdown", f"{path.name}: first cell must be markdown")
    code_cells: list[tuple[int, str, ast.Module]] = []
    markdown_parts: list[str] = []
    for index, cell in enumerate(cells):
        source = _cell_source(cell)
        if cell.get("cell_type") == "markdown":
            markdown_parts.append(source)
            continue
        _check(cell.get("cell_type") == "code", f"{path.name}: unexpected cell type at {index}")
        _check(cell.get("execution_count") is None, f"{path.name}: code cell {index} has execution_count")
        _check(not cell.get("outputs"), f"{path.name}: code cell {index} persists outputs")
        _check(index > 0 and cells[index - 1].get("cell_type") == "markdown", f"{path.name}: code cell {index} lacks a preceding explanatory markdown cell")
        for line in source.splitlines():
            _check(not line.lstrip().startswith(("%", "!")), f"{path.name}: cell {index} uses a magic")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ValidationError(f"{path.name}: code cell {index} does not compile: {exc}") from exc
        code_cells.append((index, source, tree))
    markdown = "\n".join(markdown_parts)
    kernel = "\n".join(source for index, source, _ in code_cells if not _is_carrier(cells[index]))
    _check(not PLACEHOLDER.search(kernel + markdown), f"{path.name}: placeholder text found")
    _check(not UNSUPPORTED_CLAIMS.search(markdown), f"{path.name}: unsupported release/benchmark claim")
    return code_cells, markdown


def _is_carrier(cell: dict) -> bool:
    return bool(cell.get("metadata", {}).get("dimer", {}).get("embedded_sources"))


def _literal(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValidationError(f"{name} is not assigned as a literal at the top level of its cell")


def _validate_carrier(path: Path, notebook: dict, code_cells: list[tuple[int, str, ast.Module]], build) -> tuple[int, dict[str, str]]:
    """PAR1/SRC4: one carrier cell; every carried file is the repository file and its CARRIED_HASHES entry is correct;
    the cell verifies each written file against CARRIED_HASHES and stops on a mismatch."""
    carriers = [(index, tree) for index, _source, tree in code_cells if _is_carrier(notebook["cells"][index])]
    _check(len(carriers) == 1, f"{path.name}: exactly one carrier cell (metadata.dimer.embedded_sources) is expected, found {len(carriers)}")
    index, tree = carriers[0]
    files, hashes = _literal(tree, "CARRIED_FILES"), _literal(tree, "CARRIED_HASHES")
    template = _load_tool("notebook_template").TEMPLATE
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    ctx = build.load_context(ROOT, template, recorded)
    _check(files == ctx["files"], f"{path.name}: carried files differ from the repository sources (PAR1); regenerate the notebook")
    for name, text in files.items():
        _check(hashes.get(name) == hashlib.sha256(text.encode("utf-8")).hexdigest(), f"{path.name}: CARRIED_HASHES[{name!r}] is wrong (SRC4)")
    _check(set(hashes) == set(files), f"{path.name}: CARRIED_HASHES and CARRIED_FILES name different files")
    _check(notebook["cells"][index]["metadata"]["dimer"].get("files") == hashes, f"{path.name}: carrier metadata must record the carried hashes")
    _check(notebook["metadata"]["dimer"]["generated_from"].get("files") == hashes, f"{path.name}: generated_from.files must record the carried hashes")
    _check(notebook["metadata"]["dimer"]["generated_from"].get("module_sha256") == ctx["module_sha256"], f"{path.name}: generated_from.module_sha256 does not match src/ (PAR4)")
    verifies = False
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and "CARRIED_FILES.items()" in ast.unparse(node.iter):
            body = ast.unparse(node)
            verifies = "CARRIED_HASHES[name]" in body and "raise RuntimeError(" in body and "hashlib.sha256(path.read_bytes())" in body
    _check(verifies, f"{path.name}: the carrier must verify every written file against CARRIED_HASHES and raise on a mismatch (SRC4)")
    runner = template["stage_runner"]
    _check(files.get(runner) == _read(STAGE_RUNNER), f"{path.name}: the carried stage runner is not tools/tutorial_stages.py (PAR1)")
    lock_source = ROOT / template["carried"][template["lock"]]
    _check(files.get(template["lock"]) == _read(lock_source), f"{path.name}: the carried lock is not {lock_source.relative_to(ROOT).as_posix()} (PAR2)")
    return index, files


def _validate_lock(path: Path, files: dict[str, str], build) -> None:
    """ENV1/ENV2: the carried lock pins every pyproject pin, hashes every entry, and was compiled wheel-only."""
    template = _load_tool("notebook_template").TEMPLATE
    lock = files[template["lock"]]
    try:
        build.check_lock(build._pins(ROOT), lock)
    except SystemExit as exc:
        raise ValidationError(f"{path.name}: {exc}") from exc
    header = "\n".join(lock.splitlines()[:3])
    _check("--generate-hashes" in header and "--only-binary :all:" in header, f"{path.name}: the lock must be compiled with --generate-hashes --only-binary :all:")
    _check("--python-platform x86_64-manylinux" in header, f"{path.name}: the lock must target manylinux x86_64 (Colab/Kaggle)")


def _validate_isolated_install(path: Path, code_cells: list[tuple[int, str, ast.Module]], carrier: int) -> None:
    """RUN10/ENV6 (§25.13): nothing is pip-installed into the kernel; the only installer is the pinned uv, which installs
    the lock with --require-hashes into a separate environment; run_stage re-raises a stage's own error message."""
    template = _load_tool("notebook_template").TEMPLATE
    uv = template["uv"]
    install = [(index, source, tree) for index, source, tree in code_cells if source.startswith(INFRASTRUCTURE_TITLES["install"])]
    _check(len(install) == 1, f"{path.name}: exactly one isolated-install cell is expected")
    install_index, source, tree = install[0]
    _check(_literal(tree, "UV_URL") == uv["url"] and uv["url"].startswith("https://files.pythonhosted.org/"), f"{path.name}: UV_URL must be the pinned PyPI wheel")
    _check(_literal(tree, "UV_BYTES") == uv["bytes"], f"{path.name}: UV_BYTES must be the pinned wheel size")
    _check(SHA64.match(str(_literal(tree, "UV_SHA256"))) is not None and _literal(tree, "UV_SHA256") == uv["sha256"], f"{path.name}: UV_SHA256 must pin the uv wheel (64-hex)")
    run_stage = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_stage"]
    _check(len(run_stage) == 1, f"{path.name}: the install cell must define run_stage")
    _check("raise RuntimeError(" in ast.unparse(run_stage[0]) and "error_file" in ast.unparse(run_stage[0]), f"{path.name}: run_stage must re-raise the stage's error message")
    for index, cell_source, cell_tree in code_cells:
        if index == carrier:
            continue
        stripped = _strip_comments(cell_source)
        leaked = [marker for marker in FORBIDDEN_IN_KERNEL if marker in stripped]
        _check(not leaked, f"{path.name}: cell {index} does kernel-side work that belongs in the isolated environment: {leaked}")
        for node in ast.walk(cell_tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                bad = [n for n in names if n not in ALLOWED_KERNEL_IMPORTS]
                _check(not bad, f"{path.name}: cell {index} imports {bad} into the kernel; only the standard library, IPython.display and google.colab are allowed")
            if isinstance(node, ast.List) and any(isinstance(e, ast.Constant) and e.value == "pip" for e in node.elts):
                call = ast.unparse(node)
                _check(
                    call.startswith("[str(UV), 'pip', 'install', '--python', str(PYTHON), '--require-hashes'"),
                    f"{path.name}: cell {index} runs pip other than `uv pip install --python <isolated env> --require-hashes`: {call[:80]}",
                )
            if isinstance(node, ast.Name) and node.id == "urllib" and index != install_index:
                raise ValidationError(f"{path.name}: cell {index} downloads outside the isolated-install cell")


def _validate_identity(path: Path, code_cells: list[tuple[int, str, ast.Module]], carrier: int, revision: str) -> None:
    for index, source, tree in code_cells:
        if index == carrier:
            continue
        for node in ast.walk(tree):
            rebound = [name for name in _assignment_targets(node) if name in IDENTITY_NAMES]
            _check(not rebound, f"{path.name}: {rebound} must not be rebound outside the carried package (cell {index})")
        _check(revision not in source, f"{path.name}: the model revision may appear only in the carried package and manifests (cell {index})")


def _validate_parity(path: Path, notebook: dict, build) -> None:
    template = _load_tool("notebook_template").TEMPLATE
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    rendered = build.to_bytes(build.render(ROOT, template, recorded))
    current = path.read_bytes().replace(b"\r\n", b"\n")
    _check(current == rendered, f"{path.name}: differs from tools/build_notebook.py output (PAR3); regenerate")


def _validate_notebook_content(path: Path, notebook: dict, code_cells: list[tuple[int, str, ast.Module]], markdown: str, carrier: int, files: dict[str, str]) -> None:
    model_id, _revision = _package_identity()
    kernel_cells = [(index, source) for index, source, _ in code_cells if index != carrier]
    kernel = "\n".join(_strip_comments(source) for _, source in kernel_cells)
    carrier_source = next(source for index, source, _ in code_cells if index == carrier)
    missing = [marker for marker in KERNEL_CODE_MARKERS if marker not in kernel + "\n" + carrier_source]
    _check(not missing, f"{path.name}: missing required kernel-code markers: {missing}")
    for title in INFRASTRUCTURE_TITLES.values():
        _check(sum(source.startswith(title) for _, source, _ in code_cells) == 1, f"{path.name}: expected exactly one cell titled {title!r}")
    positions = []
    for call in STAGE_CALLS:
        hits = [index for index, source in kernel_cells if call in source]
        _check(len(hits) == 1, f"{path.name}: expected exactly one `{call}`, found {len(hits)}")
        positions.append(hits[0])
    _check(positions == sorted(positions), f"{path.name}: the stages must run in order {[c.split(chr(39))[1] for c in STAGE_CALLS]} (RUN1)")
    byod_cells = [source for _, source in kernel_cells if all(line in source.splitlines() for line in BYOD_FIELD_LINES)]
    _check(len(byod_cells) == 1, f"{path.name}: one learner cell must hold every BYOD form field exactly: {BYOD_FIELD_LINES} (EXE1/EXE2)")
    for stage in ("prepare", "baselines", "adapt", "evaluate", "infer", "reload"):
        _check(f"('{stage}', [" in byod_cells[0], f"{path.name}: the BYOD dataset branch must run the {stage!r} stage (DAT14)")
    for label, pattern in FORBIDDEN_PATTERNS:
        _check(not pattern.search(kernel), f"{path.name}: forbidden/insecure kernel source: {label}")
    for name, text in files.items():
        present = [label for label, pattern in CARRIED_FORBIDDEN if pattern.search(text)]
        _check(not present, f"{path.name}: forbidden/insecure source in carried {name}: {present}")
    runner = files[_load_tool("notebook_template").TEMPLATE["stage_runner"]]
    missing = [marker for marker in RUNNER_MARKERS if marker not in runner]
    _check(not missing, f"{path.name}: the carried stage runner is missing required markers: {missing}")
    missing = [name for name in EXPECTED_OUTPUTS if name not in runner]
    _check(not missing, f"{path.name}: the stage runner must export {missing}")
    _validate_gates(path, code_cells)
    missing_md = [marker for marker in COMMON_MARKDOWN_MARKERS + MARKDOWN_MARKERS if marker not in markdown]
    _check(not missing_md, f"{path.name}: missing learner-facing markers: {missing_md}")
    _check("restart" not in markdown.lower().replace("no runtime restart", "").replace("no restart", ""), f"{path.name}: learner prose must not instruct a runtime restart (RUN10)")
    _check(f"**Profile:** `{EXPECTED_PROFILE}`" in markdown, f"{path.name}: markdown must state the profile")
    _check("https://github.com/princeton-vl/RAFT-Stereo" in markdown and "arXiv:2109.07547" in markdown, f"{path.name}: references must link the upstream repository and paper of {model_id}")


def _validate_gates(path: Path, code_cells: list[tuple[int, str, ast.Module]]) -> None:
    for gate in BYOD_GATES:
        assignments = []
        for index, _source, tree in code_cells:
            for node in ast.walk(tree):
                if gate in _assignment_targets(node):
                    assignments.append((index, node))
        _check(len(assignments) == 1, f"{path.name}: {gate} must be assigned exactly once, found {len(assignments)}")


def _validate_guided_layer(path: Path, notebook: dict, markdown: str) -> None:
    """GDL1–GDL15 (§3.5): orientation, predictions, checkpoints, an optional activity, collapsed infrastructure."""
    missing = [marker for marker in GUIDED_MARKDOWN_MARKERS if marker not in markdown]
    _check(not missing, f"{path.name}: guided layer (§3.5) is missing: {missing}")
    for marker, minimum in GUIDED_MIN_COUNTS.items():
        found = markdown.count(marker)
        _check(found >= minimum, f"{path.name}: guided layer needs at least {minimum} × {marker!r}, found {found}")
    _check(markdown.count("<details>") == markdown.count("</details>"), f"{path.name}: unbalanced <details> blocks")
    untagged = [line for line in markdown.splitlines() if re.match(r"^## \d+\. ", line) and not line.rstrip().endswith(SECTION_TAGS)]
    _check(not untagged, f"{path.name}: numbered sections must end with a section tag {SECTION_TAGS} (GDL12): {untagged}")
    _check("workshop" not in markdown.lower(), f"{path.name}: learner prose must say 'notebook', not 'workshop' (GDL15)")
    cells = notebook["cells"]
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell)
        meta = cell.get("metadata", {})
        infrastructure = source.startswith("# @title Infrastructure: ")
        collapsed = meta.get("cellView") == "form" and meta.get("jupyter", {}).get("source_hidden") is True
        if infrastructure or _is_carrier(cell):
            _check(collapsed and infrastructure, f"{path.name}: infrastructure cell {index} must be titled '# @title Infrastructure: …' and collapsed (GDL11)")
        else:
            _check(not collapsed, f"{path.name}: learner cell {index} must not be collapsed")
    for number in GUIDED_PREDICT_SECTIONS:
        heads = [i for i, c in enumerate(cells) if c.get("cell_type") == "markdown" and f"## {number}. " in _cell_source(c)]
        _check(len(heads) == 1, f"{path.name}: expected one '## {number}.' section")
        _check("**Predict before running:**" in _cell_source(cells[heads[0]]).split(f"## {number}. ", 1)[1], f"{path.name}: Section {number} must ask for a prediction before it runs (GDL7)")
    glossary = markdown.split("<summary><strong>Glossary</strong>", 1)[-1]
    missing = [term for term in GLOSSARY_TERMS if f"- **{term}" not in glossary]
    _check(not missing, f"{path.name}: glossary misses {missing} (GDL6)")
    # GDL10: the change-one-thing activity changes a learner field of the adaptation cell; the field defaults to the canonical value.
    adapt = [_cell_source(c) for c in cells if c.get("cell_type") == "code" and "run_stage('adapt'" in _cell_source(c) and "--namespace" not in _cell_source(c)]
    _check(len(adapt) == 1 and 'FREEZE_ENCODERS = True  # @param {type:"boolean"}' in adapt[0], f"{path.name}: the adaptation cell must hold FREEZE_ENCODERS = True as a form field (GDL10)")


def validate_notebooks() -> None:
    tutorials = ROOT / "tutorials"
    notebooks = sorted(tutorials.glob("*.ipynb"))
    _check(len(notebooks) == 1, f"exactly one tutorial notebook is expected, found {len(notebooks)}")
    path = notebooks[0]
    _check(path.name == NOTEBOOK_NAME, f"tutorial notebook must be named {NOTEBOOK_NAME}, found {path.name}")
    build = _load_tool("build_notebook")
    notebook = json.loads(_read(path))
    code_cells, markdown = _validate_notebook_structure(path, notebook)
    carrier, files = _validate_carrier(path, notebook, code_cells, build)
    _validate_lock(path, files, build)
    _validate_isolated_install(path, code_cells, carrier)
    _model_id, revision = _package_identity()
    _validate_identity(path, code_cells, carrier, revision)
    _validate_parity(path, notebook, build)
    _validate_notebook_content(path, notebook, code_cells, markdown, carrier, files)
    _validate_guided_layer(path, notebook, markdown)
    registry = _read(tutorials / "README.md")
    _check(f"`{path.name}`" in registry, f"{path.name} missing from tutorials/README.md")
    _check(f"`{EXPECTED_PROFILE}`" in registry, f"tutorials/README.md must record `{EXPECTED_PROFILE}`")
    _check(f"DIMER Notebook Specification {NOTEBOOK_SPEC}" in registry, "tutorials/README.md must name the notebook spec version")
    _check("standalone" in registry.lower(), "tutorials/README.md must record that the notebook is standalone")
    _check("isolated" in registry.lower(), "tutorials/README.md must describe the isolated environment")


def validate_all() -> list[str]:
    validate_model_card()
    validate_pin_state()
    validate_upstream_source()
    validate_identity_consistency()
    validate_weight_facts()
    validate_release_status()
    validate_notebooks()
    return ["model-card", "pin-state", "upstream-source", "identity-consistency", "weight-facts", "release-status", "notebook+carrier+lock+parity"]


def main() -> int:
    passed = validate_all()
    print(f"release asset validation: PASS ({', '.join(passed)})")
    print("NOTE: static source validation only; not clean-runtime execution evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
