"""Freeze a representative within-risk sample, then extract verified inputs.

Internal research artifacts only: metadata and permutation files contain source
identifiers or row mappings. Selection uses fixed Fusion-DL risk membership and
separate deterministic random permutations, never death labels or old IG sample
membership. Larger requested sizes use nested prefixes of those permutations.
No model imports, checkpoint loading, inference, training, or IG calculation.
"""
from __future__ import annotations

import argparse
import hashlib
import ast
import io
import json
import platform
import struct
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
ORIGINAL = ROOT / "tmp/med_ig_20260915"
REFERENCE_SHA = "dea7ec863e5117fbc2074a62090b86fdfb86a2c5d1d2d1843ba86cbd128eddb3"
ORIGINAL_PREPARE_SHA = "be0561122ee0eedc7ad355f2a143506167712c0e5c66150174a36464123fc3d2"
DEFAULT_ARCHIVE = Path("/path/to/research/aiecg/data/multi_ecg_x.npz")
SEED = 20260916
RISK_STREAM_KEYS = {"high": 1, "low": 2}
RISK_ORDER = ["high", "low"]
ALL_RISKS = ["low", "intermediate", "high"]
EXPECTED_CELLS = {
    (0, "low"): 27008, (0, "intermediate"): 7000, (0, "high"): 1638,
    (1, "low"): 31, (1, "intermediate"): 127, (1, "high"): 144,
}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_frozen(path, content):
    """Idempotent identical bytes are allowed; existing selections cannot change."""
    payload = content.encode() if isinstance(content, str) else content
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite a different frozen artifact: {path.name}")
        return
    with path.open("xb") as f:
        f.write(payload)


def save_arrays_frozen(path, **arrays):
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != set(arrays) or any(
                z[k].dtype != v.dtype or not np.array_equal(z[k], v) for k, v in arrays.items()
            ):
                raise RuntimeError(f"Refusing to overwrite a different frozen array archive: {path.name}")
        return
    # Use a staged new file; never mutate an existing input artifact.
    staged = path.with_name(path.name + ".partial")
    with staged.open("xb") as f:
        np.savez(f, **arrays)
    staged.rename(path)


def readonly_npz_memmap(archive, member, expected_header):
    """Adapted from the verified IG helper without importing its model runtime."""
    with zipfile.ZipFile(archive) as z:
        info = z.getinfo(member)
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError("Direct bounded extraction requires an uncompressed NPY member")
    with archive.open("rb") as f:
        f.seek(info.header_offset)
        local_header = f.read(30)
        if local_header[:4] != b"PK\x03\x04":
            raise ValueError("Invalid ZIP local header")
        filename_n, extra_n = struct.unpack("<HH", local_header[26:30])
        f.seek(filename_n + extra_n, 1)
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError("Unsupported NPY header version")
        offset = f.tell()
    assert list(shape) == expected_header["shape"]
    assert str(dtype) == expected_header["dtype"]
    assert fortran == expected_header["fortran_order"]
    return np.memmap(archive, mode="r", dtype=dtype, shape=shape, offset=offset,
                     order="F" if fortran else "C")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high-n", type=int, required=True)
    parser.add_argument("--low-n", type=int, required=True)
    parser.add_argument("--npz", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=BASE)
    args = parser.parse_args()
    sizes = {"high": args.high_n, "low": args.low_n}
    assert all(n > 0 for n in sizes.values())
    out = args.output_dir.resolve()
    assert out != ORIGINAL.resolve(), "Never write to the prior IG analysis directory"
    out.mkdir(parents=True, exist_ok=True)

    # Reuse the original safe header parser, with both its file and the original
    # reference specification hash-pinned. Execute just that function's AST;
    # avoid running its main or writing an import cache in the old directory.
    ref_path = ORIGINAL / "sampling_spec.json"
    helper_path = ORIGINAL / "prepare_sample_metadata.py"
    assert sha_file(ref_path) == REFERENCE_SHA
    assert sha_file(helper_path) == ORIGINAL_PREPARE_SHA
    helper_ast = ast.parse(helper_path.read_text())
    helper_nodes = [node for node in helper_ast.body if isinstance(node, ast.FunctionDef) and node.name == "header_info"]
    assert len(helper_nodes) == 1
    helper_namespace = {"np": np, "zipfile": zipfile}
    exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), str(helper_path), "exec"), helper_namespace)
    header_info = helper_namespace["header_info"]
    reference = json.loads(ref_path.read_text())
    paths = {k: Path(v["path"]) for k, v in reference["source_files"].items()}
    hashes = {k: sha_file(p) for k, p in paths.items()}
    assert all(hashes[k] == reference["source_files"][k]["sha256"] for k in paths)
    locked_hashes = json.loads(paths["locked_source_manifest"].read_text())
    for k in ["predictions", "cohort"]:
        assert hashes[k] == locked_hashes[str(paths[k])]

    # Preserve score strings verbatim. Numeric casts are for membership/QA only.
    pred = pd.read_csv(paths["predictions"], dtype=str)
    cohort_cols = ["file_name", "subject_id", "death_72h", "general_strat_fold", "age", "gender", "gender_unknown"]
    cohort = pd.read_csv(paths["cohort"], usecols=cohort_cols, dtype=str)
    assert not pred.file_name.duplicated().any() and not cohort.file_name.duplicated().any()
    pred["prediction_row_0based"] = np.arange(len(pred))
    cohort["cohort_row_0based"] = np.arange(len(cohort))
    cohort["general_strat_fold"] = cohort.general_strat_fold.astype(int)
    test = cohort[cohort.general_strat_fold.isin([8, 9])].copy()
    test["test_partition_row_0based"] = np.arange(len(test))

    names = ["arr_0.npy", "file_name.npy", "subject_id.npy", "x_test.npy", "y_test_72_death.npy"]
    read_members = {}
    with zipfile.ZipFile(args.npz) as z:
        headers = {name: header_info(z, name) for name in names}
        assert headers == reference["archive"]["headers"]
        def small(name):
            assert name != "arr_0.npy" and z.getinfo(name).file_size < 10 * 1024 * 1024
            raw = z.read(name)
            read_members[name] = {"sha256_of_npy_member": hashlib.sha256(raw).hexdigest(), "bytes_read": len(raw)}
            if name in reference["archive"]["fully_read_members"]:
                assert read_members[name] == reference["archive"]["fully_read_members"][name]
            return np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)
        archive_ids = small("file_name.npy").astype(str)
        archive_subjects = small("subject_id.npy").astype(str)
        archive_y = small("y_test_72_death.npy")
        # The 1.7-MB processed tabular matrix is small enough for full CRC/hash QA.
        tabular = small("x_test.npy")
    assert args.npz.stat().st_size == reference["archive"]["bytes"]
    assert len(set(archive_ids)) == len(archive_ids) == 180686
    assert np.array_equal(archive_ids, cohort.file_name.to_numpy())
    assert np.array_equal(archive_subjects, cohort.subject_id.to_numpy())
    assert np.array_equal(test.file_name.to_numpy(), pred.file_name.to_numpy())
    assert np.array_equal(test.subject_id.to_numpy(), pred.subject_id.to_numpy())
    assert np.array_equal(test.death_72h.astype(int), pred.death_72h.astype(int))
    assert np.array_equal(archive_y, pred.death_72h.astype(int).to_numpy())
    assert len(pred) == 35948 and pred.death_72h.astype(int).sum() == 302
    assert pred.subject_id.nunique() == 19595
    assert tabular.shape == (35948, 12) and tabular.dtype == np.float32 and np.isfinite(tabular).all()
    run_cols = [f"proba_run{i}" for i in range(10)]
    delta = float(np.max(np.abs(pred[run_cols].astype(float).mean(axis=1) - pred.proba_mean.astype(float))))
    assert delta < 1e-12
    frozen = json.loads(paths["frozen_thresholds"].read_text())
    low, high = frozen["thresholds_raw"]["low"], frozen["thresholds_raw"]["high"]
    assert low == 0.456303122639656 and high == 0.6162797212600708
    scores = pred.proba_mean.astype(float).to_numpy()
    pred["risk_stratum"] = np.where(scores < low, "low", np.where(scores < high, "intermediate", "high"))
    pred["death_72h"] = pred.death_72h.astype(int)
    assert pred.groupby(["death_72h", "risk_stratum"]).size().to_dict() == EXPECTED_CELLS
    archive_map = {v: i for i, v in enumerate(archive_ids)}
    pred["npz_ecg_row_0based"] = pred.file_name.map(archive_map)
    assert pred.npz_ecg_row_0based.notna().all()
    pred["npz_tabular_test_row_0based"] = np.arange(len(pred))
    pred["npz_label_test_row_0based"] = np.arange(len(pred))
    pred["test_partition_row_0based"] = np.arange(len(pred))
    for col in ["cohort_row_0based", "general_strat_fold", "age", "gender", "gender_unknown"]:
        pred[col] = test[col].to_numpy()
    assert np.array_equal(pred.npz_ecg_row_0based, pred.cohort_row_0based)
    locked = pd.read_csv(paths["locked_risk_strata"])
    risk_col = next(c for c in ["stratum", "risk_stratum", "risk_group"] if c in locked.columns)
    n_col = next(c for c in ["group_n", "n", "encounters", "n_encounters"] if c in locked.columns)
    event_col = next(c for c in ["group_events", "events", "deaths"] if c in locked.columns)
    for risk in ALL_RISKS:
        row = locked[locked[risk_col].str.lower() == risk].iloc[0]
        assert int(row[n_col]) == int((pred.risk_stratum == risk).sum())
        assert int(row[event_col]) == int(pred.loc[pred.risk_stratum == risk, "death_72h"].sum())

    # Do not inspect death labels, waveform appearance, existing IG availability,
    # or original-sample membership when choosing rows. Separate streams make
    # one group's prefix independent of the requested size of the other group.
    selected_blocks, permutations, group_specs = [], {}, []
    for risk in RISK_ORDER:
        candidates = pred.index[pred.risk_stratum == risk].to_numpy(dtype=np.int64)
        assert sizes[risk] <= len(candidates)
        seed_sequence = np.random.SeedSequence([SEED, RISK_STREAM_KEYS[risk]])
        rng = np.random.Generator(np.random.PCG64(seed_sequence))
        perm = rng.permutation(candidates)
        chosen = perm[:sizes[risk]]
        assert len(set(chosen.tolist())) == sizes[risk]
        permutations[risk + "_prediction_rows"] = perm
        block = pred.loc[chosen].copy()
        block["sample_rank_within_risk"] = np.arange(1, len(block) + 1)
        selected_blocks.append(block)
        group_specs.append({"risk_stratum": risk, "population_encounters": len(candidates),
                            "sample_encounters": sizes[risk], "sampling_fraction": sizes[risk] / len(candidates),
                            "stream_seed_entropy": [SEED, RISK_STREAM_KEYS[risk]],
                            "full_permutation_sha256": sha_array(perm), "take_all": sizes[risk] == len(candidates)})
    chosen = pd.concat(selected_blocks).reset_index(drop=True)
    chosen.insert(0, "sample_row_0based", np.arange(len(chosen)))
    assert chosen.file_name.nunique() == len(chosen) == args.high_n + args.low_n
    columns = ["sample_row_0based", "sample_rank_within_risk", "file_name", "subject_id", "death_72h", "risk_stratum",
               "prediction_row_0based", "test_partition_row_0based", "npz_ecg_row_0based", "npz_tabular_test_row_0based",
               "npz_label_test_row_0based", "cohort_row_0based", "general_strat_fold", "age", "gender", "gender_unknown",
               "proba_mean", "proba_std"] + run_cols
    metadata_path = out / "sample_metadata.csv"
    write_frozen(metadata_path, chosen[columns].to_csv(index=False))
    save_arrays_frozen(out / "risk_permutations_private.npz", **permutations)
    # Selected IDs and their order are now immutable, before any ECG extraction.
    sample_counts = []
    for risk in RISK_ORDER:
        block = chosen[chosen.risk_stratum == risk]
        sample_counts.append({"risk_stratum": risk, "encounters": len(block), "patients": int(block.subject_id.nunique()),
                              "death_72h_events": int(block.death_72h.sum())})
    spec = {
        "purpose": "Post hoc representative within-risk sample for current Fusion-DL high-versus-low C display and nested-prefix stability checks.",
        "privacy": "INTERNAL ONLY: metadata and full permutations include research identifiers or source-row mappings; do not publish them.",
        "source_files": {k: {"path": str(p), "sha256": hashes[k]} for k, p in paths.items()},
        "original_source_verification_reference_sha256": REFERENCE_SHA,
        "archive": {"path": str(args.npz), "bytes": args.npz.stat().st_size, "headers": headers,
                    "fully_read_members": read_members, "full_archive_sha256_computed": False,
                    "arr_0_crc_recomputed": False, "ECG_extraction": "Selected rows only via read-only memory map; full ECG payload is not materialized."},
        "population": {"encounters": len(pred), "patients": int(pred.subject_id.nunique()), "death_72h_events": 302,
                       "risk_counts": {r: int((pred.risk_stratum == r).sum()) for r in ALL_RISKS}},
        "thresholds": {"score_field": "proba_mean", "raw_low_exact_decimal": "0.456303122639656",
                       "raw_high_exact_decimal": "0.6162797212600708", "low_rule": "score < low threshold",
                       "high_rule": "score >= high threshold", "source": str(paths["frozen_thresholds"])},
        "sampling": {"seed": SEED, "bit_generator": "PCG64", "draw_api": "Independent full Generator.permutation per risk, then prefix",
                     "unit": "encounter", "candidate_order": "Ascending zero-based locked prediction row",
                     "group_order": RISK_ORDER, "within_group_output_order": "Random permutation rank, not re-sorted",
                     "sample_rank_within_risk_base": 1, "group_specs": group_specs,
                     "outcome_used_for_selection": False, "old_sample_overlap_forced": False,
                     "nested_prefix_rule": "Within each risk group, use sample_rank_within_risk <= requested prefix size. Larger samples use the same frozen full permutation.",
                     "sample_encounters": len(chosen), "sample_patients": int(chosen.subject_id.nunique()),
                     "post_selection_outcome_counts": sample_counts,
                     "interpretation": "Each risk sample is uniform without replacement in that risk population. Groups need not have equal n and are not matched pairs; QC can change the retained target population."},
        "input_interface": {"ecg": "arr_0 rows joined by exact unique file_name", "tab": "Stored x_test rows, all 12 processed columns unchanged; do not reconstruct masks from cohort CSV",
                            "sample_row_base": 0, "sample_order": "High-risk permutation prefix followed by low-risk permutation prefix",
                            "score_strings": "Preserved from locked prediction CSV in metadata", "input_archive_keys": ["ecg", "tab", "y", "expected", "expected_mean"]},
        "qa": {"all_locked_source_hashes_match": True, "archive_headers_and_original_small_member_hashes_match": True,
               "all_archive_ID_patient_cohort_mappings_match": True, "test_order_and_labels_match": True,
               "all_sample_IDs_unique": True, "risk_counts_and_events_match_locks": True,
               "maximum_saved_ensemble_vs_ten_member_mean_difference": delta,
               "sample_metadata_sha256": sha_file(metadata_path), "sample_frozen_before_ECG_extraction": True,
               "checkpoint_loaded": False, "inference_run": False},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "script_sha256": sha_file(Path(__file__)),
    }
    write_frozen(out / "sampling_spec.json", json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"event": "sample_frozen", "sample_n": len(chosen), "groups": sample_counts,
                      "metadata_sha256": sha_file(metadata_path)}, ensure_ascii=False), flush=True)

    mapped_ecg = readonly_npz_memmap(args.npz, "arr_0.npy", headers["arr_0.npy"])
    ecg = np.array(mapped_ecg[chosen.npz_ecg_row_0based.to_numpy(dtype=int)], dtype=np.float32)
    tab = np.array(tabular[chosen.npz_tabular_test_row_0based.to_numpy(dtype=int)], dtype=np.float32)
    labels = np.array(archive_y[chosen.npz_label_test_row_0based.to_numpy(dtype=int)], dtype=np.int64)
    expected = chosen[run_cols].to_numpy(dtype=np.float64)
    expected_mean = chosen.proba_mean.to_numpy(dtype=np.float64)
    assert ecg.shape == (len(chosen), 12, 1000) and tab.shape == (len(chosen), 12)
    assert np.isfinite(ecg).all() and np.isfinite(tab).all()
    assert np.array_equal(labels, chosen.death_72h.to_numpy())
    assert np.array_equal(archive_ids[chosen.npz_ecg_row_0based], chosen.file_name)
    assert np.array_equal(archive_subjects[chosen.npz_ecg_row_0based], chosen.subject_id)
    save_arrays_frozen(out / "inputs.npz", ecg=ecg, tab=tab, y=labels, expected=expected, expected_mean=expected_mean)
    # Re-open the exported arrays and verify exact selected source-row equality.
    with np.load(out / "inputs.npz", allow_pickle=False) as z:
        assert np.array_equal(z["ecg"], mapped_ecg[chosen.npz_ecg_row_0based.to_numpy(dtype=int)])
        assert np.array_equal(z["tab"], tabular[chosen.npz_tabular_test_row_0based.to_numpy(dtype=int)])
        assert np.array_equal(z["y"], labels) and np.array_equal(z["expected"], expected)
    manifest = {"status": "INPUTS_VERIFIED", "n": len(chosen), "groups": sample_counts,
                "ecg_shape": list(ecg.shape), "tab_shape": list(tab.shape),
                "ecg_sha256": sha_array(ecg), "tab_sha256": sha_array(tab), "labels_sha256": sha_array(labels),
                "expected_member_scores_sha256": sha_array(expected), "expected_mean_scores_sha256": sha_array(expected_mean),
                "sample_metadata_sha256": sha_file(metadata_path), "sampling_spec_sha256": sha_file(out / "sampling_spec.json"),
                "inputs_npz_sha256": sha_file(out / "inputs.npz"),
                "risk_permutations_sha256": sha_file(out / "risk_permutations_private.npz"),
                "extracted_ECG_and_tab_exactly_equal_selected_source_rows": True,
                "ECG_filtering_reordering_scaling_or_clipping_applied": False,
                "tabular_masks_reconstructed": False, "full_ECG_array_loaded": False,
                "sample_frozen_before_extraction": True, "model_or_IG_execution": False}
    write_frozen(out / "input_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"event": "inputs_verified", "n": len(chosen), "ecg_shape": list(ecg.shape),
                      "tab_shape": list(tab.shape), "inputs_sha256": manifest["inputs_npz_sha256"],
                      "full_ECG_array_loaded": False, "model_or_IG_execution": False}), flush=True)


if __name__ == "__main__":
    main()
