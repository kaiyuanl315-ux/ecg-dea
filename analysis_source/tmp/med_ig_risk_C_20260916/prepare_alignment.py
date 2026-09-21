"""Prepare label-blinded QRS alignment from the ECG member of inputs.npz only.

Detection matches the prior 191-encounter display pipeline. The original input
array is never changed. Output waveform arrays retain input-row order; only the
separate private alias map links that order to the shuffled review aliases.
"""

from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import warnings

BASE = Path(__file__).resolve().parent
LEGACY = BASE.parent / "med_ig_legacy_style_20260916"
os.environ.setdefault("MPLCONFIGDIR", str(BASE / "alignment" / "mplconfig"))
sys.path.insert(0, str(LEGACY / "deps"))

import numpy as np
import neurokit2 as nk
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

FS = 100
LEFT, RIGHT = 40, 60
MIN_WINDOWS = 3


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(array):
    # The shape/dtype are recorded separately; bytes use the unchanged C order.
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def freeze_json(path, obj):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != obj:
            raise RuntimeError(f"Frozen file differs: {path.name}")
    else:
        write_json(path, obj)


def detect_record(raw_ecg, alias):
    """No outcomes, risks, attribution arrays, or source identifiers enter here."""
    signal = raw_ecg[1, :].astype(float, copy=True)
    caught, flags = [], []
    error, inverted = "", None
    peaks = np.array([], dtype=int)
    finite = bool(np.isfinite(raw_ecg).all())
    if not finite:
        flags.append("nonfinite_ECG_values")
        error = "Input ECG contains nonfinite values; detector was not run."
    else:
        if np.ptp(signal) == 0:
            flags.append("flat_lead_II")
        with warnings.catch_warnings(record=True) as wrn:
            warnings.simplefilter("always")
            try:
                oriented, inverted = nk.ecg_invert(
                    signal, sampling_rate=FS, force=False, show=False
                )
                clean = nk.ecg_clean(oriented, sampling_rate=FS, method="neurokit")
                _, info = nk.ecg_peaks(
                    clean, sampling_rate=FS, method="neurokit", correct_artifacts=False
                )
                peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
                if peaks.ndim != 1 or not np.all(np.diff(peaks) > 0):
                    raise ValueError("Detected peak indices must be strictly increasing.")
                if not np.all((peaks >= 0) & (peaks < 1000)):
                    raise ValueError("Detected peak index outside the original ECG.")
            except Exception as exc:
                error = type(exc).__name__ + ": " + str(exc)
            caught = [str(w.message) for w in wrn]
    invalid = []
    if error:
        flags.append("detector_error" if finite else "input_error")
        invalid = peaks.tolist()
        peaks = np.array([], dtype=int)
    complete = peaks[(peaks >= LEFT) & (peaks + RIGHT <= 1000)]
    rr = np.diff(peaks) / FS
    if len(complete) < MIN_WINDOWS:
        flags.append("fewer_than_3_complete_windows")
    if len(rr) and rr.min() < 0.25:
        flags.append("RR_below_0.25_s")
    if len(rr) and rr.max() > 2.5:
        flags.append("RR_above_2.5_s")
    eligible = bool(len(complete) >= MIN_WINDOWS and not error)
    median = np.full((12, LEFT + RIGHT), np.nan, dtype=np.float64)
    if eligible:
        # Exactly the prior procedure: no filtering, scaling, interpolation, or
        # baseline shifting of the ECG used in the waveform summaries.
        segments = np.stack(
            [raw_ecg[:, p - LEFT : p + RIGHT].astype(float) for p in complete]
        )
        if segments.shape != (len(complete), 12, LEFT + RIGHT):
            raise AssertionError("Synchronous 12-lead window shape mismatch.")
        median = np.median(segments, axis=0)
    record = {
        "alias": alias,
        "polarity_inverted_for_detection_only": None if inverted is None else bool(inverted),
        "detected_peaks": peaks.tolist(),
        "complete_window_peaks": complete.tolist(),
        "n_detected": int(len(peaks)),
        "n_complete_windows": int(len(complete)),
        "RR_min_seconds": float(rr.min()) if len(rr) else None,
        "RR_max_seconds": float(rr.max()) if len(rr) else None,
        "flags": flags,
        "warnings": caught,
        "error": error,
        "automatic_eligibility": eligible,
        "finite_original_ECG": finite,
    }
    if invalid:
        record["invalid_detector_returned_peaks"] = invalid
    return record, median


def render_sheets(ecg, records, mapping, output):
    folder = output / "qc_blinded"
    folder.mkdir(exist_ok=True)
    by_alias = {r["alias"]: r for r in records}
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.linewidth": 0.4,
    })
    times = np.arange(1000) / FS
    n_sheets = (len(mapping) + 15) // 16
    for page in range(n_sheets):
        chunk = mapping[page * 16 : (page + 1) * 16]
        fig, axes = plt.subplots(16, 1, figsize=(12, 16), sharex=True)
        fig.subplots_adjust(left=0.12, right=0.985, bottom=0.03, top=0.977, hspace=0.40)
        fig.suptitle(
            f"Blinded QRS detection review — sheet {page + 1:02d} "
            "(red=complete window, gray=boundary)", fontsize=10, y=0.995,
        )
        for ax, entry in zip(axes, chunk):
            rec = by_alias[entry["alias"]]
            signal = ecg[entry["input_row_0based"], 1]
            ax.plot(times, signal, color="#222222", lw=0.65)
            complete = set(rec["complete_window_peaks"])
            for peak in rec["detected_peaks"]:
                color = "#bb3333" if peak in complete else "#aaaaaa"
                ax.axvline(peak / FS, color=color, lw=0.6, alpha=0.65)
                ax.scatter([peak / FS], [signal[peak]], s=7, color=color, zorder=3)
            ax.set_ylabel(rec["alias"], rotation=0, ha="right", va="center",
                          labelpad=17, fontsize=9, weight="bold")
            ax.set_xlim(0, 10)
            ax.tick_params(axis="y", labelsize=6, length=2, pad=2)
            inv = rec["polarity_inverted_for_detection_only"]
            status = f"windows={rec['n_complete_windows']} · invert={int(inv) if inv is not None else 'NA'}"
            if rec["error"]:
                status += " · DETECTION/INPUT FAILURE"
            elif rec["flags"]:
                status += " · REVIEW"
            ax.text(0.995, 0.92, status, transform=ax.transAxes, ha="right", va="top",
                    fontsize=6, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1})
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes[len(chunk):]:
            ax.set_visible(False)
        axes[len(chunk) - 1].set_xlabel("Time (s)")
        fig.savefig(folder / f"sheet_{page + 1:02d}.png", dpi=160, metadata={"Software": ""})
        plt.close(fig)
        if (page + 1) % 8 == 0 or page + 1 == n_sheets:
            print(json.dumps({"rendered_sheets": page + 1, "total_sheets": n_sheets}), flush=True)
    return n_sheets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE / "inputs.npz")
    parser.add_argument("--output-dir", type=Path, default=BASE / "alignment")
    parser.add_argument("--expected-n", type=int, default=1536)
    parser.add_argument("--alias-seed", type=int, default=20260916)
    parser.add_argument("--phase", choices=["all", "detect", "render"], default="all")
    args = parser.parse_args()
    if nk.__version__ != "0.2.13":
        raise RuntimeError(f"Expected NeuroKit2 0.2.13, found {nk.__version__}")
    source_hash = file_sha(args.input)
    # Deliberately read only this NPZ member, even if other members are present.
    with np.load(args.input, allow_pickle=False) as source:
        ecg = source["ecg"]
    if ecg.shape != (args.expected_n, 12, 1000) or ecg.dtype.kind != "f":
        raise ValueError(f"Unexpected ECG contract: shape={ecg.shape}, dtype={ecg.dtype}")
    ecg.setflags(write=False)
    fingerprint = {
        "input_file_sha256": source_hash, "npz_member_read": "ecg",
        "ECG_C_order_bytes_sha256": array_sha(ecg),
        "ECG_shape": list(ecg.shape), "ECG_dtype": str(ecg.dtype),
        "all_members_other_than_ecg_unread": True,
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if args.phase != "render" and (output / "detection_records_private.json").exists():
        raise RuntimeError("Detection output already exists; use --phase render or a new output directory.")
    freeze_json(output / "source_fingerprint.json", fingerprint)
    permutation = np.random.default_rng(args.alias_seed).permutation(args.expected_n)
    mapping = [{"alias": f"R{j+1:04d}", "input_row_0based": int(i)}
               for j, i in enumerate(permutation)]
    freeze_json(output / "private_alias_index_map.json", {
        "seed": args.alias_seed, "generator": "NumPy default_rng PCG64",
        "source_ECG_sha256": fingerprint["ECG_C_order_bytes_sha256"],
        "rows": mapping,
    })
    qa_path = output / "alignment_QA.json"
    if args.phase != "render":
        aliases = {r["input_row_0based"]: r["alias"] for r in mapping}
        records = []
        medians = np.full((args.expected_n, 12, 100), np.nan, dtype=np.float64)
        nwindows = np.zeros(args.expected_n, dtype=np.int64)
        eligible = np.zeros(args.expected_n, dtype=bool)
        for i in range(args.expected_n):
            rec, medians[i] = detect_record(ecg[i], aliases[i])
            records.append(rec)
            nwindows[i] = rec["n_complete_windows"]
            eligible[i] = rec["automatic_eligibility"]
            if (i + 1) % 128 == 0 or i + 1 == args.expected_n:
                print(json.dumps({"detected_records": i + 1, "total_records": args.expected_n}), flush=True)
        records.sort(key=lambda r: r["alias"])
        write_json(output / "detection_records_private.json", records)
        np.save(output / "waveform_medians.npy", medians, allow_pickle=False)
        np.save(output / "nwindows.npy", nwindows, allow_pickle=False)
        np.save(output / "automatic_eligible.npy", eligible, allow_pickle=False)
        qa = {
            "n": args.expected_n, "sampling_rate_hz": FS, "alignment_lead_index_0based": 1,
            "window_samples_end_exclusive": [-LEFT, RIGHT], "minimum_complete_windows": MIN_WINDOWS,
            "neurokit_version": nk.__version__, "script_sha256": file_sha(__file__),
            "previous_detector_script_sha256": file_sha(LEGACY / "prepare_aligned.py"),
            "automatic_eligible_n": int(eligible.sum()),
            "automatic_ineligible_n": int((~eligible).sum()),
            "detector_or_input_failure_n": sum(bool(r["error"]) for r in records),
            "flagged_aliases": [{k: r[k] for k in ["alias", "n_complete_windows", "flags", "error"]}
                                for r in records if r["flags"]],
            "polarity_inverted_for_detection_only_n": sum(r["polarity_inverted_for_detection_only"] is True for r in records),
            "RR_flags_are_review_only_not_automatic_exclusions": True,
            "waveform_array_order": "Original inputs.npz ECG row order; see separate private alias map.",
            "failed_or_ineligible_waveform_medians": "NaN rows retained, not dropped or padded.",
            "no_waveform_smoothing_interpolation_scaling_or_baseline_shift": True,
            "IG_or_outcome_or_risk_data_read": False,
            "all_records_preserved_in_detection_output": len(records) == args.expected_n,
            "sheet_count": 0, "visual_QC_status": "PENDING",
        }
        qa["output_sha256"] = {name: file_sha(output / name) for name in [
            "detection_records_private.json", "private_alias_index_map.json",
            "waveform_medians.npy", "nwindows.npy", "automatic_eligible.npy",
        ]}
        write_json(qa_path, qa)
    else:
        records = json.loads((output / "detection_records_private.json").read_text())
        qa = json.loads(qa_path.read_text())
        for name, expected in qa["output_sha256"].items():
            if file_sha(output / name) != expected:
                raise RuntimeError(f"Frozen detection output changed: {name}")
    if args.phase in ["all", "render"]:
        qa["sheet_count"] = render_sheets(ecg, records, mapping, output)
        qa["all_records_shown_in_blinded_sheets"] = len(records) == args.expected_n
    if file_sha(args.input) != source_hash or array_sha(ecg) != fingerprint["ECG_C_order_bytes_sha256"]:
        raise RuntimeError("Source ECG file or in-memory array changed during processing.")
    qa["source_file_and_array_unchanged"] = True
    write_json(qa_path, qa)
    print(json.dumps({k: qa[k] for k in ["n", "automatic_eligible_n", "detector_or_input_failure_n",
                                       "sheet_count", "visual_QC_status"]}), flush=True)


if __name__ == "__main__":
    main()
