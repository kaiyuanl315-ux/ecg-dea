"""Prepare an internal-only, stratified metadata sample without loading ECGs.

Run with the bundled Python. This script reads only small ID and label members
of the current NPZ. It never materializes arr_0, loads a checkpoint, or runs a
model. The resulting CSV contains research IDs and is not a public artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = Path('/path/to/research/aiecg/data/multi_ecg_x.npz')
SEED = 20260915
PER_CELL = 32
RISK_ORDER = ['low', 'intermediate', 'high']
EXPECTED = {
    (0, 'low'): 27008, (0, 'intermediate'): 7000, (0, 'high'): 1638,
    (1, 'low'): 31, (1, 'intermediate'): 127, (1, 'high'): 144,
}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def header_info(z: zipfile.ZipFile, member: str) -> dict:
    info = z.getinfo(member)
    with z.open(member) as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f'Unsupported NPY header version {version}')
        payload_offset = stream.tell()
    return {
        'shape': list(shape), 'dtype': str(dtype), 'fortran_order': fortran,
        'member_uncompressed_bytes': info.file_size,
        'member_compressed_bytes': info.compress_size,
        'compression_type': info.compress_type,
        'central_directory_crc32': f'{info.CRC:08x}',
        'npy_payload_offset_within_member': payload_offset,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npz', type=Path, default=DEFAULT_NPZ)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        'predictions': ROOT / 'data/proba_test_72.csv',
        'cohort': ROOT / 'data/ecg_muti.csv',
        'locked_source_manifest': ROOT / 'project_control/REVISION_ANALYSIS_20260906/batch1_verified/source_manifest.json',
        'frozen_thresholds': ROOT / 'project_control/REVISION_ANALYSIS_20260906/batch1_verified/CALIBRATOR_AND_THRESHOLDS_FROZEN.json',
        'locked_risk_strata': ROOT / 'project_control/REVISION_ANALYSIS_20260906/batch1_verified/internal_test/risk_strata.csv',
    }
    file_hashes = {name: sha_file(path) for name, path in paths.items()}
    locked_hashes = json.loads(paths['locked_source_manifest'].read_text())
    for name in ['predictions', 'cohort']:
        assert file_hashes[name] == locked_hashes[str(paths[name])], f'Locked {name} changed'

    # Keep original score strings in the CSV; numerical casts are only for QA
    # and membership assignment, not for replacing the locked predictions.
    p = pd.read_csv(paths['predictions'], dtype=str)
    columns = ['file_name', 'subject_id', 'death_72h', 'general_strat_fold', 'age', 'gender', 'gender_unknown']
    m = pd.read_csv(paths['cohort'], usecols=columns, dtype=str)
    assert not p.file_name.duplicated().any()
    assert not m.file_name.duplicated().any()
    p['prediction_row_0based'] = np.arange(len(p))
    m['cohort_row_0based'] = np.arange(len(m))
    m['general_strat_fold'] = m.general_strat_fold.astype(int)
    t = m[m.general_strat_fold.isin([8, 9])].copy()
    t['test_partition_row_0based'] = np.arange(len(t))

    members_read = {}
    with zipfile.ZipFile(args.npz) as z:
        archive_headers = {name: header_info(z, name) for name in
                           ['arr_0.npy', 'file_name.npy', 'subject_id.npy', 'x_test.npy', 'y_test_72_death.npy']}
        def small_member(name: str):
            assert name != 'arr_0.npy'
            assert z.getinfo(name).file_size < 10 * 1024 * 1024
            raw = z.read(name)
            members_read[name] = {'sha256_of_npy_member': hashlib.sha256(raw).hexdigest(), 'bytes_read': len(raw)}
            return np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)
        file_name_all = small_member('file_name.npy').astype(str)
        subject_id_all = small_member('subject_id.npy').astype(str)
        y_test = small_member('y_test_72_death.npy')

    assert archive_headers['arr_0.npy']['shape'] == [180686, 12, 1000]
    assert archive_headers['arr_0.npy']['dtype'] == 'float32'
    assert archive_headers['x_test.npy']['shape'] == [35948, 12]
    assert len(set(file_name_all)) == len(file_name_all)
    assert np.array_equal(file_name_all, m.file_name.to_numpy())
    assert np.array_equal(subject_id_all, m.subject_id.to_numpy())
    assert np.array_equal(t.file_name.to_numpy(), p.file_name.to_numpy())
    assert np.array_equal(t.subject_id.to_numpy(), p.subject_id.to_numpy())
    assert np.array_equal(t.death_72h.astype(int).to_numpy(), p.death_72h.astype(int).to_numpy())
    assert np.array_equal(y_test, p.death_72h.astype(int).to_numpy())
    assert len(p) == 35948 and p.death_72h.astype(int).sum() == 302
    assert p.subject_id.nunique() == 19595
    runs = [f'proba_run{i}' for i in range(10)]
    ensemble_delta = float(np.max(np.abs(p[runs].astype(float).mean(axis=1) - p.proba_mean.astype(float))))
    assert ensemble_delta < 1e-12

    frozen = json.loads(paths['frozen_thresholds'].read_text())
    low, high = frozen['thresholds_raw']['low'], frozen['thresholds_raw']['high']
    assert low == 0.456303122639656 and high == 0.6162797212600708
    scores = p.proba_mean.astype(float).to_numpy()
    p['risk_stratum'] = np.where(scores < low, 'low', np.where(scores < high, 'intermediate', 'high'))
    p['death_72h'] = p.death_72h.astype(int)
    archive_map = {file_name: i for i, file_name in enumerate(file_name_all)}
    p['npz_ecg_row_0based'] = p.file_name.map(archive_map)
    assert p.npz_ecg_row_0based.notna().all()
    p['npz_tabular_test_row_0based'] = np.arange(len(p))
    p['npz_label_test_row_0based'] = np.arange(len(p))
    for col in ['cohort_row_0based', 'general_strat_fold', 'age', 'gender', 'gender_unknown']:
        p[col] = t[col].to_numpy()
    assert np.array_equal(p.npz_ecg_row_0based, p.cohort_row_0based)

    locked_strata = pd.read_csv(paths['locked_risk_strata'])
    risk_column = next(c for c in ['stratum', 'risk_stratum', 'risk_group'] if c in locked_strata.columns)
    n_column = next(c for c in ['group_n', 'n', 'encounters', 'n_encounters'] if c in locked_strata.columns)
    event_column = next(c for c in ['group_events', 'events', 'deaths'] if c in locked_strata.columns)
    for risk in RISK_ORDER:
        row = locked_strata[locked_strata[risk_column].str.lower() == risk].iloc[0]
        assert int(row[n_column]) == int((p.risk_stratum == risk).sum())
        assert int(row[event_column]) == int(p.loc[p.risk_stratum == risk, 'death_72h'].sum())

    rng = np.random.Generator(np.random.PCG64(SEED))
    selected, cell_specs = [], []
    for status in [0, 1]:
        for risk in RISK_ORDER:
            candidates = p.index[(p.death_72h == status) & (p.risk_stratum == risk)].to_numpy()
            assert len(candidates) == EXPECTED[(status, risk)]
            n = min(PER_CELL, len(candidates))
            # Candidate order is the locked prediction CSV order. A complete
            # cell consumes no RNG draw, while sampled cells use choice once.
            chosen = candidates if n == len(candidates) else np.sort(rng.choice(candidates, size=n, replace=False))
            selected.extend(chosen.tolist())
            cell_specs.append({'death_72h': status, 'risk_stratum': risk, 'population_encounters': len(candidates),
                               'sample_encounters': n, 'take_all': n == len(candidates)})
    sampled = p.loc[selected].copy().reset_index(drop=True)
    sampled.insert(0, 'sample_row_0based', np.arange(len(sampled)))
    assert len(sampled) == 191 and sampled.file_name.nunique() == 191
    cols = ['sample_row_0based', 'file_name', 'subject_id', 'death_72h', 'risk_stratum',
            'prediction_row_0based', 'test_partition_row_0based', 'npz_ecg_row_0based',
            'npz_tabular_test_row_0based', 'npz_label_test_row_0based', 'cohort_row_0based',
            'general_strat_fold', 'age', 'gender', 'gender_unknown', 'proba_mean', 'proba_std'] + runs
    sampled['test_partition_row_0based'] = sampled.npz_tabular_test_row_0based
    sampled[cols].to_csv(out / 'sample_metadata.csv', index=False)

    spec = {
        'purpose': 'Reproducible small sample for current Fusion-DL Integrated Gradients input and output QA. No model execution is performed by this script.',
        'privacy': 'sample_metadata.csv contains internal research IDs; do not publish IDs, timestamps, or this internal CSV in figure labels or public supplementary materials.',
        'source_files': {name: {'path': str(path), 'sha256': file_hashes[name]} for name, path in paths.items()},
        'archive': {'path': str(args.npz), 'bytes': args.npz.stat().st_size,
                    'full_archive_sha256_computed': False,
                    'headers': archive_headers, 'fully_read_members': members_read,
                    'arr_0_payload_loaded': False, 'arr_0_crc_recomputed': False},
        'population': {'encounters': len(p), 'death_72h_events': int(p.death_72h.sum()), 'patients': int(p.subject_id.nunique()),
                       'folds': [8, 9], 'mortality_origin': 'ED arrival', 'mortality_horizon_hours': 72,
                       'risk_counts': {r: int((p.risk_stratum == r).sum()) for r in RISK_ORDER}},
        'thresholds': {'score_field': 'proba_mean', 'score_definition': 'Arithmetic mean of proba_run0 through proba_run9; locked current Fusion-DL ensemble scores.',
                       'raw_low_exact_decimal': '0.456303122639656', 'raw_high_exact_decimal': '0.6162797212600708',
                       'low_rule': 'score < low threshold', 'intermediate_rule': 'low threshold <= score < high threshold',
                       'high_rule': 'score >= high threshold', 'threshold_source': str(paths['frozen_thresholds'])},
        'sampling': {'seed': SEED, 'numpy_bit_generator': 'PCG64', 'draw_api': 'Generator.choice(replace=False)',
                     'unit': 'encounter', 'max_per_cell': PER_CELL, 'group_order': 'death_72h 0 then 1, each low/intermediate/high',
                     'candidate_order': 'ascending zero-based locked prediction CSV row',
                     'within_cell_output_order': 'selected rows sorted in ascending prediction row',
                     'take_all_policy': 'Include every candidate if <=32; take-all groups consume no RNG draw.',
                     'sample_encounters': len(sampled), 'sample_patients': int(sampled.subject_id.nunique()),
                     'cells': cell_specs,
                     'interpretation': 'Deliberate balance across outcome-by-risk cells; unweighted pooled summaries do not estimate population-average attribution. Summarize cells separately or declare aggregation weights.'},
        'input_interface': {
            'row_base': 0,
            'ecg': 'Read arr_0[npz_ecg_row_0based, :, :] from archive; stored shape per encounter is 12 x 1000 float32.',
            'tabular': 'Read x_test[npz_tabular_test_row_0based, :] from the same archive. Stored row has 12 columns; apply the current model input-column specification separately.',
            'label': 'Read y_test_72_death[npz_label_test_row_0based]. It equals the CSV death_72h.',
            'prediction_row': 'prediction_row_0based and test_partition_row_0based index the locked proba_test_72.csv and source cohort folds 8–9 in original order.',
            'identifier_map': 'file_name.npy indexes arr_0 rows; exact unique file_name join. subject_id.npy confirms patient alignment.',
            'sample_order': 'sample_row_0based is the required output-array ordering for the 191 selected encounters.',
            'score_columns': 'Original per-run and ensemble strings are copied from locked prediction CSV for inference comparison.',
            'clinical_columns': 'age, gender and gender_unknown are copied for internal sample metadata only; model inference must use the archive tabular rows and verified current input specification.',
        },
        'qa': {'locked_csv_hashes_match': True, 'archive_all_ids_and_subjects_match_cohort_order': True,
               'locked_prediction_test_order_matches_cohort_fold_order': True,
               'npz_y_test_matches_locked_csv_labels_exactly': True,
               'all_sample_ids_unique': True, 'all_sample_ecg_tabular_label_indices_verified': True,
               'locked_risk_stratum_counts_and_events_match': True,
               'maximum_saved_ensemble_vs_ten_run_mean_difference': ensemble_delta,
               'checkpoint_loaded': False, 'inference_run': False,
               'sample_metadata_sha256': sha_file(out / 'sample_metadata.csv')},
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
        'script_sha256': sha_file(Path(__file__)),
    }
    (out / 'sampling_spec.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'population': spec['population'], 'sampling_cells': cell_specs,
                      'sample_encounters': len(sampled), 'sample_patients': spec['sampling']['sample_patients'],
                      'metadata_sha256': spec['qa']['sample_metadata_sha256'],
                      'arr_0_payload_loaded': False, 'output_dir': str(out)}, indent=2))


if __name__ == '__main__':
    main()
