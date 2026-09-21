"""Independent, read-only numerical audit of saved Fusion-DL IG outputs.

No model imports, checkpoint loading, or inference. In-progress run folders are
reported as pending. Public CSVs contain study-local case numbers, never source
research IDs, dates, or archive row indices.

Example:
  python audit_ig.py --primary-run zero_128 --refinement-run refine_256 \
    --refinement-run refine_512 --final-arrays sample_arrays.npz \
    --compare nodes128_256=zero_128:zero_256 \
    --compare baseline=zero_256:lead_mean_256

Alternatively --primary-manifest accepts {"cases": [{"sample_row_0based": 0,
"source_run": "zero_128"}, ...]}. All 191 cases must have one source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ATOL, RTOL = 0.0005, 0.02
RISK_ORDER = ['low', 'intermediate', 'high']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def summarize(a):
    x = np.asarray(a, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {'finite_n': 0, 'min': None, 'q25': None, 'median': None, 'q75': None, 'max': None, 'mean': None}
    q = np.quantile(x, [0, .25, .5, .75, 1])
    return {'finite_n': len(x), **dict(zip(['min', 'q25', 'median', 'q75', 'max'], map(float, q))), 'mean': float(x.mean())}


def safe_ratio(n, d):
    return float(n / d) if d != 0 else None


def spearman(a, b):
    ra = pd.Series(a).rank(method='average').to_numpy()
    rb = pd.Series(b).rank(method='average').to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def lead_shares(ig):
    totals = np.abs(ig.astype(np.float64)).sum(axis=2)
    denominator = totals.sum(axis=1, keepdims=True)
    return np.divide(totals, denominator, out=np.full_like(totals, np.nan), where=denominator != 0)


def case_identity(metadata, index):
    row = metadata.iloc[int(index)]
    return {'case_number': f'Case{int(index) + 1:03d}', 'death_72h': int(row.death_72h), 'risk_stratum': row.risk_stratum}


def completeness_rows(metadata, label, indices, ig, pred, base, **extra):
    attribution_sum = ig.sum(axis=(1, 2), dtype=np.float64)
    difference = pred - base
    residual = attribution_sum - difference
    tolerance = ATOL + RTOL * np.abs(difference)
    rows = []
    for j, idx in enumerate(indices):
        rows.append({**case_identity(metadata, idx), 'analysis': label, **extra,
                     'prediction': float(pred[j]), 'baseline_prediction': float(base[j]),
                     'output_difference': float(difference[j]), 'signed_attribution_sum': float(attribution_sum[j]),
                     'completeness_residual': float(residual[j]), 'absolute_residual': float(abs(residual[j])),
                     'candidate_tolerance': float(tolerance[j]),
                     'within_candidate_tolerance': bool(abs(residual[j]) <= tolerance[j]),
                     'residual_over_tolerance': float(abs(residual[j]) / tolerance[j]),
                     'absolute_residual_over_absolute_output_difference': safe_ratio(abs(residual[j]), abs(difference[j]))})
    return rows


def completeness_summary(rows):
    return {'n': len(rows), 'outside_candidate_tolerance': sum(not r['within_candidate_tolerance'] for r in rows),
            'absolute_residual': summarize([r['absolute_residual'] for r in rows]),
            'residual_over_tolerance': summarize([r['residual_over_tolerance'] for r in rows])}


def audit_run(path, inputs, metadata, input_hash, source_hash, checkpoint_hashes):
    tag = path.name
    needed = [path / 'manifest.json', path / 'ensemble.npz'] + [path / f'member_{k:02d}.npz' for k in range(10)]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        return None, {'status': 'PENDING', 'missing': missing}, [], []
    manifest = json.loads((path / 'manifest.json').read_text())
    indices = np.asarray(manifest['sample_indices'], dtype=int)
    assert indices.ndim == 1 and len(indices) == len(set(indices.tolist()))
    assert ((indices >= 0) & (indices < len(metadata))).all()
    shape = (len(indices), 12, 1000)
    signed_mean = np.zeros(shape, dtype=np.float64)
    signed_sum = np.zeros(shape, dtype=np.float64)
    mean_member_absolute = np.zeros(shape, dtype=np.float64)
    predictions, baselines, per_member_rows, member_reports = [], [], [], []
    for k in range(10):
        file = path / f'member_{k:02d}.npz'
        with np.load(file, allow_pickle=False) as z:
            a = z['ig']; px = z['prediction']; pb = z['baseline_prediction']
            assert a.shape == shape and px.shape == pb.shape == (len(indices),)
            assert np.isfinite(a).all() and np.isfinite(px).all() and np.isfinite(pb).all()
            assert np.array_equal(z['indices'], indices)
            assert int(z['steps']) == manifest['steps']
            provenance = json.loads(str(z['provenance']))
            assert provenance['input_sha256'] == input_hash
            assert provenance['source_sha256'] == source_hash
            assert provenance['device'] == manifest['device'] and provenance['baseline'] == manifest['baseline']
            if str(k) in checkpoint_hashes:
                assert provenance['checkpoint_sha256'] == checkpoint_hashes[str(k)]
            signed_mean += a.astype(np.float64) / 10
            signed_sum += a.astype(np.float64)
            mean_member_absolute += np.abs(a.astype(np.float64)) / 10
            predictions.append(px); baselines.append(pb)
            rows = completeness_rows(metadata, tag, indices, a, px, pb, member=k,
                                     device=manifest['device'], baseline=manifest['baseline'], nodes=manifest['steps'])
            per_member_rows.extend(rows)
            saved_diff = np.abs(px - inputs['expected'][indices, k])
            member_reports.append({'member': k, 'sha256': digest(file), 'provenance': provenance,
                                   'completeness_recomputed_from_saved_float32_ig': completeness_summary(rows),
                                   'prediction_absolute_difference_from_locked_member': summarize(saved_diff)})
    pred = np.mean(predictions, axis=0)
    base = np.mean(baselines, axis=0)
    with np.load(path / 'ensemble.npz', allow_pickle=False) as z:
        a = z['ig']; px = z['prediction']; pb = z['baseline_prediction']
        assert a.shape == shape and np.array_equal(z['indices'], indices)
        assert np.isfinite(a).all()
        exact_mean = np.array_equal(a, signed_mean.astype(np.float32))
        assert exact_mean, 'Saved ensemble differs from signed member arithmetic mean'
        assert np.array_equal(px, pred) and np.array_equal(pb, base)
        expected_residual_before_float32 = signed_mean.sum(axis=(1, 2)) - (pred - base)
        residual_delta = np.abs(z['residual'] - expected_residual_before_float32)
        assert np.all(residual_delta <= 1e-12)
        assert np.array_equal(z['tolerance'], ATOL + RTOL * np.abs(pred - base))
        residual_saved = z['residual'].copy()
        a = a.copy()
    rows = completeness_rows(metadata, tag, indices, a, pred, base, member='ensemble',
                             device=manifest['device'], baseline=manifest['baseline'], nodes=manifest['steps'])
    shares = lead_shares(a)
    nonzero = np.isfinite(shares).all(axis=1)
    if nonzero.any():
        assert np.max(np.abs(shares[nonzero].sum(axis=1) - 1)) < 1e-12
    report = {'status': 'INTEGRITY_PASS', 'manifest': manifest, 'n': len(indices),
              'ensemble_sha256': digest(path / 'ensemble.npz'),
              'exact_signed_member_mean_after_float32_cast': exact_mean,
              'exact_member_prediction_and_baseline_mean': True,
              'all_indices_and_input_source_checkpoint_provenance_match': True,
              'maximum_float32_difference_from_sum_then_divide_mean': float(np.abs(a - (signed_sum / 10).astype(np.float32)).max()),
              'maximum_stored_residual_difference_from_float64_member_mean': float(residual_delta.max()),
              'maximum_residual_change_due_to_float32_ensemble_storage': float(np.max(np.abs(np.array([r['completeness_residual'] for r in rows]) - residual_saved))),
              'max_mean_absolute_member_minus_absolute_mean_member_ig': float((mean_member_absolute - np.abs(signed_mean)).max()),
              'zero_total_absolute_attribution_cases': int((~nonzero).sum()),
              'completeness_recomputed_from_saved_float32_ensemble': completeness_summary(rows),
              'members': member_reports}
    result = {'tag': tag, 'ig': a, 'prediction': pred, 'baseline_prediction': base, 'indices': indices,
              'manifest': manifest, 'nodes_by_case': np.repeat(manifest['steps'], len(indices)),
              'source_run_by_case': [tag] * len(indices)}
    return result, report, rows, per_member_rows


def compare_results(name, reference, alternative, metadata, leads, restriction=None):
    ref_map = {int(i): j for j, i in enumerate(reference['indices'])}
    alt_map = {int(i): j for j, i in enumerate(alternative['indices'])}
    common = sorted(set(ref_map) & set(alt_map))
    if restriction is not None:
        common = [i for i in common if i in set(restriction)]
    rows, lead_rows = [], []
    for idx in common:
        j, k = ref_map[idx], alt_map[idx]
        a = reference['ig'][j].astype(np.float64)
        b = alternative['ig'][k].astype(np.float64)
        sa, sb = lead_shares(a[None])[0], lead_shares(b[None])[0]
        l1a, l1b = np.abs(a).sum(), np.abs(b).sum()
        signed_l1 = np.abs(b - a).sum()
        cosine = safe_ratio(float(np.sum(a * b)), float(np.linalg.norm(a) * np.linalg.norm(b)))
        r = {**case_identity(metadata, idx), 'comparison': name, 'reference': reference['tag'], 'alternative': alternative['tag'],
             'reference_nodes': int(reference['nodes_by_case'][j]), 'alternative_nodes': int(alternative['nodes_by_case'][k]),
             'signed_relative_L1_to_reference': safe_ratio(signed_l1, l1a),
             'signed_symmetric_relative_L1': safe_ratio(2 * signed_l1, l1a + l1b),
             'signed_cosine_similarity': cosine,
             'lead_share_spearman': spearman(sa, sb) if np.isfinite(sa).all() and np.isfinite(sb).all() else None,
             'lead_share_max_absolute_difference_percentage_points': float(np.max(np.abs(sb - sa)) * 100),
             'lead_share_mean_absolute_difference_percentage_points': float(np.mean(np.abs(sb - sa)) * 100),
             'lead_share_total_variation': float(.5 * np.sum(np.abs(sb - sa))),
             'prediction_absolute_difference': float(abs(alternative['prediction'][k] - reference['prediction'][j])),
             'baseline_prediction_absolute_difference': float(abs(alternative['baseline_prediction'][k] - reference['baseline_prediction'][j]))}
        rows.append(r)
        for l, lead in enumerate(leads):
            lead_rows.append({**case_identity(metadata, idx), 'comparison': name, 'lead': lead,
                              'reference_share_percent': float(100 * sa[l]), 'alternative_share_percent': float(100 * sb[l]),
                              'difference_percentage_points': float(100 * (sb[l] - sa[l]))})
    quantitative = {key: summarize([r[key] for r in rows]) for key in [
        'signed_relative_L1_to_reference', 'signed_symmetric_relative_L1', 'signed_cosine_similarity',
        'lead_share_spearman', 'lead_share_max_absolute_difference_percentage_points',
        'lead_share_mean_absolute_difference_percentage_points', 'lead_share_total_variation',
        'prediction_absolute_difference', 'baseline_prediction_absolute_difference']}
    groups = []
    for status in [0, 1]:
        for risk in RISK_ORDER:
            group_idx = [i for i in common if int(metadata.iloc[i].death_72h) == status and metadata.iloc[i].risk_stratum == risk]
            if not group_idx:
                continue
            med_a = np.median(lead_shares(reference['ig'][[ref_map[i] for i in group_idx]]), axis=0)
            med_b = np.median(lead_shares(alternative['ig'][[alt_map[i] for i in group_idx]]), axis=0)
            groups.append({'death_72h': status, 'risk_stratum': risk, 'n': len(group_idx),
                           'spearman_of_cell_median_lead_shares': spearman(med_a, med_b),
                           'maximum_median_lead_share_difference_percentage_points': float(np.max(np.abs(med_b - med_a)) * 100)})
    report = {'reference': reference['tag'], 'alternative': alternative['tag'], 'paired_n': len(common),
              'fixed_sensitivity_cases_only': restriction is not None,
              'reference_device': reference['manifest'].get('device'), 'alternative_device': alternative['manifest'].get('device'),
              'reference_baseline': reference['manifest'].get('baseline'), 'alternative_baseline': alternative['manifest'].get('baseline'),
              'quantitative': quantitative, 'cell_median_comparisons': groups,
              'interpretation': 'Descriptive paired differences; no stability threshold or rank-correlation cutoff is asserted.'}
    return report, rows, lead_rows


def assemble_primary(args, runs, metadata):
    if args.primary_manifest:
        spec = json.loads(args.primary_manifest.read_text())
        choices = spec['cases']
        source_map = {int(r['sample_row_0based']): r.get('source_run', r.get('run')) for r in choices}
        assert len(source_map) == len(choices) == len(metadata)
        assert set(source_map) == set(range(len(metadata)))
        provenance = {'mode': 'explicit_case_map', 'manifest': str(args.primary_manifest), 'sha256': digest(args.primary_manifest)}
    else:
        if args.primary_run not in runs:
            return None, {'status': 'PENDING', 'reason': f'Primary run {args.primary_run} is not complete'}
        source_map = {int(i): args.primary_run for i in runs[args.primary_run]['indices']}
        for tag in args.refinement_run:
            if tag not in runs:
                return None, {'status': 'PENDING', 'reason': f'Refinement run {tag} is not complete'}
            source_map.update({int(i): tag for i in runs[tag]['indices']})
        assert set(source_map) == set(range(len(metadata))), 'Primary assembly must retain every sampled case'
        provenance = {'mode': 'base_plus_ordered_overlays', 'base': args.primary_run, 'overlays': args.refinement_run}
    a = np.empty((len(metadata), 12, 1000), dtype=np.float32)
    px, pb = np.empty(len(metadata)), np.empty(len(metadata))
    nodes, sources, devices = [], [], set()
    for i in range(len(metadata)):
        tag = source_map[i]
        if tag not in runs:
            return None, {'status': 'PENDING', 'reason': f'Chosen source run {tag} is not complete'}
        run = runs[tag]
        assert run['manifest']['baseline'] == 'zero', 'Primary figure must use zero baseline'
        matches = np.flatnonzero(run['indices'] == i)
        assert len(matches) == 1
        j = int(matches[0])
        a[i], px[i], pb[i] = run['ig'][j], run['prediction'][j], run['baseline_prediction'][j]
        nodes.append(int(run['nodes_by_case'][j])); sources.append(tag); devices.add(run['manifest']['device'])
    final_check = None
    if args.final_arrays:
        if not args.final_arrays.exists():
            return None, {'status': 'PENDING', 'reason': 'Final arrays do not exist yet'}
        with np.load(args.final_arrays, allow_pickle=False) as z:
            key = 'ig_zero' if 'ig_zero' in z.files else 'ig'
            assert np.array_equal(z[key], a), 'Final public attribution arrays differ from declared case sources'
            if 'indices' in z.files:
                assert np.array_equal(z['indices'], np.arange(len(metadata)))
        final_check = {'path': str(args.final_arrays), 'sha256': digest(args.final_arrays), 'exact_declared_source_match': True}
    result = {'tag': 'assembled_primary', 'ig': a, 'prediction': px, 'baseline_prediction': pb,
              'indices': np.arange(len(metadata)), 'manifest': {'device': '/'.join(sorted(devices)), 'baseline': 'zero'},
              'nodes_by_case': np.array(nodes), 'source_run_by_case': sources}
    return result, {'status': 'INTEGRITY_PASS', 'provenance': provenance, 'final_array_check': final_check,
                    'source_run_counts': {s: sources.count(s) for s in sorted(set(sources))},
                    'integration_node_counts': {str(n): nodes.count(n) for n in sorted(set(nodes))},
                    'all_191_cases_retained': len(metadata) == 191}


def auto_pairs(runs, sensitivity):
    def choose(baseline, device=None, steps=None, require_sensitivity=False):
        options = [r for r in runs.values() if r['manifest'].get('baseline') == baseline
                   and (device is None or r['manifest'].get('device') == device)
                   and (steps is None or r['manifest'].get('steps') == steps)]
        if require_sensitivity:
            options = [r for r in options if set(sensitivity).issubset(set(r['indices']))]
        return sorted(options, key=lambda r: (-len(r['indices']), r['tag']))[0] if options else None
    pairs = []
    for device in ['mps', 'cpu']:
        a, b = choose('zero', device, 128, True), choose('zero', device, 256, True)
        if a and b:
            pairs.append((f'nodes128_256_{device}', a['tag'], b['tag'], sensitivity))
        for steps in [256, 128, 512, 1024, 2048]:
            a, b = choose('zero', device, steps, True), choose('lead_mean', device, steps, True)
            if a and b:
                pairs.append((f'baseline_zero_leadmean_{device}_{steps}', a['tag'], b['tag'], sensitivity))
                break
    for steps in [128, 256, 512, 1024, 2048]:
        a, b = choose('zero', 'cpu', steps), choose('zero', 'mps', steps)
        if a and b:
            pairs.append((f'backend_cpu_mps_zero_{steps}', a['tag'], b['tag'], None))
    return pairs


def write_public_csv(path, rows):
    if not rows:
        # These are audit-owned artifacts. Do not leave a previous completed
        # table appearing current when this invocation has no corresponding data.
        if path.exists():
            path.unlink()
        return
    frame = pd.DataFrame(rows)
    forbidden = {'file_name', 'subject_id', 'study_id', 'npz_ecg_row_0based', 'sample_row_0based', 'ecg_time', 'edregtime'}
    assert not set(frame.columns) & forbidden
    frame.to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--primary-run', default='zero_128')
    p.add_argument('--refinement-run', action='append', default=[])
    p.add_argument('--primary-manifest', type=Path)
    p.add_argument('--final-arrays', type=Path)
    p.add_argument('--compare', action='append', default=[], help='name=reference_tag:alternative_tag; full index intersection')
    p.add_argument('--strict', action='store_true', help='Exit nonzero for an integrity failure; pending computations are not failures')
    args = p.parse_args()
    base = args.base.resolve()
    out = args.output_dir or base / 'audit'
    out.mkdir(parents=True, exist_ok=True)
    public = out / 'public_source'; public.mkdir(exist_ok=True)
    for attr in ['primary_manifest', 'final_arrays']:
        value = getattr(args, attr)
        if value is not None and not value.is_absolute():
            setattr(args, attr, base / value)
    inputs_path = base / 'inputs.npz'
    input_hash = digest(inputs_path)
    with np.load(inputs_path, allow_pickle=False) as z:
        inputs = {key: z[key] for key in z.files}
    metadata = pd.read_csv(base / 'sample_metadata.csv')
    im = json.loads((base / 'input_manifest.json').read_text())
    leads = json.loads((base / 'lead_order_units.json').read_text())['lead_order']
    assert len(leads) == len(set(leads)) == 12
    assert np.array_equal(metadata.sample_row_0based, np.arange(len(metadata)))
    assert len(metadata) == 191 and inputs['ecg'].shape == (191, 12, 1000)
    assert inputs['tab'].shape == (191, 12) and np.array_equal(inputs['y'], metadata.death_72h)
    assert digest(base / 'sample_metadata.csv') == im['sample_metadata_sha256']
    assert array_digest(inputs['ecg']) == im['ecg_sha256']
    assert array_digest(inputs['tab']) == im['tab_sha256']
    assert array_digest(inputs['y']) == im['labels_sha256']
    assert np.array_equal(inputs['expected'], metadata[[f'proba_run{i}' for i in range(10)]].to_numpy())
    assert np.array_equal(inputs['expected_mean'], metadata.proba_mean.to_numpy())
    checkpoint_hashes = {}
    prediction_report = base / 'prediction_verification.json'
    if prediction_report.exists():
        pr = json.loads(prediction_report.read_text())
        for device in pr.get('devices', {}).values():
            for member in device.get('members', []):
                k = str(member['run'])
                assert k not in checkpoint_hashes or checkpoint_hashes[k] == member['sha256']
                checkpoint_hashes[k] = member['sha256']
    runs, reports, run_rows, member_rows, issues = {}, {}, [], [], []
    for directory in sorted((base / 'runs').glob('*')):
        if not directory.is_dir():
            continue
        try:
            run, report, rows, member = audit_run(directory, inputs, metadata, input_hash, im['source_sha256'], checkpoint_hashes)
            reports[directory.name] = report
            if run is not None:
                runs[directory.name] = run; run_rows.extend(rows); member_rows.extend(member)
        except (json.JSONDecodeError, EOFError, zipfile.BadZipFile) as exc:
            reports[directory.name] = {'status': 'PENDING', 'reason': 'Files may still be being written', 'error': str(exc)}
        except Exception as exc:
            reports[directory.name] = {'status': 'INTEGRITY_FAILURE', 'error': f'{type(exc).__name__}: {exc}'}
            issues.append(f'{directory.name}: {type(exc).__name__}: {exc}')
    primary = None
    try:
        primary, primary_report = assemble_primary(args, runs, metadata)
    except Exception as exc:
        primary_report = {'status': 'INTEGRITY_FAILURE', 'error': f'{type(exc).__name__}: {exc}'}
        issues.append(f'primary: {type(exc).__name__}: {exc}')
    primary_rows, lead_rows, cell_rows = [], [], []
    if primary is not None:
        if args.final_arrays:
            with np.load(args.final_arrays, allow_pickle=False) as z:
                if 'ecg' in z.files:
                    assert np.array_equal(z['ecg'], inputs['ecg']), 'Final ECG display source differs from verified inputs'
        primary_rows = completeness_rows(metadata, primary['tag'], primary['indices'], primary['ig'], primary['prediction'], primary['baseline_prediction'])
        shares = lead_shares(primary['ig'])
        assert np.isfinite(shares).all()
        assert np.max(np.abs(shares.sum(axis=1) - 1)) < 1e-12
        for i, row in enumerate(primary_rows):
            row.update({'source_run': primary['source_run_by_case'][i], 'nodes': int(primary['nodes_by_case'][i])})
            for l, lead in enumerate(leads):
                lead_rows.append({**case_identity(metadata, i), 'lead': lead, 'lead_share_percent': float(shares[i, l] * 100)})
        for status in [0, 1]:
            for risk in RISK_ORDER:
                mask = (metadata.death_72h == status) & (metadata.risk_stratum == risk)
                for l, lead in enumerate(leads):
                    x = shares[mask.to_numpy(), l] * 100
                    q25, median, q75 = np.quantile(x, [.25, .5, .75])
                    cell_rows.append({'death_72h': status, 'risk_stratum': risk, 'encounters': int(mask.sum()), 'lead': lead,
                                      'mean_share_percent': float(x.mean()), 'median_share_percent': float(median),
                                      'q25_share_percent': float(q25), 'q75_share_percent': float(q75)})
        primary_report['completeness_recomputed'] = completeness_summary(primary_rows)
        primary_report['max_lead_share_sum_error'] = float(np.max(np.abs(shares.sum(axis=1) - 1)))
        runs[primary['tag']] = primary
    pairs = auto_pairs({k: v for k, v in runs.items() if k != 'assembled_primary'}, im['sensitivity_indices'])
    for text in args.compare:
        name, pair = text.split('=', 1); ref, alt = pair.split(':', 1)
        pairs.append((name, ref, alt, None))
    comparisons, comparison_rows, comparison_leads = {}, [], []
    for name, ref, alt, restriction in pairs:
        if ref not in runs or alt not in runs:
            comparisons[name] = {'status': 'PENDING', 'reference': ref, 'alternative': alt}; continue
        report, rows, leads_out = compare_results(name, runs[ref], runs[alt], metadata, leads, restriction)
        comparisons[name] = report; comparison_rows.extend(rows); comparison_leads.extend(leads_out)
    coverage = {
        'fixed_12_node_sensitivity_available': any(k.startswith('nodes128_256') and r.get('paired_n') == 12 for k, r in comparisons.items()),
        'fixed_12_baseline_sensitivity_available': any(k.startswith('baseline_zero_leadmean') and r.get('paired_n') == 12 for k, r in comparisons.items()),
        'cpu_mps_attribution_comparison_available': any(k.startswith('backend_cpu_mps') and r.get('paired_n', 0) > 0 for k, r in comparisons.items()),
    }
    display_rows = []
    if primary is not None and (base / 'example_selection.json').exists():
        selection = json.loads((base / 'example_selection.json').read_text())
        assert selection['metadata_sha256'] == digest(base / 'sample_metadata.csv')
        for chosen in selection['selected_examples']:
            i = chosen['sample_row_0based']
            r = primary_rows[i]
            display_rows.append({'figure_case': chosen['case'], **case_identity(metadata, i), 'display_lead': 'II',
                                 'saved_raw_score': float(metadata.iloc[i].proba_mean), 'nodes': r['nodes'],
                                 'completeness_residual': r['completeness_residual'], 'candidate_tolerance': r['candidate_tolerance'],
                                 'within_candidate_tolerance': r['within_candidate_tolerance']})
    for filename, rows in [('run_completeness_by_case.csv', run_rows), ('member_completeness_by_case.csv', member_rows),
                           ('primary_completeness_by_case.csv', primary_rows), ('primary_lead_shares_by_case.csv', lead_rows),
                           ('primary_cell_lead_summary.csv', cell_rows), ('paired_sensitivity_by_case.csv', comparison_rows),
                           ('paired_sensitivity_lead_shares.csv', comparison_leads), ('display_cases.csv', display_rows)]:
        write_public_csv(public / filename, rows)
    report = {'status': 'INTEGRITY_FAILURE' if issues else ('PENDING' if primary is None or not all(coverage.values()) else 'INTEGRITY_PASS'),
              'scope': 'Independent arithmetic and data-alignment audit of saved results; no inference or attribution computation.',
              'candidate_completeness_tolerance': {'absolute': ATOL, 'relative_to_absolute_output_difference': RTOL,
                                                  'source': 'ANALYSIS_PLAN.md; no sensitivity stability cutoff is introduced'},
              'inputs': {'inputs_npz_sha256': input_hash, 'metadata_sha256': digest(base / 'sample_metadata.csv'),
                         'input_manifest_sha256': digest(base / 'input_manifest.json'), 'verified_shape_and_label_alignment': True},
              'runs': reports, 'primary': primary_report, 'comparisons': comparisons, 'sensitivity_coverage': coverage,
              'integrity_issues': issues, 'public_tables': {p.name: digest(p) for p in public.glob('*.csv')},
              'interpretation_limits': ['Completeness is necessary numerical accounting, not proof of attribution stability.',
                                        'Signed relative L1 uses the reference total absolute attribution as denominator.',
                                        'Lead-share rank correlations describe 12 correlated lead channels and carry no biological or causal claim.',
                                        'The 191-case sample is deliberately balanced; summaries are by outcome-by-risk cell.'],
              'script_sha256': digest(Path(__file__))}
    (out / 'numeric_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    lines = ['# Independent IG numerical audit', '', f"Status: {report['status']}", '',
             f"Completed run audits: {len(reports) - sum(r.get('status') == 'PENDING' for r in reports.values())}; primary: {primary_report['status']}.", '',
             'Completeness uses the planned candidate tolerance 0.0005 + 0.02 × |F(x,c) − F(b,c)|. Values below are recomputed from the saved float32 arrays.']
    if primary is not None:
        cr = primary_report['completeness_recomputed']
        lines.extend(['', f"Primary: {cr['n']} cases; {cr['outside_candidate_tolerance']} outside tolerance; maximum absolute residual {cr['absolute_residual']['max']:.8g}."])
    for name, comp in comparisons.items():
        lines.extend(['', f'## {name}', ''])
        if comp.get('status') == 'PENDING':
            lines.append('Pending saved comparison outputs.'); continue
        lines.append(f"Paired cases: {comp['paired_n']}. Reference: {comp['reference']}; alternative: {comp['alternative']}.")
        for key in ['signed_relative_L1_to_reference', 'lead_share_spearman', 'lead_share_max_absolute_difference_percentage_points']:
            s = comp['quantitative'][key]
            lines.append(f"- {key}: median {s['median']}; range {s['min']} to {s['max']}.")
        lines.append('These are quantitative comparisons without an asserted stability threshold.')
    if issues:
        lines.extend(['', '## Integrity issues', ''] + [f'- {x}' for x in issues])
    (out / 'numeric_audit.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': report['status'], 'completed_runs': list(runs), 'primary': primary_report,
                      'sensitivity_coverage': coverage, 'integrity_issues': issues, 'output_dir': str(out)}, indent=2))
    if args.strict and issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
