"""Check full reruns, numerical outputs, source preservation, and local exports."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analysis import PACKAGE,write_json,sha


def main():
    comparisons=[]
    for a,b in [('batch1_verified','batch1'),('batch2','batch2_clean_recheck')]:
        for path in sorted((PACKAGE/a).rglob('*.csv')):
            relative=path.relative_to(PACKAGE/a);other=PACKAGE/b/relative
            assert other.exists(),other
            same=sha(path)==sha(other)
            if not same:
                pd.testing.assert_frame_equal(pd.read_csv(path),pd.read_csv(other),rtol=1e-12,atol=1e-12)
            comparisons.append({'pair':[a,b],'file':str(relative),'byte_identical':same,'numeric_match':True})
        for path in sorted((PACKAGE/a).rglob('evaluation_scores.npz')):
            other=PACKAGE/b/path.relative_to(PACKAGE/a)
            x,y=np.load(path),np.load(other)
            assert x.files==y.files
            for k in x:
                assert np.array_equal(x[k],y[k]),(a,path.name,k)
    for folder in ['batch1_verified','batch2']:
        manifest=json.loads((PACKAGE/folder/'source_manifest.json').read_text())
        for p,s in manifest.items():
            assert sha(p)==s,p
        cal=pd.read_csv(PACKAGE/folder/'calibration_metrics.csv')
        ci_rows=cal[cal.ci_method!='point_estimate_only']
        assert np.isfinite(ci_rows[['estimate','ci_lower','ci_upper']]).all().all()
        assert (ci_rows.ci_lower<=ci_rows.ci_upper).all()
        for path in (PACKAGE/folder).rglob('threshold_metrics.csv'):
            t=pd.read_csv(path)
            assert (t.tp+t.fp+t.tn+t.fn==t.n).all()
            assert (t.tp+t.fn==t.events).all()
            assert np.allclose(t.deaths_per_100_alerts,100*t.ppv)
            assert np.allclose(t.alerts_per_1000,(t.tp+t.fp)/t.n*1000)
            assert np.allclose(t.false_negative_rate,1-t.sensitivity)
    independent=json.loads((PACKAGE/'INDEPENDENT_R_CHECK.json').read_text())
    assert independent['status']=='PASS'
    exports=[]
    for path in (PACKAGE/'figures').glob('*.svg'):
        assert '<text' in path.read_text(),path
        pdf=path.with_suffix('.pdf');png=path.with_suffix('.png')
        assert pdf.exists() and pdf.stat().st_size>1000 and png.exists()
        exports.append({'stem':path.stem,'svg_contains_editable_text':True,'pdf_png_present':True})
    write_json(PACKAGE/'FINAL_QA.json',{'status':'PASS','full_reruns':comparisons,
        'all_patient_level_numeric_predictions_exactly_reproduced':True,
        'source_files_unchanged':True,'all_required_confidence_intervals_finite':True,
        'all_confusion_count_and_rate_identities_pass':True,'independent_R_checks':independent,
        'figure_exports':exports,'scope':'Requested supplemental analyses only; not a full study provenance or clinical validation lock.'})
    print('FINAL QA PASS:',len(comparisons),'reproduced CSV tables; independent R points and cluster CIs agree.')


if __name__=='__main__':
    main()
