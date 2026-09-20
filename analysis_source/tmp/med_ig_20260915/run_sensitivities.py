"""Continue the already authorized numerical checks after the primary run closes."""
import time,json
import ig_analysis as q
import numpy as np

def complete(tag):
 try:return 'ensemble_completeness' in json.loads((q.BASE/'runs'/tag/'manifest.json').read_text())
 except (FileNotFoundError,json.JSONDecodeError):return False
while not complete('zero_128'):time.sleep(5)
q.emit('followup_started')
primary=np.load(q.BASE/'runs/zero_128/ensemble.npz')
failed=primary['indices'][np.abs(primary['residual'])>primary['tolerance']]
refinements=[]
for steps in [256,512,1024,2048]:
 if len(failed)==0:break
 tag=f'zero_refine_{steps}'
 q.integrate('mps',steps,'zero','indices:'+','.join(str(int(i)) for i in failed),tag,64)
 refinements.append(tag)
 a=np.load(q.BASE/'runs'/tag/'ensemble.npz');failed=a['indices'][np.abs(a['residual'])>a['tolerance']]
q.savejson('production_followup.json',{'primary_refinement_tags':refinements,'primary_unresolved_indices':failed.tolist(),'status':'sensitivity_running'})
assert len(failed)==0,'Primary completeness needs investigation beyond 2048 nodes'
q.integrate('mps',256,'zero','sensitivity','zero_256',64)
q.integrate('mps',128,'lead_mean','sensitivity','lead_mean_128',64)
a=np.load(q.BASE/'runs/lead_mean_128/ensemble.npz')
if np.any(np.abs(a['residual'])>a['tolerance']):
 q.integrate('mps',256,'lead_mean','sensitivity','lead_mean_256',64)
q.savejson('production_followup.json',{'primary_refinement_tags':refinements,'primary_unresolved_indices':failed.tolist(),'status':'calculations_complete_pending_independent_audit'})
q.emit('followup_finished')
