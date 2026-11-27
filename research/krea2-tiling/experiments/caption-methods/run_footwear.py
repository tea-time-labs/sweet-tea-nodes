"""Run the exploratory geometry check on preselected footwear and distractor crops."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'test-harness'))
import run_matrix as matrix
from caption_methods import describe
from footwear import check

TARGETS={'01-florist-handoff':[2,7], '07-tango-poster':[7], '09-ballet-rehearsal':[4,8],
         '08-night-market':[8], '10-wildlife-family':[7], '16-museum-diorama':[9]}
if __name__=='__main__':
    _,clip,_=matrix.load_runtime()
    out=HERE/'footwear-results.json'
    records=[]
    for case,tiles in TARGETS.items():
        path=ROOT/'results'/case
        report=json.loads((path/'report.json').read_text())
        with matrix.Image.open(path/'source.png') as im:source=matrix.to_tensor(im)
        for i in tiles:
            tile=report['geometry']['tiles'][i-1]
            base,notes=describe('core',clip,source,report['scene']['prompt'],tile)
            r=check(clip,source,tile,base)
            records.append(dict(case=case,tile=i,base=base,**r))
            matrix.save_json(out,records)
            print(case,i,r['status'],flush=True)
