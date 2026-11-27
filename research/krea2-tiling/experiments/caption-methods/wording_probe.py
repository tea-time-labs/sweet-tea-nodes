"""Paired prompt ablation: the sole prompt change removes the reconstruction audience."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'test-harness'))
import json
import run_matrix as matrix
import caption_methods as cm
CASES={'05-pottery-class':[6], '12-urban-basketball':[6], '14-veterinary-exam':[4]}
if __name__=='__main__':
    _,clip,_=matrix.load_runtime()
    records=[]
    original=cm.CORE
    revised=original.replace(', for a painter reconstructing it','')
    assert original!=revised
    for case,ids in CASES.items():
        folder=HERE.parent/'results'/case
        report=json.loads((folder/'report.json').read_text())
        with matrix.Image.open(folder/'source.png') as im:src=matrix.to_tensor(im)
        for i in ids:
            box=report['geometry']['tiles'][i-1]
            for tag,text in [('original',original),('removed',revised)]:
                cm.CORE=text
                cap,_=cm.describe('core',clip,src,report['scene']['prompt'],box)
                records.append(dict(case=case,tile=i,wording=tag,caption=cap))
                matrix.save_json(HERE/'wording-results.json',records)
                print(case,i,tag,flush=True)
