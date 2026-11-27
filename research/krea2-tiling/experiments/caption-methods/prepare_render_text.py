"""Assemble controlled cue interventions by changing only the relevant text in the selected method."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def write(method,case,tile,base,caption):
    r=json.loads((HERE/'outputs'/'split_check'/case/f'{tile}.json').read_text())
    out=HERE/'outputs'/method/case
    out.mkdir(parents=True,exist_ok=True)
    p=out/f'{tile}.json'
    p.write_text(json.dumps({**r,'method':method,'base_caption':base,'caption':caption},indent=2)+'\n')
def prepare():
    shoe_file=HERE/'footwear-results.json'
    if shoe_file.exists():
        for r in json.loads(shoe_file.read_text()):
            if r['case']=='01-florist-handoff' and r['tile']==7 and r['status']=='accepted':
                base=json.loads((HERE/'outputs/split_check'/r['case']/'7.json').read_text())['caption']
                assert r['caption'].startswith(r['base'])
                write('footwear_verified',r['case'],7,base,base+r['caption'][len(r['base']):])
    edge_file=HERE/'edge-parts-results.json'
    if edge_file.exists():
        for r in json.loads(edge_file.read_text()):
            if r['case']=='10-wildlife-family' and r['tile']==8 and r['caption']:
                for i in (7,8):
                    base=json.loads((HERE/'outputs/split_check'/r['case']/f'{i}.json').read_text())['caption']
                    write('edge_verified',r['case'],i,base,r['caption'] if i==8 else base)
if __name__=='__main__': prepare()
