"""Retain two verified edge observations without requiring an inference about subject identity."""
from pathlib import Path
import json,sys,re
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'test-harness'))
import run_matrix as matrix
from caption_methods import crop,generate
from edge_probe import WORDS

def parts(clip,core,labels):
    width=core.shape[2]
    k=max(1,width*35//100)
    if any(x not in WORDS for x in labels) or width-2*k<=0:return None
    middle=core[:,:,k:width-k]
    description=generate(clip,"Describe the two most obvious physical features in this photograph fragment, "
        "including what touches the surface below. Use one short English sentence. "
        "Keep cropped animal and human parts literal.",middle,55)
    lf,rt=(x.lower() for x in labels)
    la='An' if lf=='eye' else 'A'
    ra='an' if rt=='eye' else 'a'
    return f"{la} {lf} is visible at the left and {ra} {rt} is visible at the right. Between them: {description}", description

if __name__=='__main__':
    _,clip,_=matrix.load_runtime()
    rows=json.loads((HERE/'edge-results.json').read_text())
    out=[]
    for r in rows:
        if not all(label in WORDS for label in r['labels']):continue
        case=r['case'];i=r['tile']
        folder=HERE.parent/'results'/case
        report=json.loads((folder/'report.json').read_text())
        with matrix.Image.open(folder/'source.png') as im: src=matrix.to_tensor(im)
        tile=report['geometry']['tiles'][i-1]
        cap,middle=parts(clip,crop(src,tile['core_source_xyxy']),r['labels'])
        out.append({**r,'caption':cap,'center':middle,'identity_asserted':False})
    matrix.save_json(HERE/'edge-parts-results.json',out)
    print("BORDER CAPTIONS",len(out),flush=True)
