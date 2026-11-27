"""Test independent left/right facial evidence before asking whether opposite fragments are distinct."""
import json,re,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'test-harness'))
import run_matrix as matrix
from caption_methods import generate,crop,describe,CORE
TARGETS={'10-wildlife-family':[5,7,8], '05-pottery-class':[3],'14-veterinary-exam':[3],'01-florist-handoff':[2],'06-restaurant-pass':[1]}
PROBE=(
    "Inspect this strip. Does it contain a visible eye, nose, muzzle or mouth belonging to a person or animal? "
    "Hair, fur, ear, neck, leg, or an indistinct silhouette alone does not qualify. "
    "Answer with one word: EYE, NOSE, MUZZLE, MOUTH, NONE or UNCLEAR."
)
WORDS={'EYE','NOSE','MUZZLE','MOUTH'}
def token(s,allowed):
    word=s.strip().strip('.,!').upper()
    return word if word in allowed else 'UNCLEAR'
def inspect(clip,source,tile):
    core=crop(source,tile['core_source_xyxy']); w=core.shape[2]
    k=max(1,w*35//100)
    bands=(core[:,:,:k],core[:,:,-k:])
    answers=[generate(clip,PROBE,b,12) for b in bands]
    vals=[token(a,WORDS|{'NONE','UNCLEAR'}) for a in answers]
    verify=''
    if all(v in WORDS for v in vals):
        verify=generate(clip,"There is facial detail near both sides. Examine the whole fragment. "
            "Do those left and right facial features belong to two separate faces, or one continuous face? "
            "Answer with exactly TWO, ONE or UNCLEAR.",core,10)
    if token(verify,{'TWO','ONE'})=='TWO':
        fact='Fragments of two distinct faces enter from the left and right edges.'
        rest=generate(clip,"Two visible face fragments have been identified, one at each side. "
            "In a short sentence describe what visibly separates the two faces in the middle and the surface below. "
            "Name parts of bodies as parts.",core,58)
        caption=fact+' '+rest
    else:
        caption=''
    return dict(raw=answers,labels=vals,verify=verify,caption=caption)
if __name__=='__main__':
    _,clip,_=matrix.load_runtime()
    records=[]
    for case,ids in TARGETS.items():
        folder=HERE.parent/'results'/case
        report=json.loads((folder/'report.json').read_text())
        with matrix.Image.open(folder/'source.png') as im: source=matrix.to_tensor(im)
        for i in ids:
            r=inspect(clip,source,report['geometry']['tiles'][i-1])
            records.append(dict(case=case,tile=i,**r))
            matrix.save_json(HERE/'edge-results.json',records)
            print(case,i,r['labels'],r['verify'],flush=True)
