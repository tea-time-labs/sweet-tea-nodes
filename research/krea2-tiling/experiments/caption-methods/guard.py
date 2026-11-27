"""Validate a localized sentence independently against its exact padded source and the unmarked overview."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'test-harness'))
import run_matrix as matrix
from caption_methods import crop,generate

PROMPT=(
    "A caption describes a smaller region of this image. Check whether the pixels support every named type "
    "of person, animal, body part and object, and the claimed physical contact. "
    "If even one element is absent, answer UNSUPPORTED. If detail is too small to check, answer UNCLEAR. "
    "Answer SUPPORTED only if you can locate visual evidence for every element. Output one word. "
    "Proposed sentence: "
)
def validate(clip,source,tile,sentence):
    from krea16_test_node import to_tensor,to_pil
    from PIL import Image
    overview=to_pil(source);overview.thumbnail((640,640),Image.Resampling.LANCZOS)
    views=(crop(source,tile['padded_source_xyxy']),to_tensor(overview))
    raw=[generate(clip,PROMPT+sentence,im,12) for im in views]
    return raw,all(x.strip(' .!\n').upper()=='SUPPORTED' for x in raw)
if __name__=='__main__':
    _,clip,_=matrix.load_runtime()
    edge=json.loads((HERE/'edge-parts-results.json').read_text())[0]
    cases=[(edge['case'],edge['tile'],edge['center'],'generated center'),
           ('10-wildlife-family',8,'Dark fox legs touch the leaf-covered ground.','visible control'),
           ('01-florist-handoff',5,'Two hands hold the same paper-wrapped bouquet.','visible control')]
    rows=[]
    for case,i,sentence,kind in cases:
        folder=HERE.parent/'results'/case
        report=json.loads((folder/'report.json').read_text())
        with matrix.Image.open(folder/'source.png') as im:src=matrix.to_tensor(im)
        decisions,accepted=validate(clip,src,report['geometry']['tiles'][i-1],sentence)
        rows.append(dict(case=case,tile=i,sentence=sentence,kind=kind,answers=decisions,accepted=accepted))
    matrix.save_json(HERE/'guard-results.json',rows)
    if not rows[0]['accepted']:
        for i in (7,8):
            original=HERE/'outputs/edge_verified'/edge['case']/f'{i}.json'
            record=json.loads(original.read_text());record['method']='guarded_edges'
            if i==8:
                before=record['caption']
                record['caption']=before.removesuffix(' Between them: '+edge['center'])
                assert before!=record['caption']
            dest=HERE/'outputs/guarded_edges'/edge['case']/original.name
            dest.parent.mkdir(parents=True,exist_ok=True);matrix.save_json(dest,record)
    print(rows,flush=True)
