"""Compare two independently generated descriptions against the clean core image."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'test-harness'))
import run_matrix as matrix
from caption_methods import crop,describe,generate

INSTRUCTION=(
    "Look carefully at this exact image. Two observers supplied tentative descriptions. "
    "They can be wrong, and can name objects outside your view. First inspect the boundary fragments, "
    "then the dominant shapes. Write one or two short sentences describing only the visible parts and their arrangement. "
    "Resolve conflicting details by looking at the pixels. If identity is uncertain use the visible shape or material. "
    "Use ordinary descriptive language. Omit occupations, unseen actions, production instructions and framing graphics. "
    "Where clear, give viewer-relative locations and the direction of a held or touching part. "
    "Proposal 1: {one}\nProposal 2: {two}\nReturn just the caption."
)
if __name__=='__main__':
    _,clip,_=matrix.load_runtime()
    from krea16_test_node import describe_cores
    selected=json.loads((HERE/'protocol.json').read_text())['fresh_tiles']
    for case,ids in selected.items():
        folder=ROOT/'results'/case
        report=json.loads((folder/'report.json').read_text())
        src=folder/'source.png';source_sha=hashlib.sha256(src.read_bytes()).hexdigest()
        with matrix.Image.open(src) as im:source=matrix.to_tensor(im)
        for i in ids:
            tile=report['geometry']['tiles'][i-1]; brief=report['scene']['prompt']
            caps={}
            for method in ('original_t4','split_check','duel'):
                out=HERE/'outputs'/method/case/f'{i}.json';out.parent.mkdir(exist_ok=True,parents=True)
                if out.exists():
                    r=json.loads(out.read_text())
                    assert r['source_sha256']==source_sha and r['core']==tile['core_source_xyxy']
                    caps[method]=r['caption'];continue
                start=time.monotonic()
                if method=='original_t4':
                    cap,=describe_cores(source,brief,clip,[tile]);notes=[]
                elif method=='split_check':
                    cap,notes=describe(method,clip,source,brief,tile)
                else:
                    order=['original_t4','split_check']
                    if int(case[:2])%2==0:order.reverse()
                    notes=[caps[m] for m in order]
                    cap=generate(clip,INSTRUCTION.format(one=notes[0],two=notes[1]),crop(source,tile['core_source_xyxy']),100)
                caps[method]=cap
                matrix.torch.cuda.synchronize()
                implementation=HERE/('duel.py' if method=='duel' else 'caption_methods.py')
                matrix.save_json(out,dict(method=method,case=case,tile=i,core=tile['core_source_xyxy'],padded=tile['padded_source_xyxy'],
                    caption=cap,notes=notes,seconds=round(time.monotonic()-start,2),gpu=matrix.torch.cuda.get_device_name(0),
                    source_sha256=source_sha,method_sha256=hashlib.sha256(implementation.read_bytes()).hexdigest()))
                print(case,i,method,flush=True)
    print('FRESH TILE RUN COMPLETE',flush=True)
