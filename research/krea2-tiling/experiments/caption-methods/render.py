"""Run matched source-image interventions using actual generated captions from the benchmark."""
import argparse,hashlib,json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'test-harness'))
import run_matrix as matrix

AFFECTED={"01-florist-handoff":[7],"10-wildlife-family":[7,8]}

def render(method, cases, run_original):
    model,clip,vae=matrix.load_runtime()
    node=matrix.NODE_CLASS_MAPPINGS["STKrea2TurboMatrixTest"]()
    for case in cases:
        folder=ROOT/'results'/case
        report=json.loads((folder/'report.json').read_text())
        with matrix.Image.open(folder/'source.png') as image:
            source=matrix.to_tensor(image)
        for mode in (('archived',method) if run_original and method!='archived' else (method,)):
            tiles=report['geometry']['tiles']
            captions=[x['caption'] for x in tiles]
            out=HERE/'renders'/case/mode
            out.mkdir(parents=True,exist_ok=True)
            source_hash=hashlib.sha256((folder/'source.png').read_bytes()).hexdigest()
            if mode!='archived':
                for i in AFFECTED[case]:
                    record=json.loads((HERE/'outputs'/mode/case/f'{i}.json').read_text())
                    assert record['source_sha256']==source_hash and record['tile']==i
                    captions[i-1]=record['caption']
            def prepared(details):
                assert json.loads(json.dumps(details['tiles']))==[{**t,'caption':cap} for t,cap in zip(tiles,captions,strict=True)]
            def saved(name,img,metrics):
                matrix.save_png(img,out/'output.png')
                matrix.save_json(out/'report.json',dict(case=case,method=mode,variant=name,metrics=metrics,seed=report['upscale_seed'],
                    changed_tiles=[] if mode=='archived' else AFFECTED[case],captions=captions,
                    gpu=matrix.torch.cuda.get_device_name(0),source_sha256=source_hash,
                    output_sha256=hashlib.sha256((out/'output.png').read_bytes()).hexdigest()))
                print('RENDER SAVED',case,mode,flush=True)
            node.upscale(source,model,clip,vae,report['scene']['prompt'],report['upscale_seed'],variant='euler-simple',
                         reference_captions=captions,on_prepared=prepared,on_render=saved)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('method',choices=('archived','core','context_check','split_check','footwear_verified','edge_verified','guarded_edges'))
    p.add_argument('cases',nargs='+',choices=AFFECTED)
    p.add_argument('--run-original',action='store_true')
    a=p.parse_args()
    render(a.method,a.cases,a.run_original)
