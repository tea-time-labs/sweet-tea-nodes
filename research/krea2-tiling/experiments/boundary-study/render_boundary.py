"""Run the original ST sampler with the frozen one-image, full-border fox captions."""
import hashlib, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'test-harness'))
import run_matrix as matrix

def render():
    model,clip,vae=matrix.load_runtime()
    case='10-wildlife-family'
    folder=ROOT/'results'/case
    report=json.loads((folder/'report.json').read_text())
    with matrix.Image.open(folder/'source.png') as image: source=matrix.to_tensor(image)
    source_hash=hashlib.sha256((folder/'source.png').read_bytes()).hexdigest()
    tiles=report['geometry']['tiles']
    gpu=matrix.torch.cuda.get_device_name(0)
    modes=['two_target_tiles','all_nine']
    if gpu!='NVIDIA L4':modes.insert(0,'device_control')
    for mode in modes:
        out=HERE/'renders'/mode
        out.mkdir(parents=True,exist_ok=True)
        caption_list=[t['caption'] for t in tiles]
        changed={'two_target_tiles':[7,8],'all_nine':list(range(1,10)),'device_control':[]}[mode]
        for i in changed:
            r=json.loads((HERE/'outputs'/case/f'{i}-full_box.json').read_text())
            assert r['source_sha256']==source_hash and r['core']==tiles[i-1]['core_source_xyxy'] and r['mode']=='full_box'
            caption_list[i-1]=r['caption']
        def prepared(details):
            expected=[{**t,'caption':c} for t,c in zip(tiles,caption_list,strict=True)]
            assert json.loads(json.dumps(details['tiles']))==expected
        def saved(name,img,metrics):
            path=out/'output.png'
            matrix.save_png(img,path)
            matrix.save_json(out/'report.json',dict(mode=mode,case=case,captions=caption_list,
                changed_tiles=changed,seed=report['upscale_seed'],variant=name,gpu=gpu,torch=str(matrix.torch.__version__),
                source_sha256=source_hash,output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metrics=metrics))
            print('RENDER SAVED',mode,flush=True)
        matrix.NODE_CLASS_MAPPINGS['STKrea2TurboMatrixTest']().upscale(
            source,model,clip,vae,report['scene']['prompt'],report['upscale_seed'],variant='euler-simple',
            reference_captions=caption_list,on_prepared=prepared,on_render=saved)
    print('RENDERS COMPLETE',flush=True)
if __name__=='__main__':render()
