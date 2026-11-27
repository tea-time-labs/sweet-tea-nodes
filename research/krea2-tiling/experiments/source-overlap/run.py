"""Matched Colab renders. Historical captions are injected before sampling."""
from dataclasses import replace
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack

ROOT=Path('/content/st-krea16')
TRIAL=ROOT/'trial'
NODE=ROOT/'sweet-tea-nodes'
cases=sys.argv[1:]
subprocess.run(['git','apply','--check',str(TRIAL/'integration.patch')],cwd=NODE,check=True)
subprocess.run(['git','apply',str(TRIAL/'integration.patch')],cwd=NODE,check=True)
shutil.copyfile(TRIAL/'krea2_overlap.py',NODE/'usdu_pose_tiled/krea2_overlap.py')
sys.path.insert(0,str(ROOT/'test-harness'))
import run_matrix as matrix
model,clip,vae=matrix.load_runtime()
import torch
from PIL import Image
from usdu_pose_tiled import krea2_vision as vision,krea2_overlap as overlap
from usdu_pose_tiled.krea2_control import STKrea2UltimateUpscale,square_tile_grid
import krea16_test_node as harness
import comfy.model_management as mm

native=vision.TiledVision
selection=overlap.SourceMatchedVision
assert torch.cuda.get_device_name(0)=='NVIDIA L4'

def run(case):
    is_portrait=case.startswith('portrait')
    folder=ROOT/'results'/('portrait' if is_portrait else case)
    out=ROOT/'auto-results'/case;out.mkdir(exist_ok=True,parents=True)
    report=json.loads((folder/'report.json').read_text())
    with Image.open(folder/'source.png') as im: source=matrix.to_tensor(im)
    captions=([t['caption'] for t in report['captions']['256']] if is_portrait
               else [t['caption'] for t in report['geometry']['tiles']])
    seed=report['upscale_seed']
    auto=case not in ('portrait-core','portrait-blend')
    choices=[]
    class Record(selection):
        def __call__(self,*args,**opts):
            result=super().__call__(*args,**opts)
            if self.choices and not choices:choices.extend(self.choices)
            return result
    class CoreWriter(native):
        def __init__(self,tiles,contexts,patch_size,h,w):
            xe,ye=square_tile_grid(w*8,h*8,'9')
            tnew=[]
            for i,t in enumerate(tiles):
                col,row=i%3,i//3
                ww=torch.zeros_like(t.weight)
                ww[ye[row]//8-t.top:ye[row+1]//8-t.top,xe[col]//8-t.left:xe[col+1]//8-t.left]=1
                tnew.append(replace(t,weight=ww))
            super().__init__(tnew,contexts,patch_size,h,w)
            assert torch.all(self.weights==1)
    start=time.monotonic()
    with ExitStack() as stack:
        if is_portrait:
            cap=iter(captions);encode=vision._encode_tile
            def local(c,template,text,crop,f):return encode(c,template,next(cap),crop,f)
            stack.enter_context(patch.object(vision,'_encode_tile',local))
            stack.enter_context(patch.object(vision,'_position_on_canvas',lambda args:args))
            stack.enter_context(patch.object(vision,'upscale_with_vision',vision.upscale_with_vision.__wrapped__))
            stack.enter_context(patch.object(overlap,'SourceMatchedVision',Record))
            if case=='portrait-core':stack.enter_context(patch.object(vision,'TiledVision',CoreWriter))
            decode=vae.decode_tiled
            def safe_decode(*args,**kw):
                mm.unload_all_models()
                with torch.inference_mode():return decode(*args,**kw)
            stack.enter_context(patch.object(vae,'decode_tiled',safe_decode))
            result,=STKrea2UltimateUpscale().upscale(source,model,[],[],vae,3.,seed,8,1.,'euler','simple',.8,1024,256,0,
                    'Vision',0.,'','','',1024,.1,.4,clip=clip,vision_prompt=(folder/'original-prompt.txt').read_text(),
                    tile_count='9',vision_overlap='Match source' if auto else 'Blend')
            assert next(cap,None) is None
        else:
            # Use the original matrix's pad-and-trim geometry and unchanged tile captions.
            import comfy.samplers
            original=harness.common_ksampler
            def sample(m,seed,steps,cfg,sampler,scheduler,pos,neg,latent,**kw):
                holder['reference']=m.model.process_latent_in(latent['samples'])
                return original(m,seed,steps,cfg,sampler,scheduler,pos,neg,latent,**kw)
            holder={}
            class Deferred(native):
                def __init__(self,*args):self.args=args;self.wrapper=None
                def __call__(self,*args,**opts):
                    if self.wrapper is None:self.wrapper=Record(*self.args,holder.pop('reference'))
                    return self.wrapper(*args,**opts)
            stack.enter_context(patch.object(harness,'common_ksampler',sample))
            stack.enter_context(patch.object(vision,'TiledVision',Deferred))
            result,=harness.STKrea2TurboMatrixTest().upscale(source,model,clip,vae,report['scene']['prompt'],seed,
                  variant='euler-simple',reference_captions=captions)
    assert not auto or choices
    assert result.shape[1:3]==(source.shape[1]*3,source.shape[2]*3) and torch.isfinite(result).all()
    matrix.save_png(result,out/'output.png')
    details=dict(case=case,mode='Match source' if auto else case,seed=seed,seconds=round(time.monotonic()-start,1),
       source_sha256=hashlib.sha256((folder/'source.png').read_bytes()).hexdigest(),captions=captions,choices=choices,
       output_sha256=hashlib.sha256((out/'output.png').read_bytes()).hexdigest(),
       torch=str(torch.__version__),gpu=torch.cuda.get_device_name(0),rule_sha256=hashlib.sha256((TRIAL/'krea2_overlap.py').read_bytes()).hexdigest())
    matrix.save_json(out/'details.json',details)
    print('RUN SAVED',case, 'owners',[(a,c) for a,b,c in choices if c is not None],flush=True)
for case in cases:
    print('RUN START',case,flush=True);run(case)
