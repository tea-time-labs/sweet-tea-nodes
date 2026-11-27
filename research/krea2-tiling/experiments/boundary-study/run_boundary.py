"""Caption benchmark and all-nine fox inputs, using pinned Krea 2 ComfyUI CLIP."""
import hashlib,json,os,sys,time
from pathlib import Path
from boundary import MODES,PROMPT,cue
from caption_methods import generate

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def run():
    comfy=ROOT/'ComfyUI'
    os.chdir(comfy)
    sys.path.insert(0,str(comfy))
    sys.argv[:]=[sys.argv[0],'--use-pytorch-cross-attention']
    import comfy.options
    comfy.options.enable_args_parsing()
    import folder_paths, nodes, numpy as np, torch
    from PIL import Image
    torch.set_grad_enabled(False)
    folder_paths.add_model_folder_path('text_encoders',str(ROOT/'weights/text_encoders'),is_default=True)
    clip,=nodes.CLIPLoader().load_clip('qwen3vl_4b_fp8_scaled.safetensors','krea2','default')
    selected=json.loads((HERE/'selection.json').read_text())
    fox='10-wildlife-family'
    jobs=[(fox,i,'full_box') for i in range(1,10)]
    jobs.extend((c,i,m) for c,i,_ in selected for m in MODES)
    for case,i,mode in dict.fromkeys(jobs):
        folder=ROOT/'results'/case
        path=HERE/'outputs'/case/f'{i}-{mode}.json'
        if path.exists():continue
        report=json.loads((folder/'report.json').read_text())
        tile=report['geometry']['tiles'][i-1]
        with Image.open(folder/'source.png') as source:
            image,box,coords=cue(source,tile,mode)
        prompt=PROMPT.format(coords=coords,brief=report['scene']['prompt'])
        data=torch.from_numpy(np.array(image)).unsqueeze(0).float()/255
        start=time.monotonic()
        caption=generate(clip,prompt,data,100)
        torch.cuda.synchronize()
        result=dict(case=case,tile=i,mode=mode,caption=caption,prompt=prompt,
                    core=tile['core_source_xyxy'],padded=tile['padded_source_xyxy'],box_in_view=box,
                    normalized=coords,view_size=image.size,view_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    source_sha256=hashlib.sha256((folder/'source.png').read_bytes()).hexdigest(),
                    gpu=torch.cuda.get_device_name(0),seconds=round(time.monotonic()-start,2))
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)
        print('SAVED',case,i,mode,flush=True)
    print('BOUNDARY STUDY COMPLETE',flush=True)
if __name__=='__main__':run()
