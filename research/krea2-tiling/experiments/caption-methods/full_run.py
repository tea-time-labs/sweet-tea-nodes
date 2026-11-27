"""Exercise all nine captions and the ordinary test node sampler entry point on the original florist input."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'test-harness'))
import run_matrix as matrix
if __name__=='__main__':
    model,clip,vae=matrix.load_runtime()
    from experimental_node import NODE_CLASS_MAPPINGS
    matrix.nodes.NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS)
    case='01-florist-handoff'
    scene=ROOT/'results'/case
    report=json.loads((scene/'report.json').read_text())
    with matrix.Image.open(scene/'source.png') as image: source=matrix.to_tensor(image)
    out=HERE/'renders'/case/'full_verified'
    out.mkdir(parents=True,exist_ok=True)
    metadata={}
    def prepared(data):
        assert data['tile_count']==9 and data['padding']==256
        for old,new in zip(report['geometry']['tiles'],json.loads(json.dumps(data['tiles'])),strict=True):
            assert all(old[k]==new[k] for k in old if k!='caption')
        metadata['captions']=[t['caption'] for t in data['tiles']]
        metadata['changed_tiles']=[i for i,(t,c) in enumerate(zip(report['geometry']['tiles'],metadata['captions']),1) if t['caption']!=c]
        matrix.save_json(out/'prepared.json',metadata)
    def render(name,img,metrics):
        path=out/'output.png';matrix.save_png(img,path)
        matrix.save_json(out/'report.json',dict(**metadata,case=case,method='full_verified',variant=name,metrics=metrics,
          seed=report['upscale_seed'],gpu=matrix.torch.cuda.get_device_name(0),
          source_sha256=hashlib.sha256((scene/'source.png').read_bytes()).hexdigest(),
          output_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print('FULL WORKFLOW COMPLETE',case,flush=True)
    node=matrix.nodes.NODE_CLASS_MAPPINGS['STKrea2CaptionMethodsTest']()
    node.upscale(source,model,clip,vae,report['scene']['prompt'],report['upscale_seed'],variant='euler-simple',
                 control='Vision',caption_method='split_check',verify_footwear=True,on_prepared=prepared,on_render=render)
