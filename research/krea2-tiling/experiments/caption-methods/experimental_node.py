"""Experimental node for the matrix harness. It uses the existing Vision/Depth/Canny render path."""
import torch
import comfy.model_management as mm
from caption_methods import METHODS,describe
from footwear import check
from krea16_test_node import STKrea2TurboMatrixTest,geometry

class STKrea2CaptionMethodsTest(STKrea2TurboMatrixTest):
    @classmethod
    def INPUT_TYPES(cls):
        inputs=super().INPUT_TYPES()
        inputs['required']['caption_method']=([*METHODS],{'default':'split_check'})
        inputs['required']['verify_footwear']=('BOOLEAN',{'default':False})
        return inputs

    DESCRIPTION='Compare per-tile caption methods with the workflow’s Qwen3-VL using the existing tiled Vision sampler.'
    @torch.inference_mode(False)
    @torch.no_grad()
    def upscale(self,image,model,clip,vae,original_prompt,seed,steps=8,denoise=.8,padding=256,
                variant='euler-simple',control='Vision',control_strength=.7,caption_method='split_check',
                verify_footwear=False,**kwargs):
        if image.shape[0]!=1:
            raise ValueError('Use one source image for this test node.')
        if caption_method not in METHODS:
            raise ValueError(f'Unknown caption method {caption_method!r}')
        source=image[...,:3]
        diffusion=model.get_model_object('diffusion_model')
        height,width=source.shape[1:3]
        _,_,_,boxes=geometry(width,height,padding,diffusion.patch)
        captions=[]
        for tile in boxes:
            mm.throw_exception_if_processing_interrupted()
            caption,_=describe(caption_method,clip,source,original_prompt,tile)
            if verify_footwear:
                caption=check(clip,source,tile,caption)['caption']
            captions.append(caption)
        return super().upscale(image,model,clip,vae,original_prompt,seed,steps,denoise,padding,variant,control,
                               control_strength,reference_captions=captions,**kwargs)

NODE_CLASS_MAPPINGS={'STKrea2CaptionMethodsTest':STKrea2CaptionMethodsTest}
NODE_DISPLAY_NAME_MAPPINGS={'STKrea2CaptionMethodsTest':'ST Krea 2 Caption Methods (Experiment)'}
