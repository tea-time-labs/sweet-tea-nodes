"""Construct one image and one region question. Source boxes use half-open pixel bounds."""
from PIL import Image, ImageDraw

MODES = ('padded_box','full_box','full_coords','full_spotlight')
SHORT = {
    'padded_box':'Padded view + border',
    'full_box':'Full scene + border',
    'full_coords':'Full scene + coordinates',
    'full_spotlight':'Full scene + dim surround',
    'split_check':'Two-half reference',
}
PROMPT = (
    "Describe the visible contents inside the target rectangle in this image. "
    "Its [left, top, right, bottom] coordinates are {coords} on a 0 to 1000 scale, "
    "with the origin at the upper left. The pixels around the target can help you recognize a fragment. "
    "Report only the portion actually inside: its main surfaces, any objects cut by the edges, and "
    "clear relative position, orientation or contact. Ignore any locator markings and lighting used "
    "to identify the rectangle. Use one or two short literal sentences. "
    "Scene brief for context only: {brief}"
)

def cue(source, tile, mode):
    if mode not in MODES:
        raise ValueError(mode)
    source=source.convert('RGB')
    window=tuple(tile['padded_source_xyxy']) if mode=='padded_box' else (0,0,*source.size)
    x0,y0,x1,y1=window
    l,t,r,b=tile['core_source_xyxy']
    box=(l-x0,t-y0,r-x0,b-y0)
    view=source.crop(window)
    w,h=view.size
    l,t,r,b=box
    if not (0<=l<r<=w and 0<=t<b<=h):
        raise ValueError((box,window))
    if mode in ('padded_box','full_box'):
        # Paint an exterior ring; preserve the exact core, even where it meets the source border.
        stroke=4
        inner=view.crop(box)
        ImageDraw.Draw(view).rectangle((l-stroke,t-stroke,r+stroke-1,b+stroke-1),fill=(255,0,175))
        view.paste(inner,(l,t))
    elif mode=='full_spotlight':
        inner=view.crop(box)
        view=view.point([round(v*.30) for v in range(256)]*3)
        view.paste(inner,(l,t))
    coords=[round(1000*l/w),round(1000*t/h),round(1000*r/w),round(1000*b/h)]
    return view,box,coords
