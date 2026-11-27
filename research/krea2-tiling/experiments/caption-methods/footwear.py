"""Exploratory footwear-direction check. The two views must agree before a fact is emitted."""
import re
from caption_methods import crop, generate

SHOE=re.compile(r'\b(?:sneakers?|shoes?|boots?|slippers?|high heels?|footwear|pointe shoes?)\b', re.I)
PROMPT=(
    "Look for the largest clearly visible shoe or boot in this image. Identify the tip at the toes, "
    "distinguishing it from the heel. From heel toward toe, what is its main direction in the picture? "
    "Use LEFT or RIGHT if it points sideways, UP or DOWN for vertical. Use FRONT for towards the viewer "
    "and BACK for away from the viewer if more prominent than a sideways direction. "
    "If the toe or footwear is unclear or absent, use UNKNOWN. Output just one uppercase word."
)
OPPOSITE=dict(LEFT="RIGHT",RIGHT="LEFT",UP="UP",DOWN="DOWN",FRONT="FRONT",BACK="BACK",UNKNOWN="UNKNOWN")
WORD=dict(LEFT="left",RIGHT="right",UP="top",DOWN="bottom",FRONT="viewer",BACK="back of the scene")
def read_direction(clip,view):
    raw=generate(clip,PROMPT,view,10)
    match=re.fullmatch(r'[\s\W]*(LEFT|RIGHT|UP|DOWN|FRONT|BACK|UNKNOWN)[\s\W]*',raw,re.I)
    return match.group(1).upper() if match else 'UNKNOWN',raw
def check(clip,source,tile,base):
    if not SHOE.search(base):
        return dict(caption=base,status='no footwear cue',answers=[])
    core=crop(source,tile['core_source_xyxy'])
    a,raw_a=read_direction(clip,core)
    b,raw_b=read_direction(clip,core.flip([2]))
    if a=='UNKNOWN' or OPPOSITE[b]!=a:
        return dict(caption=base,status='abstained',answers=[raw_a,raw_b])
    direction=WORD[a]
    if a in {'UP','DOWN','LEFT','RIGHT'}:target=f"the {direction} of the image"
    else:target=f"the {direction}"
    return dict(caption=f"{base} The toe of the most prominent shoe points toward {target}.",
                status='accepted',direction=a,answers=[raw_a,raw_b])
