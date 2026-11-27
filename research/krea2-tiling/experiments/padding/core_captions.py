from PIL import Image, ImageDraw
import numpy as np
import torch


def core_captions(source, prompt, clip, layout, tiles):
    x_edges, y_edges = layout
    source_image = Image.fromarray((source[0].detach().cpu().numpy().clip(0,1)*255).round().astype(np.uint8))
    request = (
        "The first picture is the whole source. Yellow marks the padded area and cyan marks the inner tile. "
        "The second picture is the clean padded area, to help you understand what is nearby. The third picture is "
        "the clean inner tile, the ONLY area you should describe. Give one short descriptive sentence about visible "
        "forms, texture, and partial subjects in that third picture. For partial faces, say which part actually appears. "
        "A person or detail visible only in the first or second picture must stay out of your answer. "
        "If the third picture shows hair or background alone, describe that alone. "
        "Use the original creative prompt only as context, and output just the grounded caption.\n"
        + prompt
    )

    def source_box(x1,y1,x2,y2):
        return x1//3, y1//3, (x2+2)//3, (y2+2)//3

    def crop(box):
        x1,y1,x2,y2 = box
        return source[:,y1:y2,x1:x2,:]

    captions = []
    for i, tile in enumerate(tiles):
        row,col = divmod(i,3)
        core = source_box(x_edges[col],y_edges[row],x_edges[col+1],y_edges[row+1])
        padded = source_box(tile.left*8, tile.top*8, tile.right*8, tile.bottom*8)
        ref = source_image.resize((512,512), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(ref)
        for limits,color in [(padded, "#f7cb29"), (core, "#24faff")]:
            x1,y1,x2,y2 = limits
            draw.rectangle((x1//2,y1//2,min(511,x2//2-1),min(511,y2//2-1)), outline=color, width=3)
        tokens = clip.tokenize(request, images=[torch.from_numpy(np.array(ref)).unsqueeze(0).float()/255, crop(padded), crop(core)],
                                   min_length=1, thinking=False)
        assert sum(isinstance(t[0],dict) and t[0].get("type")=="image"
                   for row in tokens["qwen3vl_4b"] for t in row) == 3
        text = clip.decode(clip.generate(tokens, do_sample=False, max_length=96))
        text = text.rsplit("</think>",1)[-1].strip().strip('"').strip()
        if not text or "<think>" in text:
            raise ValueError(text)
        captions.append({"tile": i+1, "core":core, "padded":padded,"caption":text})
        print("CORE CAPTION", i+1, text, flush=True)
    return captions
