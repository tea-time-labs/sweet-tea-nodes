"""Provisional tile-caption methods for the existing Krea 2 CLIP. All boxes are source-pixel xyxy."""

import re

METHODS = ("core", "context_check", "split_check")

CORE = (
    "Write a short literal caption for this exact image fragment, for a painter reconstructing it. "
    "Identify the dominant surfaces and distinct visible pieces, including important pieces cut by an edge. "
    "For partial objects name the visible part. Include a clear direction, contact or overlap when it matters. "
    "Use viewer-relative directions. Keep indistinct details general. Return only one or two short English sentences."
)
DRAFT = (
    "This photograph is a local neighborhood around a smaller target that you cannot see separately yet. "
    "Suggest up to six concrete object or material names actually visible in this photograph. "
    "The following creative brief may help identify them. It can describe things outside this photograph. "
    "Return a short list of candidate nouns only. Creative brief: "
)
VERIFY = (
    "You see the entire target fragment. Nearby material suggested these possible nouns: {notes}. "
    "Check each candidate against the pixels visible here. Discard candidates outside this view. "
    "Find important parts the list missed. "
)
PART = (
    "List the two or three most visually important fragments or surfaces visible here, in a short phrase. "
    "Name the visible portion of clipped subjects. Keep ambiguous forms literal."
)
JOIN = (
    "Use this one complete target image to check two field notes from its adjoining halves. "
    "The notes may be mistaken. Reconcile continuous objects across the midpoint, "
    "and keep distinct fragments at opposite outer edges when you see both. "
    "{axis} notes: {first} | {second}. "
)

def generate(clip, prompt, image, limit):
    tokens = clip.tokenize(prompt, images=[image], min_length=1, thinking=False)
    attached = sum(isinstance(elem, dict) and elem.get("type") == "image"
                   for row in tokens["qwen3vl_4b"] for elem, *_ in row)
    if attached != 1:
        raise ValueError("Expected exactly one bound Qwen3-VL image.")
    text = clip.decode(clip.generate(tokens, do_sample=False, max_length=limit))
    text = re.sub(r"^caption\s*:\s*", "", text.rsplit("</think>", 1)[-1].strip(),
                  flags=re.IGNORECASE).strip().strip('"').strip()
    if not text or "<think>" in text:
        raise ValueError("Caption generation returned no usable text.")
    return text

def crop(source, bounds):
    x1, y1, x2, y2 = bounds
    return source[:, y1:y2, x1:x2, :3]

def describe(method, clip, source, original_prompt, tile):
    core = crop(source, tile["core_source_xyxy"])
    if method == "core":
        return generate(clip, CORE, core, 100), []
    if method == "context_check":
        neighborhood = crop(source, tile["padded_source_xyxy"])
        notes = generate(clip, DRAFT + original_prompt, neighborhood, 48)
        return generate(clip, VERIFY.format(notes=notes) + CORE, core, 100), [notes]
    if method == "split_check":
        height, width = core.shape[1:3]
        if width >= height and width >= 2:
            cut = width // 2
            halves, axis = (core[:, :, :cut], core[:, :, cut:]), "Left and right"
        elif height >= 2:
            cut = height // 2
            halves, axis = (core[:, :cut], core[:, cut:]), "Upper and lower"
        else:
            return generate(clip, CORE, core, 100), []
        first, second = [generate(clip, PART, half, 48) for half in halves]
        prompt = JOIN.format(axis=axis, first=first, second=second) + CORE
        return generate(clip, prompt, core, 100), [first, second]
    raise ValueError(f"Unknown caption method {method!r}")
