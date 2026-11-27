"""Choose one source-aligned prediction in each early Krea2 Vision overlap."""
from dataclasses import replace

import torch
import torch.nn.functional as F

from .krea2_vision import TiledVision


def matched_tiles(tiles, total, errors):
    """Keep an overlap together when comparing candidates against the source."""
    height, width = total.shape
    xs = sorted({v for tile in tiles for v in (tile.left, tile.right)})
    ys = sorted({v for tile in tiles for v in (tile.top, tile.bottom)})
    weights = [tile.weight.clone() for tile in tiles]
    choices = []
    for top, bottom in zip(ys, ys[1:]):
        for left, right in zip(xs, xs[1:]):
            covering = [i for i, tile in enumerate(tiles) if tile.top <= top < tile.bottom
                        and tile.left <= left < tile.right]
            if len(covering) < 2:
                continue
            rect = (top, bottom, left, right)
            crops = {i: (slice(top-tiles[i].top, bottom-tiles[i].top),
                         slice(left-tiles[i].left, right-tiles[i].left)) for i in covering}
            scores = {i: float(errors[i][crops[i]].mean()) for i in covering}
            first, second = sorted(scores, key=scores.get)[:2]
            # Small score differences leave the existing blend alone.
            owner = first if scores[first] < .9 * scores[second] else None
            choices.append((rect, scores, owner))
            if owner is None:
                continue
            yy = torch.arange(top, bottom, device=total.device) + .5
            xx = torch.arange(left, right, device=total.device) + .5
            v = torch.ones_like(yy)
            h = torch.ones_like(xx)
            if top:
                v = v.minimum((yy-top)/4)
            if bottom != height:
                v = v.minimum((bottom-yy)/4)
            if left:
                h = h.minimum((xx-left)/4)
            if right != width:
                h = h.minimum((right-xx)/4)
            alpha = torch.minimum(v[:, None], h[None, :])
            for i in covering:
                target = total[top:bottom, left:right] if i == owner else 0.
                weights[i][crops[i]].lerp_(torch.as_tensor(target, dtype=weights[i].dtype,
                                                          device=weights[i].device), alpha)
    return [replace(tile, weight=w) for tile, w in zip(tiles, weights)], choices


class SourceMatchedVision(TiledVision):
    def __init__(self, tiles, contexts, patch, h, w, source):
        super().__init__(tiles, contexts, patch, h, w)
        self.source = source
        self.early = None
        self.choices = []

    def __call__(self, executor, x, timesteps, context, attention_mask=None,
                 ref_latents=None, transformer_options=None, **kwargs):
        options = transformer_options or {}
        args = (x, timesteps, context, attention_mask, ref_latents, options)
        sigmas = options.get("sample_sigmas")
        groups = options.get("cond_or_uncond") or [0]
        if sigmas is None or timesteps[0] <= sigmas[len(sigmas) // 2]:
            return super().__call__(executor, *args, **kwargs)
        if self.early is not None:
            return self.early(executor, *args, **kwargs)
        if 0 not in groups:
            return super().__call__(executor, *args, **kwargs)

        # The normal wrapper handles context, positions, masks and CFG groups.
        # Cache its first predictions to change the weights without another model pass.
        predictions = []
        def capture(*inputs, **opts):
            pred = executor(*inputs, **opts)
            predictions.append(pred)
            return pred

        super().__call__(capture, *args, **kwargs)
        batch = x.shape[0] // len(groups)
        cond = groups.index(0)
        row = cond * batch
        source = self.source.to(x.device, dtype=torch.float32)
        errors = []
        for i, tile in enumerate(self.tiles):
            prediction = predictions[i*len(groups)+cond][:1].float()
            # For Krea2 flow, the clean prediction is x - sigma * velocity.
            clean = x[row:row+1][tile.region].float() - timesteps[row].float() * prediction
            difference = clean - source[tile.region]
            difference = difference.reshape(1, -1, *difference.shape[-2:])
            coarse = F.avg_pool2d(difference, 7, stride=1, padding=3, count_include_pad=False)
            errors.append(coarse.square().mean(dim=(0,1)).detach().cpu())
        chosen, self.choices = matched_tiles(self.tiles, self.weights, errors)
        self.early = TiledVision(chosen, self.contexts, self.patch, *self.weights.shape)
        self.source = None
        cached = iter(predictions)
        return self.early(lambda *inputs, **opts: next(cached), *args, **kwargs)
