"""Independent depth injection and reference-token attention on the native Krea 2 forward."""

import math

import torch
import torch.nn.functional as F


def reference_attention(strengths, previous=None):
    """Bias text/target reads; reference queries retain their original softmax."""

    def attention(backend, q, k, v, heads, mask=None, **kwargs):
        options = kwargs["transformer_options"]
        lengths = options.get("reference_image_num_tokens", ())
        if options.get("block_type") != "single" or not lengths:
            if previous is not None:
                return previous(backend, q, k, v, heads, mask=mask, **kwargs)
            return backend(q, k, v, heads, mask=mask, **kwargs)
        if len(lengths) != len(strengths):
            raise ValueError("The number of Krea 2 reference images changed during sampling.")
        # One broadcast row avoids allocating a square mask for a 9 MP upscale.
        # Split query rows only. Both groups use every key/value in this layer.
        start = k.size(-2) - sum(lengths)
        bias = q.new_zeros((1, 1, 1, (k.size(-2) + 7) // 8 * 8))[..., : k.size(-2)]
        pos = start
        for length, strength in zip(lengths, strengths):
            bias[..., pos : pos + length] = math.log(strength)
            pos += length
        target_mask = bias
        ref_mask = mask
        if mask is not None:
            if mask.dtype == torch.bool:
                mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(~mask, -torch.inf)
            uniform = mask.shape[-2] == 1
            target_mask = (mask if uniform else mask[..., :start, :]) + bias
            ref_mask = mask if uniform else mask[..., start:, :]
        first = F.scaled_dot_product_attention(q[:, :, :start], k, v, attn_mask=target_mask)
        last = F.scaled_dot_product_attention(q[:, :, start:], k, v, attn_mask=ref_mask)
        out = torch.cat((first, last), dim=2)
        return out.transpose(1, 2).flatten(2)

    return attention


def load_depth(model, filename, strength):
    from comfy.patcher_extension import CallbacksMP, WrappersMP
    from .krea2_control import required_node

    loader = required_node("Krea2ControlLoRALoader", "facok/comfyui-krea2-controlnet")()
    (model,) = loader.load_lora(model, filename, strength)
    key = "krea2_control"
    projection = model.get_attachment(key)["control_projection"]
    # facok's normal wrapper expects depth tokens for the entire sequence.
    # Use its trained depth columns for target tokens only, before the blocks.
    model.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, key)
    model.remove_injections(key)
    for event in (CallbacksMP.ON_CLEANUP, CallbacksMP.ON_DETACH):
        model.remove_callbacks_with_key(event, key)
    weight = projection.weight[:, projection.image_features :].detach().contiguous()
    return model, weight


class TileGuides:
    def __init__(self, refs, depth, depth_weight, depth_strength, patch):
        self.refs = refs
        self.depth = depth
        self.depth_weight = depth_weight
        self.depth_strength = depth_strength
        self.patch = patch

    def __call__(self, tile, x, options):
        refs = [latent[tile.region].to(x) for latent, _ in self.refs]
        strengths = [strength for _, strength in self.refs]
        if any(s != 1.0 for s in strengths):
            options["optimized_attention_override"] = reference_attention(
                strengths, options.get("optimized_attention_override")
            )
        if self.depth is not None:
            depth = self.depth[tile.region].to(x)
            if depth.ndim == 5:
                depth = depth.squeeze(2)
            p = self.patch
            b, c, h, w = depth.shape
            tokens = depth.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
            tokens = tokens.reshape(b, (h // p) * (w // p), c * p * p)

            def inject(args):
                img = args["img"]
                residual = F.linear(tokens.to(img), self.depth_weight.to(img)) * self.depth_strength
                count = residual.shape[1]
                img = torch.cat((img[:, :count] + residual, img[:, count:]), dim=1)
                return {**args, "img": img}

            patches = options.get("patches", {})
            options["patches"] = {**patches, "post_input": [*patches.get("post_input", ()), inject]}
        return refs or None, options
