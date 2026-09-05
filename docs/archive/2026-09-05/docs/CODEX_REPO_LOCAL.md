# Sweet Tea Nodes Local Rules

- CI authority is limited to public node-output contracts, Python compilation, and Comfy registry validation.
- Do not test private tiling helpers, retry counts, internal call order, or upstream ComfyUI implementation behavior.
- Tests that require tensor behavior must run against the pinned CPU runtime and must fail when that runtime is unavailable; silent skips cannot gate release.
- Visual quality, performance, and compatibility explorations are diagnostic unless a direct product or architecture invariant is established.
