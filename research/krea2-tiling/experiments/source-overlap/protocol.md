# Early overlap ownership in Krea 2 Vision

Compare the same 3× source, initial noise seed, nine captions, image crops, sampler, local positions and 256-output-pixel padding. No Canny or Depth is active. Eight Euler/simple steps use denoise 0.8 and CFG 1.

The candidate scores each first-step tile prediction against the clean source latent in their common area. Scores are mean squared differences after a 7 × 7 latent-pixel average. All candidates use the same source coordinates. If one has over 10% lower error than the runner-up, it writes that overlap during the first half of the schedule, with a four-latent-pixel transition at internal borders. Close scores retain the normal blend. The second half uses ordinary blending everywhere. This rule never reads scene labels or caption text and uses no manually placed region.

Use the known failing portrait for development. Record whether its near-side eye is single and in the source position. Also test the simpler geometric rule in which padding is available to every tile for reading, but only core pixels can be written on all eight steps. Replay ordinary blending through the same portrait runner as a control.

Test the unchanged candidate on the fox family, the orchard and the museum. For the foxes, count young faces and compare them with one adult and two kits in the source. The orchard is a preservation case. Check three people, one shared crate and the lift. In the museum, check the adult, pointing child, and extra left foreground child seen under ordinary blending. Inspect the complete frames and the joints between tiles. Keep the actual rectangles, per-tile scores and decisions for each image. One seed per source.

## Disposition

The automatic score rule missed the known portrait failure. The lower tile scored 0.31153 and the upper tile 0.32981, which left the default blend active under the pilot's 10% condition. The fox also retained an extra young face. The rule was withdrawn. No retuned cutoff was tested. The orchard and museum qualification were discontinued, so this trial supplies no successful preservation control and no generalization claim. The separate portrait core-only run tests write geometry directly. The shared node checkout has no remaining changes from the pilot. Its experimental changes are preserved in pilot.patch.
