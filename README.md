# Pixel Anchored Remaster

Pixel Anchored Remaster is a remasterer, not a conventional upscaler. It takes an already HiRes Fixed image, reduces upscale artifacts through pixel-space downscaling, restores the image to working resolution in latent space, and then uses KSampler to rebuild detail. The goal is not pixel-perfect preservation — minor drift is fine if it gives a better final image.

Important: this is built for 2x upscale workflows. It is not meant for 4x. In my testing, 2x is where it makes sense and 4x starts going off the rails.

## What it is

- A single ComfyUI node for the Pixel Anchored Remaster pass
- An IMAGE-in, IMAGE-out workflow
- A tiled VAE encode/decode setup with latent sampling in the middle
- A remaster pass that cares more about the final image than exact preservation

## What it is not

- Not a conventional resize upscaler
- Not a detail-copying machine
- Not a HiRes Fix replacement
- Not meant to preserve every edge, texture, or tiny feature exactly
- Not for 4x upscale workflows

If you want perfect fidelity, this is probably the wrong tool.
If you want the image to stay recognizable while allowing a little drift for a better result, this is the right kind of tool.

## How it works

This node expects an image that has already gone through HiRes Fix or another high-res stage.

From there it does three things:

1. Downscales the image in pixel space
   - This helps remove some of the artifacts and harshness that come from aggressive upscaling.
   - It also clears out some of the noisy upscale texture so the next pass has something cleaner to work with.

2. Encodes the result into latent space and upscales the latent back to the input size
   - The downscaled image gets VAE-encoded with tiled VAE support.
   - The latent is then upscaled back to the original input dimensions.
   - This gives the sampler more room to work without losing the overall working size.

3. Runs KSampler to rebuild detail
   - This is where the remaster actually happens.
   - The sampler can add structure back in where the first pass left things soft, messy, or fake-looking.
   - The image is not just being enlarged — it is being reinterpreted.

In plain English:
pixel downscaling cleans the image up,
latent upscaling gives the model room to work,
and KSampler ties both together into one remaster pass.

## Expected behavior

- Minor drift is normal
- Small changes are acceptable if they improve the final image
- The output should still look like the same image, but not a perfect clone
- The goal is a better image, not an identical one

## Install

Drop this folder into `ComfyUI/custom_nodes/` and restart ComfyUI.

## Node

- `Pixel Anchored Remaster`

## Controls

- `upscale_method`: latent upscaling method
- `sampler_name`: sampler used for the remaster pass
- `scheduler`: scheduler used for the remaster pass
- `tile_size_vae` and `overlap`: tiled VAE settings
- `remaster_steps`, `remaster_cfg`, `remaster_denoise`: sampling strength controls

## Notes

- Default settings worked best in my testing, but some use cases may benefit from tweaking
- Best results in my testing were with anime-style models, especially portraits
- This node is intended for SDXL / Pony / Illustrious-style latent spaces
- It performs poorly with Anima, and likely other different latent spaces such as Chroma, where the remaster step tends to produce ugly results
