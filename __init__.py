import comfy.samplers
from comfy_extras.nodes_images import ResizeAndPadImage
import nodes


class PixelAnchoredRemaster:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "tile_size_vae": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
                "interpolation": (["area", "bicubic", "nearest-exact", "bilinear", "lanczos"], {"default": "lanczos"}),
                "upscale_method": (nodes.LatentUpscale.upscale_methods, {"default": "bislerp"}),
                "remaster_steps": ("INT", {"default": 16, "min": 1, "max": 1000}),
                "remaster_cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_3m_sde_gpu"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras"}),
                "remaster_denoise": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "execute"
    CATEGORY = "Image/Upscaling"

    def _sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise):
        return nodes.common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent,
            denoise=denoise,
        )[0]

    def _decode_tiled(self, vae, samples, tile_size, overlap):
        return vae.decode_tiled(samples, tile_x=tile_size, tile_y=tile_size, overlap=overlap)

    def _encode_tiled(self, vae, image, tile_size, overlap):
        return vae.encode_tiled(image[:, :, :, :3], tile_x=tile_size, tile_y=tile_size, overlap=overlap)

    @staticmethod
    def _fit_to_multiple_of_8(value):
        return max(8, (value // 8) * 8)

    def execute(
        self,
        image,
        model,
        vae,
        positive,
        negative,
        seed,
        tile_size_vae,
        overlap,
        interpolation,
        upscale_method,
        remaster_steps,
        remaster_cfg,
        sampler_name,
        scheduler,
        remaster_denoise,
    ):
        _, input_h, input_w, _ = image.shape

        # Input image is already the 2x hires-fixed image.
        # Downscale/pad to 75% of that (1.5x original), then upscale the latent back to the input size.
        remaster_w = self._fit_to_multiple_of_8(round(input_w * 1.5))
        remaster_h = self._fit_to_multiple_of_8(round(input_h * 1.5))
        final_w = self._fit_to_multiple_of_8(input_w)
        final_h = self._fit_to_multiple_of_8(input_h)

        pad_color = "white"

        remastered = ResizeAndPadImage.execute(
            image,
            remaster_w,
            remaster_h,
            pad_color,
            interpolation,
        )[0]
        remaster_latent = self._encode_tiled(vae, remastered, tile_size_vae, overlap)
        remaster_latent = nodes.LatentUpscale().upscale({"samples": remaster_latent}, upscale_method, final_w, final_h, "disabled")[0]
        remaster_samples = self._sample(
            model,
            seed,
            remaster_steps,
            remaster_cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            remaster_latent,
            remaster_denoise,
        )
        final_image = self._decode_tiled(vae, remaster_samples["samples"], tile_size_vae, overlap)

        return (final_image, remaster_samples)


NODE_CLASS_MAPPINGS = {"PixelAnchoredRemaster": PixelAnchoredRemaster, "PixelAnchordRemaster": PixelAnchoredRemaster}
NODE_DISPLAY_NAME_MAPPINGS = {"PixelAnchoredRemaster": "Pixel Anchored Remaster", "PixelAnchordRemaster": "Pixel Anchored Remaster"}
