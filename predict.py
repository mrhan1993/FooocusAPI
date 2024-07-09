"""
function to predict the output of the model
"""
import copy
import os
import time
from typing import Any, List
import uuid

from cog import BasePredictor, BaseModel, Input, Path
from modules import flags
from modules.async_worker import AsyncTask, async_tasks
from modules.config import (
    default_prompt,
    default_prompt_negative,
    default_styles,
    default_output_format,
    default_aspect_ratio,
    default_sample_sharpness,
    default_cfg_scale,
    default_base_model_name,
    default_black_out_nsfw,
    default_refiner_model_name,
    default_refiner_switch,
    default_loras,
    default_cfg_tsnr,
    default_clip_skip,
    default_sampler,
    default_scheduler,
    default_vae,
    default_overwrite_step,
    default_overwrite_switch,
    default_inpaint_engine_version,
    default_max_lora_number,
    path_outputs
)

from apis.models.requests import CommonRequest
from apis.models.base import ImagePrompt, Lora
from apis.utils.api_utils import params_to_params
from apis.utils.img_utils import upload_to_array, upload_to_base64


def lora_parser(loras: list) -> list:
    """
    Convert loras to loras list
    """
    default_lora = {
        "enabled": True,
        "model_name": "None",
        "weight": 1,
    }
    while len(loras) < default_max_lora_number:
        loras.append(Lora(**default_lora))
    loras = loras[:default_max_lora_number]
    loras_list = []
    for lora in loras:
        loras_list.extend([
            lora.enabled,
            lora.model_name,
            lora.weight
        ])
    return loras_list


def control_net_parser(control_net: list) -> list:
    """
    Convert control net to control net list
    """
    default_cn_image = {
        "cn_img": None,
        "cn_stop": 0.6,
        "cn_weight": 0.6,
        "cn_type": "ImagePrompt"
    }
    while len(control_net) < flags.controlnet_image_count:
        control_net.append(ImagePrompt(**default_cn_image))

    control_net = control_net[:flags.controlnet_image_count]
    cn_list = []
    for cn in control_net:
        cn_list.extend([
            upload_to_array(cn.cn_img),
            cn.cn_stop,
            cn.cn_weight,
            cn.cn_type.value
        ])
    return cn_list


class Output(BaseModel):
    """
    Output model
    """
    # seeds: List[str]
    paths: List[Path]


class Predictor(BasePredictor):
    """
    Predictor for the model
    """

    def setup(self):
        """
        Load the model
        """
        import modules.default_pipeline as _
        # prepare_environment()


    def predict(
            self,
            prompt: str = Input(
                default="",
                description="Prompt to generate the image"),
            negative_prompt: str = Input(
                default="",
                description="Negative prompt to generate the image"),
            style_selections: str = Input(
                default=','.join(default_styles),
                description="Fooocus styles applied for image generation, separated by comma"),
            performance_selection: str = Input(
                default='Speed',
                choices=['Speed', 'Quality', 'Extreme Speed', 'Lightning', 'Hyper-SD'],
                description="Performance selection"),
            aspect_ratios_selection: str = Input(
                default='1152*896',
                choices=flags.sdxl_aspect_ratios,
                description="The generated image's size"),
            image_number: int = Input(
                default=1,
                ge=1, le=8,
                description="How many image to generate"),
            image_seed: int = Input(
                default=-1,
                description="Seed to generate image, -1 for random"),
            use_default_loras: bool = Input(
                default=True,
                description="Use default LoRAs"),
            sharpness: float = Input(
                default=2.0,
                ge=0.0, le=30.0),
            guidance_scale: float = Input(
                default=default_cfg_scale,
                ge=1.0, le=30.0),
            refiner_switch: float = Input(
                default=default_refiner_switch,
                ge=0.1, le=1.0),
            uov_input_image: Path = Input(
                default=None,
                description="Input image for upscale or variation, keep None for not upscale or variation"),
            uov_method: str = Input(
                default='Disabled',
                choices=["Disabled", "Vary (Subtle)", "Vary (Strong)", "Upscale (1.5x)", "Upscale (2x)", "Upscale (Fast 2x)", "Upscale (Custom)",]),
            upscale_multiple: float = Input(
                default=1.0,
                description="Only when Upscale (Custom)"),
            inpaint_additional_prompt: str = Input(
                default='',
                description="Prompt for image generation"),
            inpaint_input_image: Path = Input(
                default=None,
                description="Input image for inpaint or outpaint, keep None for not inpaint or outpaint. Please noticed, `uov_input_image` has bigger priority is not None."),
            inpaint_input_mask: Path = Input(
                default=None,
                description="Input mask for inpaint"),
            outpaint_selections: str = Input(
                default='',
                description="Outpaint expansion selections, literal 'Left', 'Right', 'Top', 'Bottom' separated by comma"),
            outpaint_distance: str = Input(
                default="0,0,0,0",
                description="Outpaint expansion distance from Left of the image"),
            cn_img1: Path = Input(
                default=None,
                description="Input image for image prompt. If all cn_img[n] are None, image prompt will not applied."),
            cn_stop1: float = Input(
                default=0.5,
                ge=0, le=1,
                description="Stop at for image prompt, None for default value"),
            cn_weight1: float = Input(
                default=0.6,
                ge=0, le=2,
                description="Weight for image prompt, None for default value"),
            cn_type1: str = Input(
                default='ImagePrompt',
                choices=['ImagePrompt', 'FaceSwap', 'PyraCanny', 'CPDS'],
                description="ControlNet type for image prompt"),
            cn_img2: Path = Input(
                default=None,
                description="Input image for image prompt. If all cn_img[n] are None, image prompt will not applied."),
            cn_stop2: float = Input(
                default=0.5,
                ge=0, le=1,
                description="Stop at for image prompt, None for default value"),
            cn_weight2: float = Input(
                default=0.6,
                ge=0, le=2,
                description="Weight for image prompt, None for default value"),
            cn_type2: str = Input(
                default='ImagePrompt',
                choices=['ImagePrompt', 'FaceSwap', 'PyraCanny', 'CPDS'],
                description="ControlNet type for image prompt"),
            cn_img3: Path = Input(
                default=None,
                description="Input image for image prompt. If all cn_img[n] are None, image prompt will not applied."),
            cn_stop3: float = Input(
                default=0.5,
                ge=0, le=1,
                description="Stop at for image prompt, None for default value"),
            cn_weight3: float = Input(
                default=0.6,
                ge=0, le=2,
                description="Weight for image prompt, None for default value"),
            cn_type3: str = Input(
                default='ImagePrompt',
                choices=['ImagePrompt', 'FaceSwap', 'PyraCanny', 'CPDS'],
                description="ControlNet type for image prompt"),
            cn_img4: Path = Input(
                default=None,
                description="Input image for image prompt. If all cn_img[n] are None, image prompt will not applied."),
            cn_stop4: float = Input(
                default=0.5,
                ge=0, le=1,
                description="Stop at for image prompt, None for default value"),
            cn_weight4: float = Input(
                default=0.6,
                ge=0, le=2,
                description="Weight for image prompt, None for default value"),
            cn_type4: str = Input(
                default='ImagePrompt',
                choices=['ImagePrompt', 'FaceSwap', 'PyraCanny', 'CPDS'],
                description="ControlNet type for image prompt"),
            black_out_nsfw: bool = Input(
                default=False,
                description="Black out NSFW image"),
            adm_scaler_positive: float = Input(
                default=1.5,
                ge=0.0, le=3.0,
                description="Adaptive Mask scaler for positive prompt"),
            adm_scaler_negative: float = Input(
                default=0.8,
                ge=0.0, le=3.0,
                description="Adaptive Mask scaler for negative prompt"),
            adm_scaler_end: float = Input(
                default=0.3,
                ge=0.0, le=1.0,
                description="Adaptive Mask scaler end at step"),
            adaptive_cfg: float = Input(
                default=default_cfg_tsnr,
                ge=0.0, le=30.0,
                description="Adaptive Control Net scaler"),
            clip_skip: int = Input(
                default=default_clip_skip,
                ge=0, le=12,
                description="Skip how many CLIP tokens"),
            sampler_name: str = Input(
                default=default_sampler,
                choices=flags.sampler_list,
                description="Sampler name"),
            scheduler_name: str = Input(
                default=default_scheduler,
                choices=flags.scheduler_list,
                description="Scheduler name"),
            vae_name: str = Input(
                default=default_vae,
                description="VAE name"),
            overwrite_step: int = Input(
                default=default_overwrite_step,
                description="Overwrite step"),
            overwrite_switch: int = Input(
                default=default_overwrite_switch,
                description="Overwrite switch"),
            overwrite_width: int = Input(
                default=-1,
                ge=-1, le=2048,
                description="Overwrite width"),
            overwrite_height: int = Input(
                default=-1,
                ge=-1, le=2048,
                description="Overwrite height"),
            overwrite_vary_strength: float = Input(
                default=-1,
                ge=-1, le=1.0,
                description="Overwrite vary strength"),
            overwrite_upscale_strength: float = Input(
                default=-1,
                ge=-1, le=1.0,
                description="Overwrite upscale strength"),
            mixing_image_prompt_and_vary_upscale: bool = Input(
                default=False,
                description="Mixing image prompt and vary upscale"),
            mixing_image_prompt_and_inpaint: bool = Input(
                default=False,
                description="Mixing image prompt and inpaint"),
            canny_low_threshold: int = Input(
                default=64,
                ge=1, le=255,
                description="Canny low threshold"),
            canny_high_threshold: int = Input(
                default=128,
                ge=1, le=255,
                description="Canny high threshold"),
            controlnet_softness: float = Input(
                default=0.25,
                ge=0.0, le=1.0,
                description="ControlNet softness"),
            inpaint_strength: float = Input(
                default=1.0,
                ge=0.0, le=1.0,
                description="Inpaint denoising strength"),
            inpaint_engine_version: str = Input(
                default=default_inpaint_engine_version,
                choices=flags.inpaint_engine_versions,
                description="Inpaint engine version"),
            save_metadata_to_images: bool = Input(
                default=True,
                description="Save metadata to images"),
            metadata_scheme: str = Input(
                default='fooocus',
                choices=['fooocus', 'a1111'],
                description="Metadata scheme"),
    ) -> Output:
        request_params = CommonRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            style_selections=style_selections.split(','),
            performance_selection=performance_selection,
            aspect_ratios_selection=aspect_ratios_selection,
            image_number=image_number,
            image_seed=image_seed,
            use_default_loras=use_default_loras,
            sharpness=sharpness,
            guidance_scale=guidance_scale,
            refiner_switch=refiner_switch,
            uov_input_image=upload_to_base64(uov_input_image),
            uov_method=uov_method,
            upscale_multiple=upscale_multiple,
            inpaint_additional_prompt=inpaint_additional_prompt,
            inpaint_input_image=upload_to_base64(inpaint_input_image),
            inpaint_input_mask=upload_to_base64(inpaint_input_mask),
            outpaint_selections=outpaint_selections.split(',') if outpaint_selections != '' else [],
            outpaint_distance=[int(i) for i in outpaint_distance.split(',')],
            black_out_nsfw=black_out_nsfw,
            adm_scaler_positive=adm_scaler_positive,
            adm_scaler_negative=adm_scaler_negative,
            adm_scaler_end=adm_scaler_end,
            adaptive_cfg=adaptive_cfg,
            clip_skip=clip_skip,
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            vae_name=vae_name,
            overwrite_step=overwrite_step,
            overwrite_switch=overwrite_switch,
            overwrite_width=overwrite_width,
            overwrite_height=overwrite_height,
            overwrite_vary_strength=overwrite_vary_strength,
            overwrite_upscale_strength=overwrite_upscale_strength,
            controlnet_image=[
                ImagePrompt(cn_img=upload_to_base64(cn_img1), cn_stop=cn_stop1, cn_weight=cn_weight1, cn_type=cn_type1),
                ImagePrompt(cn_img=upload_to_base64(cn_img2), cn_stop=cn_stop2, cn_weight=cn_weight2, cn_type=cn_type2),
                ImagePrompt(cn_img=upload_to_base64(cn_img3), cn_stop=cn_stop3, cn_weight=cn_weight3, cn_type=cn_type3),
                ImagePrompt(cn_img=upload_to_base64(cn_img4), cn_stop=cn_stop4, cn_weight=cn_weight4, cn_type=cn_type4)
            ],
            mixing_image_prompt_and_vary_upscale=mixing_image_prompt_and_vary_upscale,
            mixing_image_prompt_and_inpaint=mixing_image_prompt_and_inpaint,
            canny_low_threshold=canny_low_threshold,
            canny_high_threshold=canny_high_threshold,
            controlnet_softness=controlnet_softness,
            inpaint_strength=inpaint_strength,
            inpaint_engine_version=inpaint_engine_version,
            save_metadata_to_images=save_metadata_to_images,
            metadata_scheme=metadata_scheme
        )

        request_params.loras = lora_parser(request_params.loras)
        request_params.controlnet_image=control_net_parser(request_params.controlnet_image)
        request_params.aspect_ratios_selection=request_params.aspect_ratios_selection.replace("*", "×")
        params = params_to_params(request_params)

        task = AsyncTask(task_id=uuid.uuid4().hex, args=params)
        async_tasks.append(task)
        while True:
            time.sleep(0.2)
            if len(task.yields) > 0:
                flag, _ = task.yields.pop(0)
                if flag == 'preview':
                    if len(task.yields) > 0:
                        if task.yields[0][0] == 'preview':
                            continue
                if flag == 'finish':
                    results = task.results
                    break
        return Output(
            paths=[Path(r) for r in results]
        )
