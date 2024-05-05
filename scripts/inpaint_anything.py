import gc
import math
import os
import platform
from typing import Tuple, List, Union
import sys

# 获取当前脚本所在的目录
current_path = os.path.dirname(os.path.realpath(__file__))
print("当前路径:", current_path)

# 目录添加到 Python 路径中
sys.path.append(current_path)

# sys.path.append("../scripts")

import PIL
from fastapi import FastAPI, Body
from insightface.app import FaceAnalysis
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import LCMScheduler
from model_util import load_checkpoint_model_xl, load_diffusers_model_xl, create_noise_scheduler, load_models_xl
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline

if platform.system() == "Darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import random
import re
import traceback
from modules.api import api
import cv2
import gradio as gr
import numpy as np
import torch
import json
import pickle
import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers import (DDIMScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       KDPM2AncestralDiscreteScheduler, KDPM2DiscreteScheduler,
                       StableDiffusionInpaintPipeline, UNet2DConditionModel, SchedulerMixin)
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler, SDSampler
from modules import devices, script_callbacks, shared
from modules.processing import create_infotext, process_images
from modules.sd_models import get_closet_checkpoint_match
from modules.sd_samplers import samplers_for_img2img
from PIL import Image, ImageFilter, ImageOps
from PIL.PngImagePlugin import PngInfo
from torch.hub import download_url_to_file
from torchvision import transforms
from style_template import styles

import inpalib
from ia_check_versions import ia_check_versions
from ia_config import (IAConfig, get_ia_config_index, get_webui_setting, set_ia_config,
                       setup_ia_config_ini)
from ia_file_manager import IAFileManager, download_model_from_hf, ia_file_manager
from ia_logging import draw_text_image, ia_logging
from ia_threading import (async_post_reload_model_weights, await_backup_reload_ckpt_info,
                          await_pre_reload_model_weights, clear_cache_decorator,
                          offload_reload_decorator)
from ia_ui_items import (get_cleaner_model_ids, get_inp_model_ids, get_inp_webui_model_ids,
                         get_padding_mode_names, get_sam_model_ids, get_sampler_names)
from ia_webui_controlnet import (backup_alwayson_scripts, clear_controlnet_cache,
                                 disable_all_alwayson_scripts, disable_alwayson_scripts_wo_cn,
                                 find_controlnet, get_controlnet_args_to, get_max_args_to,
                                 get_sd_img2img_processing, restore_alwayson_scripts)
import random
import string
from pathlib import Path
# 字符集：包含所有大写和小写字母以及数字
characters = string.ascii_letters + string.digits

PIC_OUT_PATH=Path("E:/opt/upFiles")

STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "festive"

# r = redis.Redis('127.0.0.1', port=6379, db=0)
expire_time = 24 * 60 * 60  # 24小时的秒数

from huggingface_hub import hf_hub_download

SDXL_TEXT_ENCODER_TYPE = Union[CLIPTextModel, CLIPTextModelWithProjection]
# app_face = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# app_face.prepare(ctx_id=0, det_size=(640, 640))

## 定义全局变量，用于缓存已加载的模型
pipe = None
app_face = None
# 定义模型路径和其他全局变量
base_model = f'./models/checkpoints/YamerMIX_v8'
face_adapter = f'./models/checkpoints/ip-adapter.bin'
controlnet_path = f'./models/checkpoints/ControlNetModel'
lcm_lora_path = "./models/checkpoints/pytorch_lora_weights.safetensors"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16


# 定义缓存模型的路径
cached_model_path = Path("cached_model.pth")

# 延迟加载模型，并缓存加载的结果
def load_model():
    global pipe, app_face
    if app_face is None:
        # 加载 app_face 模型的代码
        app_face = FaceAnalysis(name='antelopev2', root='./',
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app_face.prepare(ctx_id=0, det_size=(640, 640))
    if pipe is None:
        if cached_model_path.exists():
            # 从缓存中加载模型
            pipe = torch.load(cached_model_path)
        else:
            # 加载模型的代码
            controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)
            pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
                base_model,
                controlnet=controlnet,
                torch_dtype=dtype,
                safety_checker=None,
                feature_extractor=None,
            ).to(device)
            pipe.load_ip_adapter_instantid(face_adapter)
            pipe.load_lora_weights(lcm_lora_path)
            pipe.fuse_lora()
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            # 将加载的模型缓存起来
            torch.save(pipe, cached_model_path)

# # Path to InstantID models
# face_adapter = f'./models/checkpoints/ip-adapter.bin'
# controlnet_path = f'./models/checkpoints/ControlNetModel'
#
# # device = torch.device(torch.cuda.current_device())
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dtype = torch.float16
#
# # base_model = 'wangqixun/YamerMIX_v8'
# base_model= f'./models/checkpoints/YamerMIX_v8'
# # Load pipeline
# controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)
#
# pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
#             base_model,
#             controlnet=controlnet,
#             torch_dtype=dtype,
#             safety_checker=None,
#             feature_extractor=None,
#         ).to(device)
#
# # pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config)
# pipe.load_ip_adapter_instantid(face_adapter)
#
# lcm_lora_path = "./models/checkpoints/pytorch_lora_weights.safetensors"
#
# pipe.load_lora_weights(lcm_lora_path)
# pipe.fuse_lora()
# pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
# pipe.enable_model_cpu_offload()
def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + ' ' + negative


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def resize_img(input_image, max_side=1280, min_side=1024, size=None,
               pad_to_max_side=False, mode=PIL.Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y + h_resize_new, offset_x:offset_x + w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

@clear_cache_decorator
def download_model(sam_model_id):
    """Download SAM model.

    Args:
        sam_model_id (str): SAM model id

    Returns:
        str: download status
    """
    if "_hq_" in sam_model_id:
        url_sam = "https://huggingface.co/Uminosachi/sam-hq/resolve/main/" + sam_model_id
    elif "FastSAM" in sam_model_id:
        url_sam = "https://huggingface.co/Uminosachi/FastSAM/resolve/main/" + sam_model_id
    elif "mobile_sam" in sam_model_id:
        url_sam = "https://huggingface.co/Uminosachi/MobileSAM/resolve/main/" + sam_model_id
    else:
        # url_sam_vit_h_4b8939 = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        url_sam = "https://dl.fbaipublicfiles.com/segment_anything/" + sam_model_id

    sam_checkpoint = os.path.join(ia_file_manager.models_dir, sam_model_id)
    if not os.path.isfile(sam_checkpoint):
        try:
            download_url_to_file(url_sam, sam_checkpoint)
        except Exception as e:
            ia_logging.error(str(e))
            return str(e)

        return IAFileManager.DOWNLOAD_COMPLETE
    else:
        return "Model already exists"


sam_dict = dict(sam_masks=None, mask_image=None, cnet=None, orig_image=None, pad_mask=None)


def draw_kps(image_pil, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0,
                                   360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

def save_mask_image(mask_image, save_mask_chk=False):
    """Save mask image.

    Args:
        mask_image (np.ndarray): mask image
        save_mask_chk (bool, optional): If True, save mask image. Defaults to False.

    Returns:
        None
    """
    if save_mask_chk:
        save_name = "_".join([ia_file_manager.savename_prefix, "created_mask"]) + ".png"
        save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
        Image.fromarray(mask_image).save(save_name)


@clear_cache_decorator
def input_image_upload(input_image, sam_image, sel_mask):
    global sam_dict
    sam_dict["orig_image"] = input_image
    sam_dict["pad_mask"] = None

    if (sam_dict["mask_image"] is None or not isinstance(sam_dict["mask_image"], np.ndarray) or
            sam_dict["mask_image"].shape != input_image.shape):
        sam_dict["mask_image"] = np.zeros_like(input_image, dtype=np.uint8)

    ret_sel_image = cv2.addWeighted(input_image, 0.5, sam_dict["mask_image"], 0.5, 0)

    if sam_image is None or not isinstance(sam_image, dict) or "image" not in sam_image:
        sam_dict["sam_masks"] = None
        ret_sam_image = np.zeros_like(input_image, dtype=np.uint8)
    elif sam_image["image"].shape == input_image.shape:
        ret_sam_image = gr.update()
    else:
        sam_dict["sam_masks"] = None
        ret_sam_image = gr.update(value=np.zeros_like(input_image, dtype=np.uint8))

    if sel_mask is None or not isinstance(sel_mask, dict) or "image" not in sel_mask:
        ret_sel_mask = ret_sel_image
    elif sel_mask["image"].shape == ret_sel_image.shape and np.all(sel_mask["image"] == ret_sel_image):
        ret_sel_mask = gr.update()
    else:
        ret_sel_mask = gr.update(value=ret_sel_image)

    return ret_sam_image, ret_sel_mask, gr.update(interactive=True)


@clear_cache_decorator
def run_padding(input_image, pad_scale_width, pad_scale_height, pad_lr_barance, pad_tb_barance, padding_mode="edge"):
    global sam_dict
    if input_image is None or sam_dict["orig_image"] is None:
        sam_dict["orig_image"] = None
        sam_dict["pad_mask"] = None
        return None, "Input image not found"

    orig_image = sam_dict["orig_image"]

    height, width = orig_image.shape[:2]
    pad_width, pad_height = (int(width * pad_scale_width), int(height * pad_scale_height))
    ia_logging.info(f"resize by padding: ({height}, {width}) -> ({pad_height}, {pad_width})")

    pad_size_w, pad_size_h = (pad_width - width, pad_height - height)
    pad_size_l = int(pad_size_w * pad_lr_barance)
    pad_size_r = pad_size_w - pad_size_l
    pad_size_t = int(pad_size_h * pad_tb_barance)
    pad_size_b = pad_size_h - pad_size_t

    pad_width = [(pad_size_t, pad_size_b), (pad_size_l, pad_size_r), (0, 0)]
    if padding_mode == "constant":
        fill_value = get_webui_setting("inpaint_anything_padding_fill", 127)
        pad_image = np.pad(orig_image, pad_width=pad_width, mode=padding_mode, constant_values=fill_value)
    else:
        pad_image = np.pad(orig_image, pad_width=pad_width, mode=padding_mode)

    mask_pad_width = [(pad_size_t, pad_size_b), (pad_size_l, pad_size_r)]
    pad_mask = np.zeros((height, width), dtype=np.uint8)
    pad_mask = np.pad(pad_mask, pad_width=mask_pad_width, mode="constant", constant_values=255)
    sam_dict["pad_mask"] = dict(segmentation=pad_mask.astype(bool))

    return pad_image, "Padding done"


@offload_reload_decorator
@clear_cache_decorator
def run_sam(input_image, sam_model_id, sam_image, anime_style_chk=False):
    global sam_dict
    if not inpalib.sam_file_exists(sam_model_id):
        ret_sam_image = None if sam_image is None else gr.update()
        return ret_sam_image, f"{sam_model_id} not found, please download"

    if input_image is None:
        ret_sam_image = None if sam_image is None else gr.update()
        return ret_sam_image, "Input image not found"

    set_ia_config(IAConfig.KEYS.SAM_MODEL_ID, sam_model_id, IAConfig.SECTIONS.USER)

    if sam_dict["sam_masks"] is not None:
        sam_dict["sam_masks"] = None
        gc.collect()

    ia_logging.info(f"input_image: {input_image.shape} {input_image.dtype}")

    try:
        sam_masks = inpalib.generate_sam_masks(input_image, sam_model_id, anime_style_chk)
        sam_masks = inpalib.sort_masks_by_area(sam_masks)
        sam_masks = inpalib.insert_mask_to_sam_masks(sam_masks, sam_dict["pad_mask"])

        seg_image = inpalib.create_seg_color_image(input_image, sam_masks)

        sam_dict["sam_masks"] = sam_masks

    except Exception as e:
        print(traceback.format_exc())
        ia_logging.error(str(e))
        ret_sam_image = None if sam_image is None else gr.update()
        return ret_sam_image, "Segment Anything failed"

    if sam_image is None:
        return seg_image, "Segment Anything complete"
    else:
        if sam_image["image"].shape == seg_image.shape and np.all(sam_image["image"] == seg_image):
            return gr.update(), "Segment Anything complete"
        else:
            return gr.update(value=seg_image), "Segment Anything complete"


@clear_cache_decorator
def select_mask(input_image, sam_image, invert_chk, ignore_black_chk, sel_mask):
    global sam_dict
    if sam_dict["sam_masks"] is None or sam_image is None:
        ret_sel_mask = None if sel_mask is None else gr.update()
        return ret_sel_mask
    sam_masks = sam_dict["sam_masks"]

    # image = sam_image["image"]
    Image.fromarray(sam_image["mask"]).save("../../../sam_image_mask.png")
    mask = sam_image["mask"][:, :, 0:1]
    # Image.fromarray(mask).save("./sam_image_mask_mask.png")
    try:
        seg_image = inpalib.create_mask_image(mask, sam_masks, ignore_black_chk)
        if invert_chk:
            seg_image = inpalib.invert_mask(seg_image)

        sam_dict["mask_image"] = seg_image

    except Exception as e:
        print(traceback.format_exc())
        ia_logging.error(str(e))
        ret_sel_mask = None if sel_mask is None else gr.update()
        return ret_sel_mask

    if input_image is not None and input_image.shape == seg_image.shape:
        ret_image = cv2.addWeighted(input_image, 0.5, seg_image, 0.5, 0)
    else:
        ret_image = seg_image

    if sel_mask is None:
        return ret_image
    else:
        if sel_mask["image"].shape == ret_image.shape and np.all(sel_mask["image"] == ret_image):
            return gr.update()
        else:
            return gr.update(value=seg_image)


@clear_cache_decorator
def expand_mask(input_image, sel_mask, expand_iteration=1):
    global sam_dict
    if sam_dict["mask_image"] is None or sel_mask is None:
        return None

    new_sel_mask = sam_dict["mask_image"]

    expand_iteration = int(np.clip(expand_iteration, 1, 100))

    new_sel_mask = cv2.dilate(new_sel_mask, np.ones((3, 3), dtype=np.uint8), iterations=expand_iteration)

    sam_dict["mask_image"] = new_sel_mask

    if input_image is not None and input_image.shape == new_sel_mask.shape:
        ret_image = cv2.addWeighted(input_image, 0.5, new_sel_mask, 0.5, 0)
    else:
        ret_image = new_sel_mask

    if sel_mask["image"].shape == ret_image.shape and np.all(sel_mask["image"] == ret_image):
        return gr.update()
    else:
        return gr.update(value=ret_image)


@clear_cache_decorator
def apply_mask(input_image, sel_mask):
    global sam_dict
    if sam_dict["mask_image"] is None or sel_mask is None:
        return None

    sel_mask_image = sam_dict["mask_image"]
    sel_mask_mask = np.logical_not(sel_mask["mask"][:, :, 0:3].astype(bool)).astype(np.uint8)
    new_sel_mask = sel_mask_image * sel_mask_mask

    sam_dict["mask_image"] = new_sel_mask

    if input_image is not None and input_image.shape == new_sel_mask.shape:
        ret_image = cv2.addWeighted(input_image, 0.5, new_sel_mask, 0.5, 0)
    else:
        ret_image = new_sel_mask

    if sel_mask["image"].shape == ret_image.shape and np.all(sel_mask["image"] == ret_image):
        return gr.update()
    else:
        return gr.update(value=ret_image)


@clear_cache_decorator
def add_mask(input_image, sel_mask):
    global sam_dict
    if sam_dict["mask_image"] is None or sel_mask is None:
        return None

    sel_mask_image = sam_dict["mask_image"]
    sel_mask_mask = sel_mask["mask"][:, :, 0:3].astype(bool).astype(np.uint8)
    new_sel_mask = sel_mask_image + (sel_mask_mask * np.invert(sel_mask_image, dtype=np.uint8))

    sam_dict["mask_image"] = new_sel_mask

    if input_image is not None and input_image.shape == new_sel_mask.shape:
        ret_image = cv2.addWeighted(input_image, 0.5, new_sel_mask, 0.5, 0)
    else:
        ret_image = new_sel_mask

    if sel_mask["image"].shape == ret_image.shape and np.all(sel_mask["image"] == ret_image):
        return gr.update()
    else:
        return gr.update(value=ret_image)


def auto_resize_to_pil(input_image, mask_image):
    init_image = Image.fromarray(input_image).convert("RGB")
    mask_image = Image.fromarray(mask_image).convert("RGB")
    assert init_image.size == mask_image.size, "The sizes of the image and mask do not match"
    width, height = init_image.size

    new_height = (height // 8) * 8
    new_width = (width // 8) * 8
    if new_width < width or new_height < height:
        if (new_width / width) < (new_height / height):
            scale = new_height / height
        else:
            scale = new_width / width
        resize_height = int(height*scale+0.5)
        resize_width = int(width*scale+0.5)
        if height != resize_height or width != resize_width:
            ia_logging.info(f"resize: ({height}, {width}) -> ({resize_height}, {resize_width})")
            init_image = transforms.functional.resize(init_image, (resize_height, resize_width), transforms.InterpolationMode.LANCZOS)
            mask_image = transforms.functional.resize(mask_image, (resize_height, resize_width), transforms.InterpolationMode.LANCZOS)
        if resize_height != new_height or resize_width != new_width:
            ia_logging.info(f"center_crop: ({resize_height}, {resize_width}) -> ({new_height}, {new_width})")
            init_image = transforms.functional.center_crop(init_image, (new_height, new_width))
            mask_image = transforms.functional.center_crop(mask_image, (new_height, new_width))

    return init_image, mask_image


@offload_reload_decorator
@clear_cache_decorator
def run_inpaint(input_image, sel_mask, prompt, n_prompt, ddim_steps, cfg_scale, seed, inp_model_id, save_mask_chk, composite_chk,
                sampler_name="DDIM", iteration_count=1):
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        ia_logging.error("The image or mask does not exist")
        return

    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.error("The sizes of the image and mask do not match")
        return

    set_ia_config(IAConfig.KEYS.INP_MODEL_ID, inp_model_id, IAConfig.SECTIONS.USER)

    save_mask_image(mask_image, save_mask_chk)

    ia_logging.info(f"Loading model {inp_model_id}")
    config_offline_inpainting = get_webui_setting("inpaint_anything_offline_inpainting", False)
    if config_offline_inpainting:
        ia_logging.info("Run Inpainting on offline network: {}".format(str(config_offline_inpainting)))
    local_files_only = False
    local_file_status = download_model_from_hf(inp_model_id, local_files_only=True)
    if local_file_status != IAFileManager.DOWNLOAD_COMPLETE:
        if config_offline_inpainting:
            ia_logging.warning(local_file_status)
            return
    else:
        local_files_only = True
        ia_logging.info("local_files_only: {}".format(str(local_files_only)))

    if platform.system() == "Darwin" or devices.device == devices.cpu or ia_check_versions.torch_on_amd_rocm:
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16

    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(inp_model_id, torch_dtype=torch_dtype, local_files_only=local_files_only)
    except Exception as e:
        ia_logging.error(str(e))
        if not config_offline_inpainting:
            try:
                pipe = StableDiffusionInpaintPipeline.from_pretrained(inp_model_id, torch_dtype=torch_dtype, resume_download=True)
            except Exception as e:
                ia_logging.error(str(e))
                try:
                    pipe = StableDiffusionInpaintPipeline.from_pretrained(inp_model_id, torch_dtype=torch_dtype, force_download=True)
                except Exception as e:
                    ia_logging.error(str(e))
                    return
        else:
            return
    pipe.safety_checker = None

    ia_logging.info(f"Using sampler {sampler_name}")
    if sampler_name == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "DPM2 Karras":
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "DPM2 a Karras":
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        ia_logging.info("Sampler fallback to DDIM")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if platform.system() == "Darwin":
        pipe = pipe.to("mps" if ia_check_versions.torch_mps_is_available else "cpu")
        pipe.enable_attention_slicing()
        torch_generator = torch.Generator(devices.cpu)
    else:
        if ia_check_versions.diffusers_enable_cpu_offload and devices.device != devices.cpu:
            ia_logging.info("Enable model cpu offload")
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(devices.device)
        if shared.xformers_available:
            ia_logging.info("Enable xformers memory efficient attention")
            pipe.enable_xformers_memory_efficient_attention()
        else:
            ia_logging.info("Enable attention slicing")
            pipe.enable_attention_slicing()
        if "privateuseone" in str(getattr(devices.device, "type", "")):
            torch_generator = torch.Generator(devices.cpu)
        else:
            torch_generator = torch.Generator(devices.device)

    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    width, height = init_image.size

    output_list = []
    iteration_count = iteration_count if iteration_count is not None else 1
    for count in range(int(iteration_count)):
        gc.collect()
        if seed < 0 or count > 0:
            seed = random.randint(0, 2147483647)

        generator = torch_generator.manual_seed(seed)

        pipe_args_dict = {
            "prompt": prompt,
            "image": init_image,
            "width": width,
            "height": height,
            "mask_image": mask_image,
            "num_inference_steps": ddim_steps,
            "guidance_scale": cfg_scale,
            "negative_prompt": n_prompt,
            "generator": generator,
        }

        output_image = pipe(**pipe_args_dict).images[0]

        if composite_chk:
            dilate_mask_image = Image.fromarray(cv2.dilate(np.array(mask_image), np.ones((3, 3), dtype=np.uint8), iterations=4))
            output_image = Image.composite(output_image, init_image, dilate_mask_image.convert("L").filter(ImageFilter.GaussianBlur(3)))

        generation_params = {
            "Steps": ddim_steps,
            "Sampler": sampler_name,
            "CFG scale": cfg_scale,
            "Seed": seed,
            "Size": f"{width}x{height}",
            "Model": inp_model_id,
        }

        generation_params_text = ", ".join([k if k == v else f"{k}: {v}" for k, v in generation_params.items() if v is not None])
        prompt_text = prompt if prompt else ""
        negative_prompt_text = "\nNegative prompt: " + n_prompt if n_prompt else ""
        infotext = f"{prompt_text}{negative_prompt_text}\n{generation_params_text}".strip()

        metadata = PngInfo()
        metadata.add_text("parameters", infotext)

        save_name = "_".join([ia_file_manager.savename_prefix, os.path.basename(inp_model_id), str(seed)]) + ".png"
        save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
        output_image.save(save_name, pnginfo=metadata)

        output_list.append(output_image)

        yield output_list, max([1, iteration_count - (count + 1)])


@offload_reload_decorator
@clear_cache_decorator
def run_cleaner(input_image, sel_mask, cleaner_model_id, cleaner_save_mask_chk):
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        ia_logging.error("The image or mask does not exist")
        return None

    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.error("The sizes of the image and mask do not match")
        return None

    save_mask_image(mask_image, cleaner_save_mask_chk)

    ia_logging.info(f"Loading model {cleaner_model_id}")
    if platform.system() == "Darwin":
        model = ModelManager(name=cleaner_model_id, device=devices.cpu)
    else:
        model = ModelManager(name=cleaner_model_id, device=devices.device)

    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    width, height = init_image.size

    init_image = np.array(init_image)
    mask_image = np.array(mask_image.convert("L"))

    config = Config(
        ldm_steps=20,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.ORIGINAL,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=512,
        hd_strategy_resize_limit=512,
        prompt="",
        sd_steps=20,
        sd_sampler=SDSampler.ddim
    )

    output_image = model(image=init_image, mask=mask_image, config=config)
    output_image = cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(output_image)

    save_name = "_".join([ia_file_manager.savename_prefix, os.path.basename(cleaner_model_id)]) + ".png"
    save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
    output_image.save(save_name)

    del model
    return [output_image]


@clear_cache_decorator
def run_get_alpha_image(input_image, sel_mask):
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        ia_logging.error("The image or mask does not exist")
        return None, ""

    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.error("The sizes of the image and mask do not match")
        return None, ""

    alpha_image = Image.fromarray(input_image).convert("RGBA")
    mask_image = Image.fromarray(mask_image).convert("L")

    alpha_image.putalpha(mask_image)

    save_name = "_".join([ia_file_manager.savename_prefix, "rgba_image"]) + ".png"
    save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
    alpha_image.save(save_name)

    return alpha_image, f"saved: {save_name}"


@clear_cache_decorator
def run_get_mask(sel_mask):
    global sam_dict
    if sam_dict["mask_image"] is None or sel_mask is None:
        return None

    mask_image = sam_dict["mask_image"]

    save_name = "_".join([ia_file_manager.savename_prefix, "created_mask"]) + ".png"
    save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
    Image.fromarray(mask_image).save(save_name)

    return mask_image


@clear_cache_decorator
def run_cn_inpaint(input_image, sel_mask,
                   cn_prompt, cn_n_prompt, cn_sampler_id, cn_ddim_steps, cn_cfg_scale, cn_strength, cn_seed,
                   cn_module_id, cn_model_id, cn_save_mask_chk,
                   cn_low_vram_chk, cn_weight, cn_mode, cn_iteration_count=1,
                   cn_ref_module_id=None, cn_ref_image=None, cn_ref_weight=1.0, cn_ref_mode="Balanced", cn_ref_resize_mode="resize",
                   cn_ipa_or_ref=None, cn_ipa_model_id=None):
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        ia_logging.error("The image or mask does not exist")
        return

    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.error("The sizes of the image and mask do not match")
        return

    await_pre_reload_model_weights()

    if (shared.sd_model.parameterization == "v" and "sd15" in cn_model_id):
        ia_logging.error("The SDv2 model is not compatible with the ControlNet model")
        ret_image = draw_text_image(input_image, "The SD v2 model is not compatible with the ControlNet model")
        yield [ret_image], 1
        return

    if (getattr(shared.sd_model, "is_sdxl", False) and "sd15" in cn_model_id):
        ia_logging.error("The SDXL model is not compatible with the ControlNet model")
        ret_image = draw_text_image(input_image, "The SD XL model is not compatible with the ControlNet model")
        yield [ret_image], 1
        return

    cnet = sam_dict.get("cnet", None)
    if cnet is None:
        ia_logging.warning("The ControlNet extension is not loaded")
        return

    save_mask_image(mask_image, cn_save_mask_chk)

    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    width, height = init_image.size

    input_mask = None if "inpaint_only" in cn_module_id else mask_image
    p = get_sd_img2img_processing(init_image, input_mask,
                                  cn_prompt, cn_n_prompt, cn_sampler_id, cn_ddim_steps, cn_cfg_scale, cn_strength, cn_seed)

    backup_alwayson_scripts(p.scripts)
    disable_alwayson_scripts_wo_cn(cnet, p.scripts)

    cn_units = [cnet.to_processing_unit(dict(
        enabled=True,
        module=cn_module_id,
        model=cn_model_id,
        weight=cn_weight,
        image={"image": np.array(init_image), "mask": np.array(mask_image)},
        resize_mode=cnet.ResizeMode.RESIZE,
        low_vram=cn_low_vram_chk,
        processor_res=min(width, height),
        guidance_start=0.0,
        guidance_end=1.0,
        pixel_perfect=True,
        control_mode=cn_mode,
    ))]

    if cn_ref_module_id is not None and cn_ref_image is not None:
        if cn_ref_resize_mode == "tile":
            ref_height, ref_width = cn_ref_image.shape[:2]
            num_h = math.ceil(height / ref_height) if height > ref_height else 1
            num_h = num_h + 1 if (num_h % 2) == 0 else num_h
            num_w = math.ceil(width / ref_width) if width > ref_width else 1
            num_w = num_w + 1 if (num_w % 2) == 0 else num_w
            cn_ref_image = np.tile(cn_ref_image, (num_h, num_w, 1))
            cn_ref_image = transforms.functional.center_crop(Image.fromarray(cn_ref_image), (height, width))
            ia_logging.info(f"Reference image is tiled ({num_h}, {num_w}) times and cropped to ({height}, {width})")
        else:
            cn_ref_image = ImageOps.fit(Image.fromarray(cn_ref_image), (width, height), method=Image.Resampling.LANCZOS)
            ia_logging.info(f"Reference image is resized and cropped to ({height}, {width})")
        assert cn_ref_image.size == init_image.size, "The sizes of the reference image and input image do not match"

        cn_ref_model_id = None
        if cn_ipa_or_ref is not None and cn_ipa_model_id is not None:
            cn_ipa_module_ids = [cn for cn in cnet.get_modules() if "ip-adapter" in cn and "sd15" in cn]
            if len(cn_ipa_module_ids) > 0 and cn_ipa_or_ref == "IP-Adapter":
                cn_ref_module_id = cn_ipa_module_ids[0]
                cn_ref_model_id = cn_ipa_model_id

        cn_units.append(cnet.to_processing_unit(dict(
            enabled=True,
            module=cn_ref_module_id,
            model=cn_ref_model_id,
            weight=cn_ref_weight,
            image={"image": np.array(cn_ref_image), "mask": None},
            resize_mode=cnet.ResizeMode.RESIZE,
            low_vram=cn_low_vram_chk,
            processor_res=min(width, height),
            guidance_start=0.0,
            guidance_end=1.0,
            pixel_perfect=True,
            control_mode=cn_ref_mode,
            threshold_a=0.5,
        )))

    p.script_args = np.zeros(get_controlnet_args_to(cnet, p.scripts)).tolist()
    cnet.update_cn_script_in_processing(p, cn_units)

    no_hash_cn_model_id = re.sub(r"\s\[[0-9a-f]{8,10}\]", "", cn_model_id).strip()

    output_list = []
    cn_iteration_count = cn_iteration_count if cn_iteration_count is not None else 1
    for count in range(int(cn_iteration_count)):
        gc.collect()
        if cn_seed < 0 or count > 0:
            cn_seed = random.randint(0, 2147483647)

        p.init_images = [init_image]
        p.seed = cn_seed

        try:
            processed = process_images(p)
        except devices.NansException:
            ia_logging.error("A tensor with all NaNs was produced in VAE")
            ret_image = draw_text_image(
                input_image, "A tensor with all NaNs was produced in VAE")
            clear_controlnet_cache(cnet, p.scripts)
            restore_alwayson_scripts(p.scripts)
            yield [ret_image], 1
            return

        if processed is not None and len(processed.images) > 0:
            output_image = processed.images[0]

            infotext = create_infotext(p, all_prompts=p.all_prompts, all_seeds=p.all_seeds, all_subseeds=p.all_subseeds)

            metadata = PngInfo()
            metadata.add_text("parameters", infotext)

            save_name = "_".join([ia_file_manager.savename_prefix, os.path.basename(no_hash_cn_model_id), str(cn_seed)]) + ".png"
            save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
            output_image.save(save_name, pnginfo=metadata)

            output_list.append(output_image)

            yield output_list, max([1, cn_iteration_count - (count + 1)])

    clear_controlnet_cache(cnet, p.scripts)
    restore_alwayson_scripts(p.scripts)


@clear_cache_decorator
def run_webui_inpaint(input_image, sel_mask,
                      webui_prompt, webui_n_prompt, webui_sampler_id, webui_ddim_steps, webui_cfg_scale, webui_strength, webui_seed,
                      webui_model_id, webui_save_mask_chk,
                      webui_mask_blur, webui_fill_mode, webui_iteration_count=1,
                      webui_enable_refiner_chk=False, webui_refiner_checkpoint="", webui_refiner_switch_at=0.8):
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        ia_logging.error("The image or mask does not exist")
        return

    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.error("The sizes of the image and mask do not match")
        return

    if "sdxl_vae" in getattr(shared.opts, "sd_vae", ""):
        ia_logging.error("The SDXL VAE is not compatible with the inpainting model")
        ret_image = draw_text_image(
            input_image, "The SDXL VAE is not compatible with the inpainting model")
        yield [ret_image], 1
        return

    set_ia_config(IAConfig.KEYS.INP_WEBUI_MODEL_ID, webui_model_id, IAConfig.SECTIONS.USER)

    save_mask_image(mask_image, webui_save_mask_chk)

    info = get_closet_checkpoint_match(webui_model_id)
    if info is None:
        ia_logging.error(f"No model found: {webui_model_id}")
        return

    await_backup_reload_ckpt_info(info=info)

    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    width, height = init_image.size

    p = get_sd_img2img_processing(init_image, mask_image,
                                  webui_prompt, webui_n_prompt, webui_sampler_id, webui_ddim_steps, webui_cfg_scale, webui_strength, webui_seed,
                                  webui_mask_blur, webui_fill_mode)

    backup_alwayson_scripts(p.scripts)
    disable_all_alwayson_scripts(p.scripts)

    p.script_args = np.zeros(get_max_args_to(p.scripts)).tolist()

    if ia_check_versions.webui_refiner_is_available and webui_enable_refiner_chk:
        p.refiner_checkpoint = webui_refiner_checkpoint
        p.refiner_switch_at = webui_refiner_switch_at

    no_hash_webui_model_id = re.sub(r"\s\[[0-9a-f]{8,10}\]", "", webui_model_id).strip()
    no_hash_webui_model_id = os.path.splitext(no_hash_webui_model_id)[0]

    output_list = []
    webui_iteration_count = webui_iteration_count if webui_iteration_count is not None else 1
    for count in range(int(webui_iteration_count)):
        gc.collect()
        if webui_seed < 0 or count > 0:
            webui_seed = random.randint(0, 2147483647)

        p.init_images = [init_image]
        p.seed = webui_seed

        try:
            processed = process_images(p)
        except devices.NansException:
            ia_logging.error("A tensor with all NaNs was produced in VAE")
            ret_image = draw_text_image(
                input_image, "A tensor with all NaNs was produced in VAE")
            restore_alwayson_scripts(p.scripts)
            yield [ret_image], 1
            return

        if processed is not None and len(processed.images) > 0:
            output_image = processed.images[0]

            infotext = create_infotext(p, all_prompts=p.all_prompts, all_seeds=p.all_seeds, all_subseeds=p.all_subseeds)

            metadata = PngInfo()
            metadata.add_text("parameters", infotext)

            save_name = "_".join([ia_file_manager.savename_prefix, os.path.basename(no_hash_webui_model_id), str(webui_seed)]) + ".png"
            save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
            output_image.save(save_name, pnginfo=metadata)

            output_list.append(output_image)

            yield output_list, max([1, webui_iteration_count - (count + 1)])

    restore_alwayson_scripts(p.scripts)


def on_ui_tabs():
    global sam_dict

    setup_ia_config_ini()
    sampler_names = get_sampler_names()
    sam_model_ids = get_sam_model_ids()
    sam_model_index = get_ia_config_index(IAConfig.KEYS.SAM_MODEL_ID, IAConfig.SECTIONS.USER)
    inp_model_ids = get_inp_model_ids()
    inp_model_index = get_ia_config_index(IAConfig.KEYS.INP_MODEL_ID, IAConfig.SECTIONS.USER)
    cleaner_model_ids = get_cleaner_model_ids()
    padding_mode_names = get_padding_mode_names()
    sam_dict["cnet"] = find_controlnet()

    cn_enabled = False
    if sam_dict["cnet"] is not None:
        cn_module_ids = [cn for cn in sam_dict["cnet"].get_modules() if "inpaint" in cn]
        cn_module_index = cn_module_ids.index("inpaint_only") if "inpaint_only" in cn_module_ids else 0

        cn_model_ids = [cn for cn in sam_dict["cnet"].get_models() if "inpaint" in cn]
        cn_modes = [mode.value for mode in sam_dict["cnet"].ControlMode]

        if len(cn_module_ids) > 0 and len(cn_model_ids) > 0:
            cn_enabled = True

    if samplers_for_img2img is not None and len(samplers_for_img2img) > 0:
        cn_sampler_ids = [sampler.name for sampler in samplers_for_img2img]
    else:
        cn_sampler_ids = ["DDIM"]
    cn_sampler_index = cn_sampler_ids.index("DDIM") if "DDIM" in cn_sampler_ids else 0

    cn_ref_only = False
    try:
        if cn_enabled and sam_dict["cnet"].get_max_models_num() > 1:
            cn_ref_module_ids = [cn for cn in sam_dict["cnet"].get_modules() if "reference" in cn]
            if len(cn_ref_module_ids) > 0:
                cn_ref_only = True
    except AttributeError:
        pass

    cn_ip_adapter = False
    if cn_ref_only:
        cn_ipa_module_ids = [cn for cn in sam_dict["cnet"].get_modules() if "ip-adapter" in cn and "sd15" in cn]
        cn_ipa_model_ids = [cn for cn in sam_dict["cnet"].get_models() if "ip-adapter" in cn and "sd15" in cn]

        if len(cn_ipa_module_ids) > 0 and len(cn_ipa_model_ids) > 0:
            cn_ip_adapter = True

    webui_inpaint_enabled = False
    webui_model_ids = get_inp_webui_model_ids()
    if len(webui_model_ids) > 0:
        webui_inpaint_enabled = True
        webui_model_index = get_ia_config_index(IAConfig.KEYS.INP_WEBUI_MODEL_ID, IAConfig.SECTIONS.USER)

    if samplers_for_img2img is not None and len(samplers_for_img2img) > 0:
        webui_sampler_ids = [sampler.name for sampler in samplers_for_img2img]
    else:
        webui_sampler_ids = ["DDIM"]
    webui_sampler_index = webui_sampler_ids.index("DDIM") if "DDIM" in webui_sampler_ids else 0

    out_gallery_kwargs = dict(columns=2, height=520, object_fit="contain", preview=True)

    with gr.Blocks(analytics_enabled=False) as inpaint_anything_interface:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        sam_model_id = gr.Dropdown(label="Segment Anything Model ID", elem_id="sam_model_id", choices=sam_model_ids,
                                                   value=sam_model_ids[sam_model_index], show_label=True)
                    with gr.Column():
                        with gr.Row():
                            load_model_btn = gr.Button("Download model", elem_id="load_model_btn")
                        with gr.Row():
                            status_text = gr.Textbox(label="", elem_id="status_text", max_lines=1, show_label=False, interactive=False)
                with gr.Row():
                    input_image = gr.Image(label="Input image", elem_id="ia_input_image", source="upload", type="numpy", interactive=True)

                with gr.Row():
                    with gr.Accordion("Padding options", elem_id="padding_options", open=False):
                        with gr.Row():
                            with gr.Column():
                                pad_scale_width = gr.Slider(label="Scale Width", elem_id="pad_scale_width", minimum=1.0, maximum=1.5, value=1.0, step=0.01)
                            with gr.Column():
                                pad_lr_barance = gr.Slider(label="Left/Right Balance", elem_id="pad_lr_barance", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                        with gr.Row():
                            with gr.Column():
                                pad_scale_height = gr.Slider(label="Scale Height", elem_id="pad_scale_height", minimum=1.0, maximum=1.5, value=1.0, step=0.01)
                            with gr.Column():
                                pad_tb_barance = gr.Slider(label="Top/Bottom Balance", elem_id="pad_tb_barance", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                        with gr.Row():
                            with gr.Column():
                                padding_mode = gr.Dropdown(label="Padding Mode", elem_id="padding_mode", choices=padding_mode_names, value="edge")
                            with gr.Column():
                                padding_btn = gr.Button("Run Padding", elem_id="padding_btn")

                with gr.Row():
                    with gr.Column():
                        anime_style_chk = gr.Checkbox(label="Anime Style (Up Detection, Down mask Quality)", elem_id="anime_style_chk",
                                                      show_label=True, interactive=True)
                    with gr.Column():
                        sam_btn = gr.Button("Run Segment Anything", elem_id="sam_btn", variant="primary", interactive=False)

                with gr.Tab("Inpainting", elem_id="inpainting_tab"):
                    with gr.Row():
                        with gr.Column():
                            prompt = gr.Textbox(label="Inpainting Prompt", elem_id="ia_sd_prompt")
                            n_prompt = gr.Textbox(label="Negative Prompt", elem_id="ia_sd_n_prompt")
                        with gr.Column(scale=0, min_width=128):
                            gr.Markdown("Get prompt from:")
                            get_txt2img_prompt_btn = gr.Button("txt2img", elem_id="get_txt2img_prompt_btn")
                            get_img2img_prompt_btn = gr.Button("img2img", elem_id="get_img2img_prompt_btn")
                    with gr.Accordion("Advanced options", elem_id="inp_advanced_options", open=False):
                        composite_chk = gr.Checkbox(label="Mask area Only", elem_id="composite_chk", value=True, show_label=True, interactive=True)
                        with gr.Row():
                            with gr.Column():
                                sampler_name = gr.Dropdown(label="Sampler", elem_id="sampler_name", choices=sampler_names,
                                                           value=sampler_names[0], show_label=True)
                            with gr.Column():
                                ddim_steps = gr.Slider(label="Sampling Steps", elem_id="ddim_steps", minimum=1, maximum=100, value=20, step=1)
                        cfg_scale = gr.Slider(label="Guidance Scale", elem_id="cfg_scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                        seed = gr.Slider(
                            label="Seed",
                            elem_id="sd_seed",
                            minimum=-1,
                            maximum=2147483647,
                            step=1,
                            value=-1,
                        )
                    with gr.Row():
                        with gr.Column():
                            inp_model_id = gr.Dropdown(label="Inpainting Model ID", elem_id="inp_model_id",
                                                       choices=inp_model_ids, value=inp_model_ids[inp_model_index], show_label=True)
                        with gr.Column():
                            with gr.Row():
                                inpaint_btn = gr.Button("Run Inpainting", elem_id="inpaint_btn", variant="primary")
                            with gr.Row():
                                save_mask_chk = gr.Checkbox(label="Save mask", elem_id="save_mask_chk",
                                                            value=False, show_label=False, interactive=False, visible=False)
                                iteration_count = gr.Slider(label="Iterations", elem_id="iteration_count", minimum=1, maximum=10, value=1, step=1)

                    with gr.Row():
                        if ia_check_versions.gradio_version_is_old:
                            out_image = gr.Gallery(label="Inpainted image", elem_id="ia_out_image", show_label=False
                                                   ).style(**out_gallery_kwargs)
                        else:
                            out_image = gr.Gallery(label="Inpainted image", elem_id="ia_out_image", show_label=False,
                                                   **out_gallery_kwargs)

                with gr.Tab("Cleaner", elem_id="cleaner_tab"):
                    with gr.Row():
                        with gr.Column():
                            cleaner_model_id = gr.Dropdown(label="Cleaner Model ID", elem_id="cleaner_model_id",
                                                           choices=cleaner_model_ids, value=cleaner_model_ids[0], show_label=True)
                        with gr.Column():
                            with gr.Row():
                                cleaner_btn = gr.Button("Run Cleaner", elem_id="cleaner_btn", variant="primary")
                            with gr.Row():
                                cleaner_save_mask_chk = gr.Checkbox(label="Save mask", elem_id="cleaner_save_mask_chk",
                                                                    value=False, show_label=False, interactive=False, visible=False)

                    with gr.Row():
                        if ia_check_versions.gradio_version_is_old:
                            cleaner_out_image = gr.Gallery(label="Cleaned image", elem_id="ia_cleaner_out_image", show_label=False
                                                           ).style(**out_gallery_kwargs)
                        else:
                            cleaner_out_image = gr.Gallery(label="Cleaned image", elem_id="ia_cleaner_out_image", show_label=False,
                                                           **out_gallery_kwargs)

                if webui_inpaint_enabled:
                    with gr.Tab("Inpainting webui", elem_id="webui_inpainting_tab"):
                        with gr.Row():
                            with gr.Column():
                                webui_prompt = gr.Textbox(label="Inpainting Prompt", elem_id="ia_webui_sd_prompt")
                                webui_n_prompt = gr.Textbox(label="Negative Prompt", elem_id="ia_webui_sd_n_prompt")
                            with gr.Column(scale=0, min_width=128):
                                gr.Markdown("Get prompt from:")
                                webui_get_txt2img_prompt_btn = gr.Button("txt2img", elem_id="webui_get_txt2img_prompt_btn")
                                webui_get_img2img_prompt_btn = gr.Button("img2img", elem_id="webui_get_img2img_prompt_btn")
                        with gr.Accordion("Advanced options", elem_id="webui_advanced_options", open=False):
                            webui_mask_blur = gr.Slider(label="Mask blur", minimum=0, maximum=64, step=1, value=4, elem_id="webui_mask_blur")
                            webui_fill_mode = gr.Radio(label="Masked content", elem_id="webui_fill_mode",
                                                       choices=["fill", "original", "latent noise", "latent nothing"], value="original", type="index")
                            with gr.Row():
                                with gr.Column():
                                    webui_sampler_id = gr.Dropdown(label="Sampling method webui", elem_id="webui_sampler_id",
                                                                   choices=webui_sampler_ids, value=webui_sampler_ids[webui_sampler_index], show_label=True)
                                with gr.Column():
                                    webui_ddim_steps = gr.Slider(label="Sampling steps webui", elem_id="webui_ddim_steps",
                                                                 minimum=1, maximum=150, value=30, step=1)
                            webui_cfg_scale = gr.Slider(label="Guidance scale webui", elem_id="webui_cfg_scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                            webui_strength = gr.Slider(label="Denoising strength webui", elem_id="webui_strength",
                                                       minimum=0.0, maximum=1.0, value=0.75, step=0.01)
                            webui_seed = gr.Slider(
                                label="Seed",
                                elem_id="webui_sd_seed",
                                minimum=-1,
                                maximum=2147483647,
                                step=1,
                                value=-1,
                            )
                        if ia_check_versions.webui_refiner_is_available:
                            with gr.Accordion("Refiner options", elem_id="webui_refiner_options", open=False):
                                with gr.Row():
                                    webui_enable_refiner_chk = gr.Checkbox(label="Enable Refiner", elem_id="webui_enable_refiner_chk",
                                                                           value=False, show_label=True, interactive=True)
                                with gr.Row():
                                    webui_refiner_checkpoint = gr.Dropdown(label="Refiner Model ID", elem_id="webui_refiner_checkpoint",
                                                                           choices=shared.list_checkpoint_tiles(), value="")
                                    webui_refiner_switch_at = gr.Slider(value=0.8, label="Switch at", minimum=0.01, maximum=1.0, step=0.01,
                                                                        elem_id="webui_refiner_switch_at")

                        with gr.Row():
                            with gr.Column():
                                webui_model_id = gr.Dropdown(label="Inpainting Model ID webui", elem_id="webui_model_id",
                                                             choices=webui_model_ids, value=webui_model_ids[webui_model_index], show_label=True)
                            with gr.Column():
                                with gr.Row():
                                    webui_inpaint_btn = gr.Button("Run Inpainting", elem_id="webui_inpaint_btn", variant="primary")
                                with gr.Row():
                                    webui_save_mask_chk = gr.Checkbox(label="Save mask", elem_id="webui_save_mask_chk",
                                                                      value=False, show_label=False, interactive=False, visible=False)
                                    webui_iteration_count = gr.Slider(label="Iterations", elem_id="webui_iteration_count",
                                                                      minimum=1, maximum=10, value=1, step=1)

                        with gr.Row():
                            if ia_check_versions.gradio_version_is_old:
                                webui_out_image = gr.Gallery(label="Inpainted image", elem_id="ia_webui_out_image", show_label=False
                                                             ).style(**out_gallery_kwargs)
                            else:
                                webui_out_image = gr.Gallery(label="Inpainted image", elem_id="ia_webui_out_image", show_label=False,
                                                             **out_gallery_kwargs)

                with gr.Tab("ControlNet Inpaint", elem_id="cn_inpaint_tab"):
                    if cn_enabled:
                        with gr.Row():
                            with gr.Column():
                                cn_prompt = gr.Textbox(label="Inpainting Prompt", elem_id="ia_cn_sd_prompt")
                                cn_n_prompt = gr.Textbox(label="Negative Prompt", elem_id="ia_cn_sd_n_prompt")
                            with gr.Column(scale=0, min_width=128):
                                gr.Markdown("Get prompt from:")
                                cn_get_txt2img_prompt_btn = gr.Button("txt2img", elem_id="cn_get_txt2img_prompt_btn")
                                cn_get_img2img_prompt_btn = gr.Button("img2img", elem_id="cn_get_img2img_prompt_btn")
                        with gr.Accordion("Advanced options", elem_id="cn_advanced_options", open=False):
                            with gr.Row():
                                with gr.Column():
                                    cn_sampler_id = gr.Dropdown(label="Sampling method", elem_id="cn_sampler_id",
                                                                choices=cn_sampler_ids, value=cn_sampler_ids[cn_sampler_index], show_label=True)
                                with gr.Column():
                                    cn_ddim_steps = gr.Slider(label="Sampling steps", elem_id="cn_ddim_steps", minimum=1, maximum=150, value=30, step=1)
                            cn_cfg_scale = gr.Slider(label="Guidance scale", elem_id="cn_cfg_scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                            cn_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Denoising strength", value=0.75, elem_id="cn_strength")
                            cn_seed = gr.Slider(
                                label="Seed",
                                elem_id="cn_sd_seed",
                                minimum=-1,
                                maximum=2147483647,
                                step=1,
                                value=-1,
                            )
                        with gr.Accordion("ControlNet options", elem_id="cn_cn_options", open=False):
                            with gr.Row():
                                with gr.Column():
                                    cn_low_vram_chk = gr.Checkbox(label="Low VRAM", elem_id="cn_low_vram_chk", value=True, show_label=True, interactive=True)
                                    cn_weight = gr.Slider(label="Control Weight", elem_id="cn_weight", minimum=0.0, maximum=2.0, value=1.0, step=0.05)
                                with gr.Column():
                                    cn_mode = gr.Dropdown(label="Control Mode", elem_id="cn_mode", choices=cn_modes, value=cn_modes[-1], show_label=True)

                            if cn_ref_only:
                                with gr.Row():
                                    with gr.Column():
                                        cn_md_text = "Reference Control (enabled with image below)"
                                        if not cn_ip_adapter:
                                            cn_md_text = cn_md_text + ("<br><span style='color: gray;'>"
                                                                       "[IP-Adapter](https://huggingface.co/lllyasviel/sd_control_collection/tree/main) "
                                                                       "is not available. Reference-Only is used.</span>")
                                        gr.Markdown(cn_md_text)
                                        if cn_ip_adapter:
                                            cn_ipa_or_ref = gr.Radio(label="IP-Adapter or Reference-Only", elem_id="cn_ipa_or_ref",
                                                                     choices=["IP-Adapter", "Reference-Only"], value="IP-Adapter", show_label=False)
                                        cn_ref_image = gr.Image(label="Reference Image", elem_id="cn_ref_image", source="upload", type="numpy",
                                                                interactive=True)
                                    with gr.Column():
                                        cn_ref_resize_mode = gr.Radio(label="Reference Image Resize Mode", elem_id="cn_ref_resize_mode",
                                                                      choices=["resize", "tile"], value="resize", show_label=True)
                                        if cn_ip_adapter:
                                            cn_ipa_model_id = gr.Dropdown(label="IP-Adapter Model ID", elem_id="cn_ipa_model_id",
                                                                          choices=cn_ipa_model_ids, value=cn_ipa_model_ids[0], show_label=True)
                                        cn_ref_module_id = gr.Dropdown(label="Reference Type for Reference-Only", elem_id="cn_ref_module_id",
                                                                       choices=cn_ref_module_ids, value=cn_ref_module_ids[-1], show_label=True)
                                        cn_ref_weight = gr.Slider(label="Reference Control Weight", elem_id="cn_ref_weight",
                                                                  minimum=0.0, maximum=2.0, value=1.0, step=0.05)
                                        cn_ref_mode = gr.Dropdown(label="Reference Control Mode", elem_id="cn_ref_mode",
                                                                  choices=cn_modes, value=cn_modes[0], show_label=True)
                            else:
                                with gr.Row():
                                    gr.Markdown("The Multi ControlNet setting is currently set to 1.<br>"
                                                "If you wish to use the Reference-Only Control, "
                                                "please adjust the Multi ControlNet setting to 2 or more and restart the Web UI.")

                        with gr.Row():
                            with gr.Column():
                                cn_module_id = gr.Dropdown(label="ControlNet Preprocessor", elem_id="cn_module_id",
                                                           choices=cn_module_ids, value=cn_module_ids[cn_module_index], show_label=True)
                                cn_model_id = gr.Dropdown(label="ControlNet Model ID", elem_id="cn_model_id",
                                                          choices=cn_model_ids, value=cn_model_ids[0], show_label=True)
                            with gr.Column():
                                with gr.Row():
                                    cn_inpaint_btn = gr.Button("Run ControlNet Inpaint", elem_id="cn_inpaint_btn", variant="primary")
                                with gr.Row():
                                    cn_save_mask_chk = gr.Checkbox(label="Save mask", elem_id="cn_save_mask_chk",
                                                                   value=False, show_label=False, interactive=False, visible=False)
                                    cn_iteration_count = gr.Slider(label="Iterations", elem_id="cn_iteration_count",
                                                                   minimum=1, maximum=10, value=1, step=1)

                        with gr.Row():
                            if ia_check_versions.gradio_version_is_old:
                                cn_out_image = gr.Gallery(label="Inpainted image", elem_id="ia_cn_out_image", show_label=False
                                                          ).style(**out_gallery_kwargs)
                            else:
                                cn_out_image = gr.Gallery(label="Inpainted image", elem_id="ia_cn_out_image", show_label=False,
                                                          **out_gallery_kwargs)

                    else:
                        if sam_dict["cnet"] is None:
                            gr.Markdown("ControlNet extension is not available.<br>"
                                        "Requires the [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) extension.")
                        elif len(cn_module_ids) > 0:
                            cn_models_directory = os.path.join("../..", "sd-webui-controlnet", "models")
                            gr.Markdown("ControlNet inpaint model is not available.<br>"
                                        "Requires the [ControlNet-v1-1](https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main) inpaint model "
                                        f"in the {cn_models_directory} directory.")
                        else:
                            gr.Markdown("ControlNet inpaint preprocessor is not available.<br>"
                                        "The local version of [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) extension may be old.")

                with gr.Tab("Mask only", elem_id="mask_only_tab"):
                    with gr.Row():
                        with gr.Column():
                            get_alpha_image_btn = gr.Button("Get mask as alpha of image", elem_id="get_alpha_image_btn")
                        with gr.Column():
                            get_mask_btn = gr.Button("Get mask", elem_id="get_mask_btn")

                    with gr.Row():
                        with gr.Column():
                            alpha_out_image = gr.Image(label="Alpha channel image", elem_id="alpha_out_image", type="pil", image_mode="RGBA", interactive=False)
                        with gr.Column():
                            mask_out_image = gr.Image(label="Mask image", elem_id="mask_out_image", type="numpy", interactive=False)

                    with gr.Row():
                        with gr.Column():
                            get_alpha_status_text = gr.Textbox(label="", elem_id="get_alpha_status_text", max_lines=1, show_label=False, interactive=False)
                        with gr.Column():
                            mask_send_to_inpaint_btn = gr.Button("Send to img2img inpaint", elem_id="mask_send_to_inpaint_btn")

            with gr.Column():
                with gr.Row():
                    gr.Markdown("Mouse over image: Press `S` key for Fullscreen mode, `R` key to Reset zoom")
                with gr.Row():
                    if ia_check_versions.gradio_version_is_old:
                        sam_image = gr.Image(label="Segment Anything image", elem_id="ia_sam_image", type="numpy", tool="sketch", brush_radius=8,
                                             show_label=True, interactive=True).style(height=480)
                    else:
                        sam_image = gr.Image(label="Segment Anything image", elem_id="ia_sam_image", type="numpy", tool="sketch", brush_radius=8,
                                             show_label=True, interactive=True, height=480)

                with gr.Row():
                    with gr.Column():
                        select_btn = gr.Button("Create Mask", elem_id="select_btn", variant="primary")
                    with gr.Column():
                        with gr.Row():
                            invert_chk = gr.Checkbox(label="Invert mask", elem_id="invert_chk", show_label=True, interactive=True)
                            ignore_black_chk = gr.Checkbox(label="Ignore black area", elem_id="ignore_black_chk", value=True, show_label=True, interactive=True)

                with gr.Row():
                    if ia_check_versions.gradio_version_is_old:
                        sel_mask = gr.Image(label="Selected mask image", elem_id="ia_sel_mask", type="numpy", tool="sketch", brush_radius=12,
                                            show_label=False, interactive=True).style(height=480)
                    else:
                        sel_mask = gr.Image(label="Selected mask image", elem_id="ia_sel_mask", type="numpy", tool="sketch", brush_radius=12,
                                            show_label=False, interactive=True, height=480)

                with gr.Row():
                    with gr.Column():
                        expand_mask_btn = gr.Button("Expand mask region", elem_id="expand_mask_btn")
                        expand_mask_iteration_count = gr.Slider(label="Expand Mask Iterations",
                                                                elem_id="expand_mask_iteration_count", minimum=1, maximum=100, value=1, step=1)
                    with gr.Column():
                        apply_mask_btn = gr.Button("Trim mask by sketch", elem_id="apply_mask_btn")
                        add_mask_btn = gr.Button("Add mask by sketch", elem_id="add_mask_btn")

            load_model_btn.click(download_model, inputs=[sam_model_id], outputs=[status_text])
            input_image.upload(input_image_upload, inputs=[input_image, sam_image, sel_mask], outputs=[sam_image, sel_mask, sam_btn]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_initSamSelMask")
            padding_btn.click(run_padding, inputs=[input_image, pad_scale_width, pad_scale_height, pad_lr_barance, pad_tb_barance, padding_mode],
                              outputs=[input_image, status_text])
            sam_btn.click(run_sam, inputs=[input_image, sam_model_id, sam_image, anime_style_chk], outputs=[sam_image, status_text]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSamMask")
            select_btn.click(select_mask, inputs=[input_image, sam_image, invert_chk, ignore_black_chk, sel_mask], outputs=[sel_mask]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSelMask")
            expand_mask_btn.click(expand_mask, inputs=[input_image, sel_mask, expand_mask_iteration_count], outputs=[sel_mask]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSelMask")
            apply_mask_btn.click(apply_mask, inputs=[input_image, sel_mask], outputs=[sel_mask]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSelMask")
            add_mask_btn.click(add_mask, inputs=[input_image, sel_mask], outputs=[sel_mask]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSelMask")
            get_txt2img_prompt_btn.click(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_getTxt2imgPrompt")
            get_img2img_prompt_btn.click(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_getImg2imgPrompt")

            inpaint_btn.click(
                run_inpaint,
                inputs=[input_image, sel_mask, prompt, n_prompt, ddim_steps, cfg_scale, seed, inp_model_id, save_mask_chk, composite_chk,
                        sampler_name, iteration_count],
                outputs=[out_image, iteration_count])
            cleaner_btn.click(
                run_cleaner,
                inputs=[input_image, sel_mask, cleaner_model_id, cleaner_save_mask_chk],
                outputs=[cleaner_out_image])
            get_alpha_image_btn.click(
                run_get_alpha_image,
                inputs=[input_image, sel_mask],
                outputs=[alpha_out_image, get_alpha_status_text])
            get_mask_btn.click(
                run_get_mask,
                inputs=[sel_mask],
                outputs=[mask_out_image])
            mask_send_to_inpaint_btn.click(
                fn=None,
                _js="inpaintAnything_sendToInpaint",
                inputs=None,
                outputs=None)
            if cn_enabled:
                cn_get_txt2img_prompt_btn.click(
                    fn=None, inputs=None, outputs=None, _js="inpaintAnything_cnGetTxt2imgPrompt")
                cn_get_img2img_prompt_btn.click(
                    fn=None, inputs=None, outputs=None, _js="inpaintAnything_cnGetImg2imgPrompt")
            if cn_enabled:
                cn_inputs = [input_image, sel_mask,
                             cn_prompt, cn_n_prompt, cn_sampler_id, cn_ddim_steps, cn_cfg_scale, cn_strength, cn_seed,
                             cn_module_id, cn_model_id, cn_save_mask_chk,
                             cn_low_vram_chk, cn_weight, cn_mode, cn_iteration_count]
                if cn_ref_only:
                    cn_inputs.extend([cn_ref_module_id, cn_ref_image, cn_ref_weight, cn_ref_mode, cn_ref_resize_mode])
                if cn_ip_adapter:
                    cn_inputs.extend([cn_ipa_or_ref, cn_ipa_model_id])
                cn_inpaint_btn.click(
                    run_cn_inpaint,
                    inputs=cn_inputs,
                    outputs=[cn_out_image, cn_iteration_count]).then(
                    fn=async_post_reload_model_weights, inputs=None, outputs=None)
            if webui_inpaint_enabled:
                webui_get_txt2img_prompt_btn.click(
                    fn=None, inputs=None, outputs=None, _js="inpaintAnything_webuiGetTxt2imgPrompt")
                webui_get_img2img_prompt_btn.click(
                    fn=None, inputs=None, outputs=None, _js="inpaintAnything_webuiGetImg2imgPrompt")
                wi_inputs = [input_image, sel_mask,
                             webui_prompt, webui_n_prompt, webui_sampler_id, webui_ddim_steps, webui_cfg_scale, webui_strength, webui_seed,
                             webui_model_id, webui_save_mask_chk,
                             webui_mask_blur, webui_fill_mode, webui_iteration_count]
                if ia_check_versions.webui_refiner_is_available:
                    wi_inputs.extend([webui_enable_refiner_chk, webui_refiner_checkpoint, webui_refiner_switch_at])
                webui_inpaint_btn.click(
                    run_webui_inpaint,
                    inputs=wi_inputs,
                    outputs=[webui_out_image, webui_iteration_count]).then(
                    fn=async_post_reload_model_weights, inputs=None, outputs=None)

    return [(inpaint_anything_interface, "Inpaint Anything", "inpaint_anything")]


def on_ui_settings():
    section = ("inpaint_anything", "Inpaint Anything")
    shared.opts.add_option("inpaint_anything_save_folder",
                           shared.OptionInfo(
                               default="inpaint-anything",
                               label="Folder name where output images will be saved",
                               component=gr.Radio,
                               component_args={"choices": ["inpaint-anything", "img2img-images (img2img output setting of web UI)"]},
                               section=section))
    shared.opts.add_option("inpaint_anything_sam_oncpu",
                           shared.OptionInfo(
                               default=False,
                               label="Run Segment Anything on CPU",
                               component=gr.Checkbox,
                               component_args={"interactive": True},
                               section=section))
    shared.opts.add_option("inpaint_anything_offline_inpainting",
                           shared.OptionInfo(
                               default=False,
                               label="Run Inpainting on offline network (Models not auto-downloaded)",
                               component=gr.Checkbox,
                               component_args={"interactive": True},
                               section=section))
    shared.opts.add_option("inpaint_anything_padding_fill",
                           shared.OptionInfo(
                               default=127,
                               label="Fill value used when Padding is set to constant",
                               component=gr.Slider,
                               component_args={"minimum": 0, "maximum": 255, "step": 1},
                               section=section))
    shared.opts.add_option("inpain_anything_sam_models_dir",
                           shared.OptionInfo(
                                default="",
                                label="Segment Anything Models Directory; If empty, defaults to [Inpaint Anything extension folder]/models",
                                component=gr.Textbox,
                                component_args={"interactive": True},
                                section=section))
def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"目录 '{directory_path}' 创建成功")
        except OSError as e:
            print(f"创建目录 '{directory_path}' 失败：{e}")
    else:
        print(f"目录 '{directory_path}' 已存在")

def mount_inpaint_anything(_: gr.Blocks, app: FastAPI):

    @app.post("/inpaint_anything/run_sam")
    async def run_sam_outer( input_image: str = Body("", title='input image'),sam_id: str = Body("sam_vit_l_0b3195.pth", title='inpaint model'),taskId: str = Body("", title='taskId')):
        global samm_dict

        input_image = api.decode_base64_to_image(input_image)
        
        # input_image.save("./input.png")
        input_image = np.array(input_image)
        print(f"input_image: {input_image.shape} {input_image.dtype}")

        input_image = input_image[:, :, :3]
        print(f"new input_image: {input_image.shape} {input_image.dtype}")

        sam_masks = inpalib.generate_sam_masks(input_image, sam_id, anime_style_chk=False)
        sam_masks = inpalib.sort_masks_by_area(sam_masks)
        sam_masks = inpalib.insert_mask_to_sam_masks(sam_masks, sam_dict["pad_mask"])

        # 生成一个5位的随机字符串
        random_string = ''.join(random.choice(characters) for _ in range(5))
        # sam_dict["sam_masks"+random_string] = sam_masks
        serialized_object = pickle.dumps(sam_masks)
        # list_as_json = json.dumps(sam_masks)
        # r.set("sam_masks"+random_string,serialized_object,expire_time)
        # seg_color_image = inpalib.create_seg_color_image(input_image, sam_masks)
        # print(f"seg_color_image: {seg_color_image.shape} {seg_color_image.dtype}")
        # Image.fromarray(seg_color_image).save("./seg_color_image.png")
        # r.set(taskId, random_string)
        # 创建一个 StreamingResponse 返回图像
        return {"image_key": random_string}

    # @app.post("/inpaint_anything/create_mask")
    # async def run_msk_outer(input_image: str = Body("", title='input image'),image_key: str = Body("", title='inpaint model')):
    #     global sam_dict
    #     input_image = api.decode_base64_to_image(input_image)
    #
    #     # input_image.save("./seg_color_image_edited.png")
    #     input_image = np.array(input_image)
    #     print(f"create_mask input_image: {input_image.shape} {input_image.dtype}")
    #
    #     target_color = (255, 255, 255)
    #     new_color = (0, 0, 0)
    #     white_pixels = np.all(input_image == target_color, axis=-1)
    #
    #     # 将白色像素点改成黑色
    #     input_image[white_pixels] = new_color
    #
    #     # 将 numpy 数组转回图片对象
    #     new_img = Image.fromarray(input_image)
    #
    #     # 保存新图片
    #     # new_img.save("./new_img_black.png")
    #
    #     # sam_masks = sam_dict["sam_masks"+image_key]
    #     # serialized_from_redis = r.get("sam_masks" + image_key)
    #
    #     sam_masks = pickle.loads(serialized_from_redis)
    #     # 将 JSON 字符串转换为 Python 列表
    #     # sam_masks = json.loads(json_from_redis)
    #
    #     mask = input_image[:, :, 0:1]
    #     mask_image = inpalib.create_mask_image(mask, sam_masks, ignore_black_chk=True)
    #
    #     serialized_object = pickle.dumps(mask_image)
    #     # sam_dict["mask_image"+image_key] = mask_image
    #     # r.set("mask_image" + image_key,serialized_object,expire_time)
    #
    #     # Image.fromarray(mask_image).save("../../../mask_image.png")
    #
    #     # print(f"create_mask mask_image: {mask_image.shape} {mask_image.dtype}")
    #     # Image.fromarray(seg_color_image).save("./seg_color_image.png")
    #
    #     # 创建一个 StreamingResponse 返回图像
    #     return {"image": api.encode_pil_to_base64(Image.fromarray(mask_image)).decode("utf-8")}

    # @app.post("/inpaint_anything/run_inpaint")
    # def run_mask_inpaint(input_image: str = Body("", title='input image'),taskid: str = Body("", title='taskid'),image_key:str = Body("", title='image_key'), prompt:str = Body("", title='prompt'), n_prompt:str = Body("", title='n_prompt'), ddim_steps:int = Body(10, title='ddim_steps'), cfg_scale:int = Body(10, title='cfg_scale'), seed:int = Body(10, title='cfg_scale'), inp_model_id:str = Body("", title='inp_model_id'),  username: str = Body("", title='username'),save_mask_chk:bool=Body(False, title='save_mask_chk'),
    #                 composite_chk:bool=Body(False, title='composite_chk'),
    #                 sampler_name:str = Body("DDIM", title='sampler_name'), iteration_count:int = Body(1, title='iteration_count')):
    #     # global sam_dict
    #
    #     if input_image is None:
    #         ia_logging.error("The image or mask does not exist")
    #         return
    #
    #     input_image = api.decode_base64_to_image(input_image)
    #     # input_image.save("./orig.png")
    #     input_image = np.array(input_image)
    #     # 4通道,提取三通道
    #     input_image = input_image[:, :, :3]
    #
    #     # mask_image = sam_dict["mask_image"+image_key]
    #     serialized_from_redis = r.get("mask_image" + image_key)
    #
    #     # 将字节串反序列化为 Python 对象
    #     mask_image = pickle.loads(serialized_from_redis)
    #     # mask_image = np.frombuffer(mask_image, dtype=np.float32)
    #
    #     if input_image.shape != mask_image.shape:
    #         ia_logging.error("The sizes of the image and mask do not match")
    #         return
    #
    #     set_ia_config(IAConfig.KEYS.INP_MODEL_ID, inp_model_id, IAConfig.SECTIONS.USER)
    #
    #     save_mask_image(mask_image, save_mask_chk)
    #
    #     ia_logging.info(f"Loading model {inp_model_id}")
    #     config_offline_inpainting = get_webui_setting("inpaint_anything_offline_inpainting", False)
    #     if config_offline_inpainting:
    #         ia_logging.info("Run Inpainting on offline network: {}".format(str(config_offline_inpainting)))
    #     local_files_only = False
    #     local_file_status = download_model_from_hf(inp_model_id, local_files_only=True)
    #     if local_file_status != IAFileManager.DOWNLOAD_COMPLETE:
    #         if config_offline_inpainting:
    #             ia_logging.warning(local_file_status)
    #             return
    #     else:
    #         local_files_only = True
    #         ia_logging.info("local_files_only: {}".format(str(local_files_only)))
    #
    #     if platform.system() == "Darwin" or devices.device == devices.cpu or ia_check_versions.torch_on_amd_rocm:
    #         torch_dtype = torch.float32
    #     else:
    #         torch_dtype = torch.float16
    #
    #     try:
    #         pipe = StableDiffusionInpaintPipeline.from_pretrained(inp_model_id, torch_dtype=torch_dtype,
    #                                                               local_files_only=local_files_only)
    #     except Exception as e:
    #         ia_logging.error(str(e))
    #         if not config_offline_inpainting:
    #             try:
    #                 pipe = StableDiffusionInpaintPipeline.from_pretrained(inp_model_id, torch_dtype=torch_dtype,
    #                                                                       resume_download=True)
    #             except Exception as e:
    #                 ia_logging.error(str(e))
    #                 try:
    #                     pipe = StableDiffusionInpaintPipeline.from_pretrained(inp_model_id, torch_dtype=torch_dtype,
    #                                                                           force_download=True)
    #                 except Exception as e:
    #                     ia_logging.error(str(e))
    #                     return
    #         else:
    #             return
    #     pipe.safety_checker = None
    #
    #     ia_logging.info(f"Using sampler {sampler_name}")
    #     if sampler_name == "DDIM":
    #         pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    #     elif sampler_name == "Euler":
    #         pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    #     elif sampler_name == "Euler a":
    #         pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    #     elif sampler_name == "DPM2 Karras":
    #         pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
    #     elif sampler_name == "DPM2 a Karras":
    #         pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    #     else:
    #         ia_logging.info("Sampler fallback to DDIM")
    #         pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    #
    #     if platform.system() == "Darwin":
    #         pipe = pipe.to("mps" if ia_check_versions.torch_mps_is_available else "cpu")
    #         pipe.enable_attention_slicing()
    #         torch_generator = torch.Generator(devices.cpu)
    #     else:
    #         if ia_check_versions.diffusers_enable_cpu_offload and devices.device != devices.cpu:
    #             ia_logging.info("Enable model cpu offload")
    #             pipe.enable_model_cpu_offload()
    #         else:
    #             pipe = pipe.to(devices.device)
    #         if shared.xformers_available:
    #             ia_logging.info("Enable xformers memory efficient attention")
    #             pipe.enable_xformers_memory_efficient_attention()
    #         else:
    #             ia_logging.info("Enable attention slicing")
    #             pipe.enable_attention_slicing()
    #         if "privateuseone" in str(getattr(devices.device, "type", "")):
    #             torch_generator = torch.Generator(devices.cpu)
    #         else:
    #             torch_generator = torch.Generator(devices.device)
    #
    #     init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    #     width, height = init_image.size
    #
    #     output_list = []
    #     iteration_count = iteration_count if iteration_count is not None else 1
    #     # for count in range(int(iteration_count)):
    #     gc.collect()
    #
    #     seed = random.randint(0, 2147483647)
    #
    #     generator = torch_generator.manual_seed(seed)
    #
    #     pipe_args_dict = {
    #         "prompt": prompt,
    #         "image": init_image,
    #         "width": width,
    #         "height": height,
    #         "mask_image": mask_image,
    #         "num_inference_steps": ddim_steps,
    #         "guidance_scale": cfg_scale,
    #         "negative_prompt": n_prompt,
    #         "generator": generator,
    #     }
    #
    #     output_image = pipe(**pipe_args_dict).images[0]
    #
    #     if composite_chk:
    #         dilate_mask_image = Image.fromarray(
    #             cv2.dilate(np.array(mask_image), np.ones((3, 3), dtype=np.uint8), iterations=4))
    #         output_image = Image.composite(output_image, init_image,
    #                                        dilate_mask_image.convert("L").filter(ImageFilter.GaussianBlur(3)))
    #
    #     generation_params = {
    #         "Steps": ddim_steps,
    #         "Sampler": sampler_name,
    #         "CFG scale": cfg_scale,
    #         "Seed": seed,
    #         "Size": f"{width}x{height}",
    #         "Model": inp_model_id,
    #     }
    #
    #     generation_params_text = ", ".join(
    #         [k if k == v else f"{k}: {v}" for k, v in generation_params.items() if v is not None])
    #     prompt_text = prompt if prompt else ""
    #     negative_prompt_text = "\nNegative prompt: " + n_prompt if n_prompt else ""
    #     infotext = f"{prompt_text}{negative_prompt_text}\n{generation_params_text}".strip()
    #
    #     metadata = PngInfo()
    #     metadata.add_text("parameters", infotext)
    #
    #     sub_directory = username
    #
    #     # 使用 Path / 运算符拼接目录
    #
    #
    #     full_path = PIC_OUT_PATH / sub_directory / "created"
    #     create_directory_if_not_exists(full_path)
    #     full_path = full_path / (taskid + ".jpg")
    #     print(full_path)
    #
    #     output_image.save(full_path)
    #     # image.save("./ddd.png")
    #
    #     # r.set(taskid, taskid, expire_time)
    #
    #
    #     # save_name = "_".join([ia_file_manager.savename_prefix, os.path.basename(inp_model_id), str(seed)]) + ".png"
    #     # save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
    #     # output_image.save(save_name, pnginfo=metadata)
    #     #
    #     # output_list.append(output_image)
    #
    #     # yield output_list, max([1, iteration_count - (count + 1)])
    #     return full_path

    @app.post("/inpaint_anything/id_upload_pic")
    def generate_image(face_image: str = Body("", title='face_image'),
                       username: str = Body("", title='username'),
                       taskid: str = Body("", title='taskid'),
                     ):

        face_image = api.decode_base64_to_image(face_image)
        if face_image is None:
            raise gr.Error(f"Cannot find any input face image! Please upload the face image")

        # face_image = load_image(face_image)

        # 加载面部识别模型
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # image =input_folder_path

        # face_image = resize_img(face_image)
        face_image_cv2 = convert_from_image_to_cv2(face_image)

        height, width, channels = face_image_cv2.shape
        # 将图像转换为灰度图像
        gray = cv2.cvtColor(face_image_cv2, cv2.COLOR_BGR2GRAY)
        # 检测面部
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0 or len(faces)>1:
            # raise gr.Error(f"Cannot find any face in the image! Please upload another person image")
            full_path=len(faces)
            return full_path

        if len(faces) > 0:
            # 取第一个脸部位置，这里假设一张图片只有一个脸部特征
            # x、y 为人脸的像素位置，w、h 为人脸的宽度和高度。
            x, y, w, h = faces[0]
            # 确定最大正方形的位置
            # 原图片竖方向长，截取正方形长度为原图横方向长，square_size截取正方形的长度
            if height > width:
                square_size = width
                x1 = 0
                x2 = square_size
                # 原图面部靠上
                if y < square_size / 2:
                    y1 = 0
                    y2 = square_size
                # 原图面部靠下
                else:
                    y1 = int(square_size / 2)
                    y2 = height
            # 原图片是横方向长，截取正方形长度为原图竖方向长度
            else:
                square_size = height
                y1 = 0
                y2 = square_size
                # 原图面部靠右
                if x < square_size / 2:
                    x1 = 0
                    x2 = square_size
                # 原图面部靠左
                else:
                    x1 = int(square_size / 2)
                    x2 = square_size

            # 根据最大正方形位置裁剪图片并保存
            cropped_img = face_image_cv2[y1:y2, x1:x2]
            # 调整图像大小为512x512
            resized = cv2.resize(cropped_img, (square_size, square_size), interpolation=cv2.INTER_AREA)
            # output_path = os.path.join(output_folder_path, "11.jpg")
            # cv2.imwrite(output_path, resized)
            image = convert_from_cv2_to_image(resized)


            sub_directory = username

            # 使用 Path / 运算符拼接目录

            # 获取图像的宽和高
            width, height = image.size

            # 打印宽和高
            print(f"图像宽度：{width}px")
            print(f"图像高度：{height}px")

            base_path = PIC_OUT_PATH / "upload" / sub_directory
            create_directory_if_not_exists(base_path)
            full_path = base_path / (taskid + ".jpg")
            full_path_orig = base_path / (taskid + "-orig.jpg")
            print(full_path)
            print(full_path_orig)
            image.save(full_path)
            face_image.save(full_path_orig)
            # image.save("./ddd.png")


            # r.set(taskid, taskid, expire_time)
            return {"taskid": taskid}


    @app.post("/inpaint_anything/id_inpaint")
    def generate_image( face_image: str = Body("", title='face_image'), pose_image: str = Body(None, title='pose_image'),  username: str = Body("", title='username'),taskid: str = Body("", title='taskid'),prompt: str = Body("", title='prompt'),negative_prompt: str = Body("", title='negative_prompt'),  style_name: str = Body("", title='style_name'),  num_steps:int = Body(10, title='num_steps'),
                        identitynet_strength_ratio: float = Body(0.8, title='identitynet_strength_ratio'),  adapter_strength_ratio: float = Body(0.8, title='adapter_strength_ratio'),  guidance_scale:int = Body(5, title='guidance_scale')):

        # 延迟加载pipe
        load_model()

        face_image = api.decode_base64_to_image(face_image)
        if face_image is None:
            raise gr.Error(f"Cannot find any input face image! Please upload the face image")

        if prompt is None:
            prompt = "Red festive,Family reunion to celebrate,Plane design,Chinese Dragon in backgound,New Year's Day,Festival celebration,Chinese cultural theme style,soft tones,warm palettes,vibrant illustrations,color mural,Minimalism"
        # else:
        #     prompt=prompt+",Red festive,Family reunion to celebrate,Plane design,Chinese Dragon in backgound,New Year's Day,Festival celebration,Chinese cultural theme style,soft tones,warm palettes,vibrant illustrations,color mural,Minimalism"
        # apply the style template
        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

        # face_image = load_image(face_image)
        face_image = resize_img(face_image)
        face_image_cv2 = convert_from_image_to_cv2(face_image)
        height, width, _ = face_image_cv2.shape

        # Extract face features
        face_info = app_face.get(face_image_cv2)

        if len(face_info) == 0:
            raise gr.Error(f"Cannot find any face in the image! Please upload another person image")

        face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * x['bbox'][3] - x['bbox'][1])[
            -1]  # only use the maximum face
        face_emb = face_info['embedding']
        face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info['kps'])

        if pose_image is not None:
            pose_image = load_image(pose_image[0])
            pose_image = resize_img(pose_image)
            pose_image_cv2 = convert_from_image_to_cv2(pose_image)

            face_info = app_face.get(pose_image_cv2)

            if len(face_info) == 0:
                raise gr.Error(f"Cannot find any face in the reference image! Please upload another person image")

            face_info = face_info[-1]
            face_kps = draw_kps(pose_image, face_info['kps'])

            width, height = face_kps.size


        seed = -1


        generator = torch.Generator(device=device).manual_seed(seed)

        print("Start inference...")
        print(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")



        pipe.set_ip_adapter_scale(adapter_strength_ratio)

        face_emb=torch.from_numpy(face_emb).to(device)
        # face_kps = torch.from_numpy(face_kps).to(device)


        # pipe.enable_model_cpu_offload()

        num_inference_steps = 10
        guidance_scale = 0.1
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=float(identitynet_strength_ratio),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator
        ).images[0]

        # sub_directory = username
        #
        # # 使用 Path / 运算符拼接目录
        #
        # 获取图像的宽和高
        width, height = image.size

        # 打印宽和高
        print(f"图像宽度：{width}px")
        print(f"图像高度：{height}px")
        #
        # full_path = PIC_OUT_PATH / "created" /sub_directory
        # create_directory_if_not_exists(full_path)
        # full_path = full_path / (taskid+"_"+str(seed) + ".jpg")
        # print(full_path)
        #
        # image.save(full_path)
        # # image.save("./ddd.png")

        # r.set(taskid, taskid,expire_time)
        return {"image": api.encode_pil_to_base64(image).decode("utf-8")}
        # return full_path

    @app.post("/inpaint_anything/id_inpaint_douyin")
    def generate_image(pic_path: str = Body("", title='pic_path'),
                       username: str = Body("", title='username'), taskid: str = Body("", title='taskid'),
                       prompt: str = Body("", title='prompt'), negative_prompt: str = Body("", title='negative_prompt'),
                       seed: str = Body("", title='seed'),
                       num_steps: int = Body(10, title='num_steps'),
                       identitynet_strength_ratio: float = Body(0.8, title='identitynet_strength_ratio'),
                       adapter_strength_ratio: float = Body(0.8, title='adapter_strength_ratio'),
                       guidance_scale: int = Body(5, title='guidance_scale')):

        # face_image = api.decode_base64_to_image(face_image)
        # if face_image is None:
        #     raise gr.Error(f"Cannot find any input face image! Please upload the face image")
        #
        # if prompt is None:
        #     prompt = "Red festive,Family reunion to celebrate,Plane design,Chinese Dragon in backgound,New Year's Day,Festival celebration,Chinese cultural theme style,soft tones,warm palettes,vibrant illustrations,color mural,Minimalism"
        # # else:
        # #     prompt=prompt+",Red festive,Family reunion to celebrate,Plane design,Chinese Dragon in backgound,New Year's Day,Festival celebration,Chinese cultural theme style,soft tones,warm palettes,vibrant illustrations,color mural,Minimalism"
        # # apply the style template
        # prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

        filepath = os.path.join(PIC_OUT_PATH, pic_path)
        # 使用Pillow加载图片
        face_image = Image.open(filepath)

        face_image = resize_img(face_image)
        face_image_cv2 = convert_from_image_to_cv2(face_image)
        height, width, _ = face_image_cv2.shape

        # Extract face features
        face_info = app_face.get(face_image_cv2)

        if len(face_info) == 0:
            raise gr.Error(f"Cannot find any face in the image! Please upload another person image")

        face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * x['bbox'][3] - x['bbox'][1])[
            -1]  # only use the maximum face
        face_emb = face_info['embedding']
        face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info['kps'])

        # if pose_image is not None:
        #     pose_image = load_image(pose_image[0])
        #     pose_image = resize_img(pose_image)
        #     pose_image_cv2 = convert_from_image_to_cv2(pose_image)
        #
        #     face_info = app_face.get(pose_image_cv2)
        #
        #     if len(face_info) == 0:
        #         raise gr.Error(f"Cannot find any face in the reference image! Please upload another person image")
        #
        #     face_info = face_info[-1]
        #     face_kps = draw_kps(pose_image, face_info['kps'])
        #
        #     width, height = face_kps.size

        # seed = random.randint(0, 2147483647)

        generator = torch.Generator(device=device).manual_seed(int(seed))
        print(f"seed= {seed}")
        print("Start inference...")
        print(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")

        pipe.set_ip_adapter_scale(adapter_strength_ratio)

        face_emb = torch.from_numpy(face_emb).to(device)
        # face_kps = torch.from_numpy(face_kps).to(device)

        # pipe.enable_model_cpu_offload()

        num_inference_steps = 10
        guidance_scale = 0.1
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=float(identitynet_strength_ratio),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator
        ).images[0]

        sub_directory = username

        # 使用 Path / 运算符拼接目录

        # 获取图像的宽和高
        width, height = image.size

        # 打印宽和高
        print(f"图像宽度：{width}px")
        print(f"图像高度：{height}px")

        full_path = PIC_OUT_PATH / "created" / sub_directory
        create_directory_if_not_exists(full_path)
        full_path = full_path / (taskid + "_" + str(seed) + ".jpg")
        print(full_path)

        image.save(full_path)
        # image.save("./ddd.png")

        # r.set(taskid, taskid, expire_time)
        return full_path
script_callbacks.on_app_started(mount_inpaint_anything)


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)
