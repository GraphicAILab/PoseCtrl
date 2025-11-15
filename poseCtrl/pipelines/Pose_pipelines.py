import os
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from torch.cuda.amp import autocast
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from poseCtrl.models.attention_processor import AttnProcessor, CNAttnProcessor, PoseAttnProcessor
# from poseCtrl.models.pose_adaptor import VPmatrixEncoder
from poseCtrl.models.utils import get_generator
import sys
sys.path.append('/content/drive/MyDrive/PoseCtrl')
sys.path.append('/content/drive/MyDrive/PoseCtrl/poseCtrl')
from poseCtrl.models.attention_processor import AttnProcessor, PoseAttnProcessor,PoseAttnProcessorV1
from poseCtrl.models.pose_adaptor import ImageProjModel, VPProjModel,VPmatrixPointsV3
from poseCtrl.data.dataset import CustomDataset, load_base_points
from poseCtrl.models.pose_controlnet import PoseControlNetModel 
from poseCtrl.models.attention_processor import PoseAttnProcessorV4, PoseAttnProcessorV2IP, PoseAttnProcessorV7
from poseCtrl.models.pose_adaptor import VPmatrixPoints, ImageProjModel, VPmatrixPointsV1, VPProjModel, PointNetEncoder
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from typing import List

""" The infence for posectrl V3 """
class PoseControlNet:
    def __init__(self, sd_pipe, image_encoder_path, pose_ckpt, raw_base_points, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.pose_ckpt = pose_ckpt
        self.num_tokens = num_tokens
        self.raw_base_points = raw_base_points.to(torch.float16).to(self.device)
        self.pipe = sd_pipe.to(self.device)
        self.unet_copy = self.init_unet_copy()
        self.set_posectrl()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()
        self.vpmatrix_points_sd = self.init_point()
        self.load_posectrl()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model
    
    def init_point(self):
        vpmatrix_points_sd = VPmatrixPointsV3(self.raw_base_points)
        return vpmatrix_points_sd.to(self.device, dtype=torch.float16)

    def init_unet_copy(self):
        return PoseControlNetModel.from_unet(self.pipe.unet).to(self.device, dtype=torch.float16)


    def set_posectrl(self):
        unet = self.pipe.unet
        unet_copy = self.unet_copy
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = PoseAttnProcessorV2IP(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)

        attn_procs_p = {}
        for name in unet_copy.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet_copy.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet_copy.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet_copy.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs_p[name] = AttnProcessor()
            else:
                attn_procs_p[name] = PoseAttnProcessorV2Ctrl(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet_copy.set_attn_processor(attn_procs_p)

    def load_posectrl(self):
        if os.path.splitext(self.pose_ckpt)[-1] == ".safetensors":
            state_dict = {"unet_copy": {}, "atten_modules": {}, "image_proj_model": {}}
            with safe_open(self.pose_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("unet_copy."):
                        state_dict["unet_copy"][key.replace("unet_copy.", "")] = f.get_tensor(key)
                    elif key.startswith("atten_modules."):
                        state_dict["atten_modules"][key.replace("atten_modules.", "")] = f.get_tensor(key)
                    elif key.startswith("image_proj_model."):
                        state_dict["image_proj_model"][key.replace("image_proj_model.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.pose_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj_model"])
        # self.image_proj_model_point.load_state_dict(state_dict["image_proj_model_point"])
        self.unet_copy.load_state_dict(state_dict['unet_copy'])
        atten_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        atten_layers.load_state_dict(state_dict["atten_modules"])

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        """ 修改: 这个逻辑应该是通过self.VP矩阵乘self.BasePoints
            输出: image_prompt_embeds, uncond_image_prompt_embeds
            但是之后不需要和原来的text embeds拼接,因为没有text embeds,
            感觉还是有点好,这个后来再看
        """
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt", do_rescale=False).pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    @torch.inference_mode()
    def get_vpmatrix_points(self, V_matrix, P_matrix):
        image_prompt_embeds = self.vpmatrix_points_sd(V_matrix, P_matrix)
        uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, PoseAttnProcessorV1):
                attn_processor.scale = scale
    

    def custom_inference(self,
                        ip_prompt_embeds,
                        point_prompt_embeds,
                        ip_negative_prompt_embeds,
                        point_negative_prompt_embeds,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        height=512,
                        width=512,
                        seed=42):
        batch_size = point_prompt_embeds.shape[0]
        device = ip_prompt_embeds.device
        dtype = torch.float16  # use half precision for memory optimization

        # Prepare random latent input
        latents = torch.randn(
            (batch_size, self.pipe.unet.config.in_channels, height // 8, width // 8),
            generator=torch.Generator(device=device).manual_seed(seed),
            device=device,
            dtype=dtype
        )

        # Expand prompt embeddings for classifier-free guidance
        point_prompt_embeds = torch.cat([point_negative_prompt_embeds, point_prompt_embeds], dim=0).to(device, dtype)
        ip_prompt_embeds = torch.cat([ip_negative_prompt_embeds, ip_prompt_embeds], dim=0).to(device, dtype)

        # Set inference timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps

        with torch.no_grad():
            # Denoising loop
            for t in timesteps:
                # Duplicate latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2, dim=0)
                t_tensor = torch.full(
                    (latent_model_input.shape[0],),
                    t,
                    device=device,
                    dtype=dtype
                )

                # Get residual hints from auxiliary UNet
                down_block_res_samples, mid_block_res_sample = self.unet_copy(
                    latent_model_input, t_tensor, point_prompt_embeds, return_dict=False
                )

                # Predict noise using main UNet
                noise_pred = self.pipe.unet(
                    latent_model_input,
                    t_tensor,
                    encoder_hidden_states=ip_prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                # Apply classifier-free guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Update latents via scheduler
                latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

            # Decode final latents into images
            latents = latents / self.pipe.vae.config.scaling_factor  # equivalent to 1 / 0.18215
            images = self.pipe.vae.decode(latents).sample
            images = (images / 2 + 0.5).clamp(0, 1)  # normalize to [0,1] range
            images = images.cpu().permute(0, 2, 3, 1).numpy()  # convert to HWC format

        return images



    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        V_matrix=None,
        P_matrix=None,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            # num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
            num_prompts = 1
        else:
            num_prompts = clip_image_embeds.size(0)
        
        # 不需要prompt
        if prompt is None:
            prompt = "a highly detailed anime person, in front of a pure black background"
            # prompt = "girl"
        if negative_prompt is None:
            # negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, noise, cluttered background"
            negative_prompt = "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, realistic photo, huge eyes, worst face, 2girl, long fingers, disconnected limbs"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        """ 修改:这个 get_image_embeds函数输入不对"""
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        vpmatrix_points_embeds, uncon_vpmatrix_points_embeds= self.get_vpmatrix_points(V_matrix, P_matrix)
        bs_embed, seq_len, _ = vpmatrix_points_embeds.shape
        vpmatrix_points_embeds = vpmatrix_points_embeds.repeat(1, num_samples, 1)
        vpmatrix_points_embeds = vpmatrix_points_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncon_vpmatrix_points_embeds = uncon_vpmatrix_points_embeds.repeat(1, num_samples, 1)
        uncon_vpmatrix_points_embeds = uncon_vpmatrix_points_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            """ 修改: 这里到底要不要拼接,原来到底是几维的,中间维度不影响,随便怎么拼"""
            ip_prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            point_prompt_embeds = torch.cat([prompt_embeds_, vpmatrix_points_embeds], dim=1)
            ip_negative_prompt_embeds = torch.cat([negative_prompt_embeds_,uncond_image_prompt_embeds], dim=1)
            point_negative_prompt_embeds = torch.cat([negative_prompt_embeds_,uncon_vpmatrix_points_embeds], dim=1)

        image = self.custom_inference(
            ip_prompt_embeds=ip_prompt_embeds,
            point_prompt_embeds=point_prompt_embeds,
            ip_negative_prompt_embeds=ip_negative_prompt_embeds,
            point_negative_prompt_embeds=point_negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=self.pipe.unet.config.sample_size * 8,
            width=self.pipe.unet.config.sample_size * 8,
            seed=seed,
        )

        return image


class PoseControlNetV3_1:
    def __init__(self, sd_pipe, image_encoder_path, pose_ckpt, raw_base_points, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.pose_ckpt = pose_ckpt
        self.num_tokens = num_tokens
        self.raw_base_points = raw_base_points.to(torch.float16).to(self.device)
        self.pipe = sd_pipe.to(self.device)
        self.unet_copy = self.init_unet_copy()
        self.set_posectrl()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model_point = self.init_VP()
        self.image_proj_model = self.init_proj()
        self.vpmatrix_points_sd = self.init_point()
        self.load_posectrl()

    def init_VP(self):
        image_proj_model_point = VPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model_point

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model
    
    def init_point(self):
        vpmatrix_points_sd = VPmatrixPointsV1(self.raw_base_points)
        return vpmatrix_points_sd.to(self.device, dtype=torch.float16)

    def init_unet_copy(self):
        return PoseControlNetModel.from_unet(self.pipe.unet).to(self.device, dtype=torch.float16)


    def set_posectrl(self):
        unet = self.pipe.unet
        unet_copy = self.unet_copy
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = PoseAttnProcessorV2IP(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)

        attn_procs_p = {}
        for name in unet_copy.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet_copy.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet_copy.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet_copy.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs_p[name] = AttnProcessor()
            else:
                attn_procs_p[name] = PoseAttnProcessorV2Ctrl(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet_copy.set_attn_processor(attn_procs_p)

    def load_posectrl(self):
        if os.path.splitext(self.pose_ckpt)[-1] == ".safetensors":
            state_dict = {"unet_copy": {}, "atten_modules": {}, "image_proj_model": {}}
            with safe_open(self.pose_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("unet_copy."):
                        state_dict["unet_copy"][key.replace("unet_copy.", "")] = f.get_tensor(key)
                    elif key.startswith("atten_modules."):
                        state_dict["atten_modules"][key.replace("atten_modules.", "")] = f.get_tensor(key)
                    elif key.startswith("image_proj_model."):
                        state_dict["image_proj_model"][key.replace("image_proj_model.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.pose_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj_model"])
        # self.image_proj_model_point.load_state_dict(state_dict["image_proj_model_point"])
        self.unet_copy.load_state_dict(state_dict['unet_copy'])
        atten_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        atten_layers.load_state_dict(state_dict["atten_modules"])

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt", do_rescale=False).pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    @torch.inference_mode()
    def get_vpmatrix_points(self, V_matrix, P_matrix):
        base_points = self.vpmatrix_points_sd(V_matrix, P_matrix)
        inputs = self.clip_image_processor(images=base_points, return_tensors="pt", do_rescale=False).pixel_values
        point_embeds = self.image_encoder(inputs.to(self.device, dtype=torch.float16)).image_embeds

        image_prompt_embeds = self.image_proj_model_point(point_embeds, V_matrix, P_matrix)
        uncond_image_prompt_embeds = self.image_proj_model_point(torch.zeros_like(point_embeds),V_matrix, P_matrix)
        return image_prompt_embeds, uncond_image_prompt_embeds


    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, PoseAttnProcessorV1):
                attn_processor.scale = scale
    
    @torch.inference_mode()
    def custom_inference(self,
                     ip_prompt_embeds,
                     point_prompt_embeds,
                     ip_negative_prompt_embeds,
                     point_negative_prompt_embeds,
                     num_inference_steps=50,
                     guidance_scale=7.5,
                     height=512,
                     width=512,
                     seed=42):
        batch_size = point_prompt_embeds.shape[0]
        device = ip_prompt_embeds.device
        generator = torch.Generator(device=device).manual_seed(seed)

        latents = torch.randn((batch_size, self.pipe.unet.in_channels, height // 8, width // 8),
                            generator=generator, device=device, dtype=ip_prompt_embeds.dtype)
        point_prompt_embeds = torch.cat([point_negative_prompt_embeds, point_prompt_embeds], dim=0).to(device)
        ip_prompt_embeds = torch.cat([ip_negative_prompt_embeds, ip_prompt_embeds], dim=0).to(device)
        # 2.  scheduler
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps

        # 3.  denoising
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            t_tensor = torch.tensor([t], device=device, dtype=latents.dtype).expand(latent_model_input.shape[0])

            down_block_res_samples, mid_block_res_sample = self.unet_copy(
                latent_model_input, t_tensor, point_prompt_embeds, return_dict=False
            )

            # 5. predict noise
            noise_pred = self.pipe.unet(
                latent_model_input,
                t_tensor,
                ip_prompt_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]

            # 6. classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # 7. update latent
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # 8. decode latent 得到图像
        latents = 1 / 0.18215 * latents
        images = self.pipe.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)  # 归一化
        images = images.cpu().permute(0, 2, 3, 1).numpy()

        return images

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        V_matrix=None,
        P_matrix=None,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            # num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
            num_prompts = 1
        else:
            num_prompts = clip_image_embeds.size(0)
        
        # 不需要prompt
        if prompt is None:
            prompt = "a highly detailed anime figure, in front of a pure black background"
            # prompt = "girl"
        if negative_prompt is None:
            # negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, noise, cluttered background"
            negative_prompt = "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, realistic photo, huge eyes, worst face, 2girl, long fingers, disconnected limbs"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        """ 修改:这个 get_image_embeds函数输入不对"""
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        vpmatrix_points_embeds, uncon_vpmatrix_points_embeds= self.get_vpmatrix_points(V_matrix, P_matrix)
        bs_embed, seq_len, _ = vpmatrix_points_embeds.shape
        vpmatrix_points_embeds = vpmatrix_points_embeds.repeat(1, num_samples, 1)
        vpmatrix_points_embeds = vpmatrix_points_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncon_vpmatrix_points_embeds = uncon_vpmatrix_points_embeds.repeat(1, num_samples, 1)
        uncon_vpmatrix_points_embeds = uncon_vpmatrix_points_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            """ 修改: 这里到底要不要拼接,原来到底是几维的,中间维度不影响,随便怎么拼"""
            ip_prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            point_prompt_embeds = torch.cat([prompt_embeds_, vpmatrix_points_embeds], dim=1)
            ip_negative_prompt_embeds = torch.cat([negative_prompt_embeds_,uncond_image_prompt_embeds], dim=1)
            point_negative_prompt_embeds = torch.cat([negative_prompt_embeds_,uncon_vpmatrix_points_embeds], dim=1)

        image = self.custom_inference(
            ip_prompt_embeds=ip_prompt_embeds,
            point_prompt_embeds=point_prompt_embeds,
            ip_negative_prompt_embeds=ip_negative_prompt_embeds,
            point_negative_prompt_embeds=point_negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=self.pipe.unet.config.sample_size * 8,
            width=self.pipe.unet.config.sample_size * 8,
            seed=seed,
        )

        return image
    

class PoseControlNetV6:
    def __init__(self, sd_pipe, image_encoder_path, pose_ckpt, raw_base_points, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.pose_ckpt = pose_ckpt
        self.num_tokens = num_tokens
        self.raw_base_points = raw_base_points.to(torch.float16).to(self.device)
        self.pipe = sd_pipe.to(self.device)
        self.unet_copy = self.init_unet_copy()
        self.set_posectrl()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.vpmatrix_points_sd = self.init_point()
        self.image_proj_model_point = self.init_VP()
        self.load_posectrl()

    def init_VP(self):
        image_proj_model_point = VPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model_point
    
    def init_point(self):
        vpmatrix_points_sd = VPmatrixPointsV1(self.raw_base_points)
        return vpmatrix_points_sd
        

    def init_unet_copy(self):
        return PoseControlNetModel.from_unet(self.pipe.unet).to(self.device, dtype=torch.float16)


    def set_posectrl(self):
        unet_copy = self.unet_copy
        attn_procs_p = {}
        for name in unet_copy.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet_copy.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet_copy.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet_copy.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs_p[name] = AttnProcessor()
            else:
                attn_procs_p[name] = PoseAttnProcessorV4(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet_copy.set_attn_processor(attn_procs_p)

    def load_posectrl(self):
        if os.path.splitext(self.pose_ckpt)[-1] == ".safetensors":
            state_dict = {"unet_copy": {}, "atten_modules": {}, "image_proj_model_point": {}}
            with safe_open(self.pose_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("unet_copy."):
                        state_dict["unet_copy"][key.replace("unet_copy.", "")] = f.get_tensor(key)
                    elif key.startswith("atten_modules."):
                        state_dict["atten_modules"][key.replace("atten_modules.", "")] = f.get_tensor(key)
                    elif key.startswith("image_proj_model_point."):
                        state_dict["image_proj_model_point"][key.replace("image_proj_model_point.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.pose_ckpt, map_location="cpu")
        self.image_proj_model_point.load_state_dict(state_dict["image_proj_model_point"])
        self.unet_copy.load_state_dict(state_dict['unet_copy'])
        atten_layers = torch.nn.ModuleList(self.unet_copy.attn_processors.values())
        atten_layers.load_state_dict(state_dict["atten_modules"])

    @torch.inference_mode()
    def get_vpmatrix_points(self, V_matrix, P_matrix, points):
        inputs = self.clip_image_processor(images=points, return_tensors="pt",do_rescale=False).pixel_values
        point_embeds = self.image_encoder(inputs.to(self.device, dtype=torch.float16)).image_embeds

        image_prompt_embeds = self.image_proj_model_point(point_embeds, V_matrix, P_matrix)
        uncond_image_prompt_embeds = self.image_proj_model_point(torch.zeros_like(point_embeds),V_matrix, P_matrix)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, PoseAttnProcessorV1):
                attn_processor.scale = scale    

    def custom_inference(self,
                        text_prompt_embeds,
                        text_negative_prompt_embeds,
                        point_prompt_embeds,
                        point_negative_prompt_embeds,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        height=512,
                        width=512,
                        seed=42):
        batch_size = point_prompt_embeds.shape[0]
        device = point_prompt_embeds.device
        dtype = torch.float16  # use half precision for memory optimization

        # Prepare random latent input
        latents = torch.randn(
            (batch_size, self.pipe.unet.config.in_channels, height // 8, width // 8),
            generator=torch.Generator(device=device).manual_seed(seed),
            device=device,
            dtype=dtype
        )

        # Expand prompt embeddings for classifier-free guidance
        point_prompt_embeds = torch.cat([point_negative_prompt_embeds, point_prompt_embeds], dim=0).to(device, dtype)
        text_prompt_embeds = torch.cat([text_negative_prompt_embeds, text_prompt_embeds], dim=0).to(device, dtype)
        # Set inference timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps

        with torch.no_grad():
            # Denoising loop
            for t in timesteps:
                # Duplicate latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2, dim=0)
                t_tensor = torch.full(
                    (latent_model_input.shape[0],),
                    t,
                    device=device,
                    dtype=dtype
                )

                # Get residual hints from auxiliary UNet
                down_block_res_samples, mid_block_res_sample = self.unet_copy(
                    latent_model_input, t_tensor, point_prompt_embeds, return_dict=False
                )

                # Predict noise using main UNet
                noise_pred = self.pipe.unet(
                    latent_model_input,
                    t_tensor,
                    encoder_hidden_states=text_prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                # Apply classifier-free guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Update latents via scheduler
                latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

            # Decode final latents into images
            latents = latents / self.pipe.vae.config.scaling_factor  # equivalent to 1 / 0.18215
            images = self.pipe.vae.decode(latents).sample
            images = (images / 2 + 0.5).clamp(0, 1)  # normalize to [0,1] range
            images = images.cpu().permute(0, 2, 3, 1).numpy()  # convert to HWC format

        return images



    def generate(
        self,
        prompt=None,
        prompt_embeds=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        V_matrix=None,
        P_matrix=None,
        points=None,
        **kwargs,
    ):
        self.set_scale(scale)

        if prompt is not None:
            # num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
            num_prompts = 1
        else:
            num_prompts = prompt_embeds.size(0)
        
        # 不需要prompt
        if prompt is None:
            prompt = "a highly detailed anime person, in front of a pure black background"
            # prompt = "girl"
        if negative_prompt is None:
            # negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, noise, cluttered background"
            negative_prompt = "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, realistic photo, huge eyes, worst face, 2girl, long fingers, disconnected limbs"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        vpmatrix_points_embeds, uncon_vpmatrix_points_embeds= self.get_vpmatrix_points(V_matrix, P_matrix, points)
        bs_embed, seq_len, _ = vpmatrix_points_embeds.shape
        vpmatrix_points_embeds = vpmatrix_points_embeds.repeat(1, num_samples, 1)
        vpmatrix_points_embeds = vpmatrix_points_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncon_vpmatrix_points_embeds = uncon_vpmatrix_points_embeds.repeat(1, num_samples, 1)
        uncon_vpmatrix_points_embeds = uncon_vpmatrix_points_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            """ 修改: 这里到底要不要拼接,原来到底是几维的,中间维度不影响,随便怎么拼"""
            point_prompt_embeds = torch.cat([prompt_embeds_, vpmatrix_points_embeds], dim=1)
            point_negative_prompt_embeds = torch.cat([negative_prompt_embeds_,uncon_vpmatrix_points_embeds], dim=1)

        image = self.custom_inference(
            text_prompt_embeds=prompt_embeds_,
            text_negative_prompt_embeds=negative_prompt_embeds_,
            point_prompt_embeds=point_prompt_embeds,
            point_negative_prompt_embeds=point_negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=self.pipe.unet.config.sample_size * 8,
            width=self.pipe.unet.config.sample_size * 8,
            seed=seed,
        )

        return image
    
class PoseControlNetV7:
    def __init__(self, sd_pipe, image_encoder_path, pose_ckpt, raw_base_points, device, num_tokens=77):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.pose_ckpt = pose_ckpt
        self.num_tokens = num_tokens
        self.raw_base_points = raw_base_points.to(torch.float16).to(self.device)
        self.pipe = sd_pipe.to(self.device)
        self.unet_copy = self.init_unet_copy()
        self.set_posectrl()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        self.image_proj_model_point = PointNetEncoder(channel=3).to(self.device, dtype=torch.float16)

        self.load_posectrl()
        

    def init_unet_copy(self):
        return PoseControlNetModel.from_unet(self.pipe.unet).to(self.device, dtype=torch.float16)


    def set_posectrl(self):
        unet_copy = self.unet_copy
        attn_procs_p = {}
        for name in unet_copy.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet_copy.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet_copy.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet_copy.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet_copy.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs_p[name] = AttnProcessor()
            else:
                attn_procs_p[name] = PoseAttnProcessorV7(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet_copy.set_attn_processor(attn_procs_p)

    def load_posectrl(self):
        if os.path.splitext(self.pose_ckpt)[-1] == ".safetensors":
            state_dict = {"unet_copy": {}, "atten_modules": {}, "image_proj_model_point": {}}
            with safe_open(self.pose_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("unet_copy."):
                        state_dict["unet_copy"][key.replace("unet_copy.", "")] = f.get_tensor(key)
                    elif key.startswith("atten_modules."):
                        state_dict["atten_modules"][key.replace("atten_modules.", "")] = f.get_tensor(key)
                    elif key.startswith("image_proj_model_point."):
                        state_dict["image_proj_model_point"][key.replace("image_proj_model_point.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.pose_ckpt, map_location="cpu")
        self.image_proj_model_point.load_state_dict(state_dict["image_proj_model_point"])
        self.unet_copy.load_state_dict(state_dict['unet_copy'])
        atten_layers = torch.nn.ModuleList(self.unet_copy.attn_processors.values())
        atten_layers.load_state_dict(state_dict["atten_modules"])

    @torch.inference_mode()
    def get_vpmatrix_points(self, V_matrix, P_matrix, points, prompt_embeds):
        image_prompt_embeds, trans_feat, importance = self.image_proj_model_point(points, V_matrix, P_matrix, prompt_embeds)
        uncond_image_prompt_embeds = self.image_proj_model_point(torch.zeros_like(points),V_matrix, P_matrix)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, PoseAttnProcessorV7):
                attn_processor.scale = scale    

    def custom_inference(self,
                        text_prompt_embeds,
                        text_negative_prompt_embeds,
                        point_prompt_embeds,
                        point_negative_prompt_embeds,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        height=512,
                        width=512,
                        seed=42):
        batch_size = point_prompt_embeds.shape[0]
        device = point_prompt_embeds.device
        dtype = torch.float16  # use half precision for memory optimization

        # Prepare random latent input
        latents = torch.randn(
            (batch_size, self.pipe.unet.config.in_channels, height // 8, width // 8),
            generator=torch.Generator(device=device).manual_seed(seed),
            device=device,
            dtype=dtype
        )

        # Expand prompt embeddings for classifier-free guidance
        point_prompt_embeds = torch.cat([point_negative_prompt_embeds, point_prompt_embeds], dim=0).to(device, dtype)
        text_prompt_embeds = torch.cat([text_negative_prompt_embeds, text_prompt_embeds], dim=0).to(device, dtype)
        # Set inference timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps

        with torch.no_grad():
            # Denoising loop
            for t in timesteps:
                # Duplicate latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2, dim=0)
                t_tensor = torch.full(
                    (latent_model_input.shape[0],),
                    t,
                    device=device,
                    dtype=dtype
                )

                # Get residual hints from auxiliary UNet
                down_block_res_samples, mid_block_res_sample = self.unet_copy(
                    latent_model_input, t_tensor, point_prompt_embeds, return_dict=False
                )

                # Predict noise using main UNet
                noise_pred = self.pipe.unet(
                    latent_model_input,
                    t_tensor,
                    encoder_hidden_states=text_prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                # Apply classifier-free guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Update latents via scheduler
                latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

            # Decode final latents into images
            latents = latents / self.pipe.vae.config.scaling_factor  # equivalent to 1 / 0.18215
            images = self.pipe.vae.decode(latents).sample
            images = (images / 2 + 0.5).clamp(0, 1)  # normalize to [0,1] range
            images = images.cpu().permute(0, 2, 3, 1).numpy()  # convert to HWC format

        return images



    def generate(
        self,
        prompt=None,
        prompt_embeds=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        V_matrix=None,
        P_matrix=None,
        points=None,
        **kwargs,
    ):
        self.set_scale(scale)

        if prompt is not None:
            # num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
            num_prompts = 1
        else:
            num_prompts = prompt_embeds.size(0)
        
        # 不需要prompt
        if prompt is None:
            prompt = "a highly detailed anime person, in front of a pure black background"
            # prompt = "girl"
        if negative_prompt is None:
            # negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, noise, cluttered background"
            negative_prompt = "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, realistic photo, huge eyes, worst face, 2girl, long fingers, disconnected limbs"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            point_prompt_embeds = torch.cat([prompt_embeds_, vpmatrix_points_embeds], dim=1)
            point_negative_prompt_embeds = torch.cat([negative_prompt_embeds_,uncon_vpmatrix_points_embeds], dim=1)

        vpmatrix_points_embeds, uncon_vpmatrix_points_embeds= self.get_vpmatrix_points(V_matrix, P_matrix, points, prompt_embeds)
        bs_embed, seq_len, _ = vpmatrix_points_embeds.shape
        vpmatrix_points_embeds = vpmatrix_points_embeds.repeat(1, num_samples, 1)
        vpmatrix_points_embeds = vpmatrix_points_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncon_vpmatrix_points_embeds = uncon_vpmatrix_points_embeds.repeat(1, num_samples, 1)
        uncon_vpmatrix_points_embeds = uncon_vpmatrix_points_embeds.view(bs_embed * num_samples, seq_len, -1)

        image = self.custom_inference(
            text_prompt_embeds=prompt_embeds_,
            text_negative_prompt_embeds=negative_prompt_embeds_,
            point_prompt_embeds=point_prompt_embeds,
            point_negative_prompt_embeds=point_negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=self.pipe.unet.config.sample_size * 8,
            width=self.pipe.unet.config.sample_size * 8,
            seed=seed,
        )

        return image
