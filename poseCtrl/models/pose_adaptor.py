import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.resnet import ResnetBlock2D, Upsample2D
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from poseCtrl.data.dataset import load_base_points
import cv2
import numpy as np

class VPmatrixEncoder(nn.Module):
    def __init__(self, input_channels=1, base_channels=64, output_size=(77, 77)):
        super(VPmatrixEncoder, self).__init__()

        # Input Layer
        self.input_layer = nn.Conv2d(input_channels, base_channels, kernel_size=1)

        # ResNet Blocks (Ensure `temb_channels=None` is passed)
        self.res_block1 = ResnetBlock2D(
            in_channels=base_channels, out_channels=base_channels * 2, temb_channels=None
        )
        self.res_block2 = ResnetBlock2D(
            in_channels=base_channels * 2, out_channels=base_channels * 4, temb_channels=None
        )
        self.res_block3 = ResnetBlock2D(
            in_channels=base_channels * 4, out_channels=base_channels * 8, temb_channels=None
        )

        # Upsampling
        self.upsample1 = Upsample2D(channels=base_channels * 8)  # Output: base_channels * 8
        self.upsample2 = Upsample2D(channels=base_channels * 4)  # Output: base_channels * 4
        self.final_conv = nn.Conv2d(base_channels * 4, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Ensure input has correct shape (batch, 1, 4, 4)
        x = x.unsqueeze(1) if x.ndim == 3 else x
        x = self.input_layer(x)

        # ResNet Feature Extraction (Passing `temb=None`)
        x = self.res_block1(x, temb=None)
        x = self.res_block2(x, temb=None)
        x = self.res_block3(x, temb=None)

        # Upsample Step (Ensure matching channels)
        if x.shape[1] != self.upsample1.channels:
            x = nn.Conv2d(x.shape[1], self.upsample1.channels, kernel_size=1)(x)

        x = self.upsample1(x)  # Expected Output: (batch, base_channels * 8, 8, 8)

        if x.shape[1] != self.upsample2.channels:
            x = nn.Conv2d(x.shape[1], self.upsample2.channels, kernel_size=1)(x)

        x = self.upsample2(x)  # Expected Output: (batch, base_channels * 4, 16, 16)

        # Interpolation to 77x77
        x = F.interpolate(x, size=(77, 77), mode='bilinear', align_corners=True)
        
        # Final Convolution to Ensure Output Shape
        x = self.final_conv(x).squeeze(1)  # (batch, 1, 77, 77) -> (batch, 77, 77)

        return x


import torch
import torch.nn as nn
from diffusers.models.resnet import ResnetBlock2D

class VPmatrixPoints(nn.Module):
    """ 
    Input:  
        V_matrix: [batch,4,4]
        P_matrix: [batch,4,4]
        raw_base_points: [13860,4]
    Output:
        base_points: [batch,77,768]
    """
    def __init__(self, raw_base_points):
        super().__init__() 
        self.register_buffer("raw_base_points", raw_base_points)

        self.resnet = nn.ModuleList([
            nn.Conv2d(720, 256, kernel_size=(3, 3), padding=(1, 1)),  
            ResnetBlock2D(in_channels=256, out_channels=256, temb_channels=None),  
            ResnetBlock2D(in_channels=256, out_channels=512, temb_channels=None),  
            ResnetBlock2D(in_channels=512, out_channels=768, temb_channels=None),  
            nn.Conv2d(768, 768, kernel_size=(1, 1))  
        ])

    def forward(self, V_matrix, P_matrix):
        VP_matrix = torch.bmm(P_matrix, V_matrix)  # [batch, 4, 4]
        points = self.raw_base_points.unsqueeze(0).expand(VP_matrix.shape[0], -1, -1)
        transformed_points = torch.bmm(points, VP_matrix.transpose(1, 2))  # [batch, 13860, 4]
        transformed_points[..., :3] = torch.where(
            transformed_points[..., 3:4] != 0,
            transformed_points[..., :3] / transformed_points[..., 3:4],
            transformed_points[..., :3]  
        ) # [batch, 13860, 3]
        transformed_points = transformed_points[..., :3]
        ones = torch.ones_like(transformed_points[..., :1])  # Create a tensor of ones with shape [batch, 13860, 1]
        transformed_points = torch.cat([transformed_points, ones], dim=-1)
        base_points = transformed_points.view(VP_matrix.shape[0], 77, 720)
        base_points = base_points.permute(0, 2, 1).unsqueeze(-1)  # [batch, 720, 77] → [batch, 720, 77, 1]

        for layer in self.resnet:
            if isinstance(layer, ResnetBlock2D):
                base_points = layer(base_points, temb=None)  
            else:
                base_points = layer(base_points)

        base_points = base_points.squeeze(-1).permute(0, 2, 1)  # [batch, 77, 768]

        return base_points

class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens
    
class VPProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

        self.vp_linear = torch.nn.Linear(torch.nn.Linear)
        self.activation = nn.Sigmoid()

    def forward(self, image_embeds, V_matrix, P_matrix):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)

        VP_matrix = torch.bmm(P_matrix, V_matrix)
        VP_embeds = self.vp_linear(VP_matrix.view(VP_matrix.shape[0], -1))
        VP_embeds = self.activation(VP_embeds)
        VP_embeds = VP_embeds.unsqueeze(1).repeat(1, self.clip_extra_context_tokens, 1)
        return clip_extra_context_tokens + VP_embeds

class VPmatrixPointsV1(nn.Module):
    """ 
    Input:  
        V_matrix: [batch,4,4]
        P_matrix: [batch,4,4]
        raw_base_points: [13860,4]
    Output:
        base_points: [batch,77,768]
    """
    def __init__(self, raw_base_points,image_width = 512,image_height=512):
        super().__init__() 
        self.register_buffer("raw_base_points", raw_base_points)
        self.image_width = image_width
        self.image_height = image_height

    def forward(self, V_matrix, P_matrix):
        VP_matrix = torch.bmm(P_matrix, V_matrix)  # [batch, 4, 4]
        points = self.raw_base_points.unsqueeze(0).expand(VP_matrix.shape[0], -1, -1).to('cuda')
        transformed_points = torch.bmm(points, VP_matrix.transpose(1, 2))  # [batch, 13860, 4]
        transformed_points[..., :3] = torch.where(
            transformed_points[..., 3:4] != 0,
            transformed_points[..., :3] / transformed_points[..., 3:4],
            transformed_points[..., :3]  
        ) # [batch, 13860, 3]
        transformed_points = transformed_points[..., :3]
        image_width, image_height = self.image_width, self.image_height

        screen_coords = transformed_points.clone()
        screen_coords[..., 0] = (screen_coords[..., 0] + 1) * 0.5 * image_width   # X: [-1,1] -> [0,512]
        screen_coords[..., 1] = (1 - (screen_coords[..., 1] + 1) * 0.5) * image_height  # Y 翻转: [-1,1] -> [512,0]

        screen_coords = screen_coords.round().long()  # [batch, 13860, 3]

        batch_size = screen_coords.shape[0]
        tensor_images = torch.zeros((batch_size, 3, image_height, image_width), dtype=torch.uint8)

        for b in range(batch_size):
            pixels = screen_coords[b].cpu().numpy()
            image_array = np.full((image_height, image_width), 255, dtype=np.uint8)

            for x, y, _ in pixels:
                if 0 <= x < image_width and 0 <= y < image_height:
                    image_array[y, x] = 0  
            inverted_array = 255 - image_array
            kernel = np.ones((3, 3), np.uint8)  
            dilated_image = cv2.dilate(inverted_array, kernel, iterations=1)  
            smoothed_image = cv2.GaussianBlur(dilated_image, (7, 7), 0)
            _, binary_mask = cv2.threshold(smoothed_image, 100, 255, cv2.THRESH_BINARY)
            binary_mask_3ch = np.stack([binary_mask] * 3, axis=-1)  # [512, 512, 3]
            tensor_images[b] = torch.from_numpy(binary_mask_3ch).permute(2, 0, 1)
        return tensor_images.float() / 255   

# --------------------- Dataset & Testing ---------------------

# import numpy as np

# from poseCtrl.data.dataset import CustomDataset

# path = r"F:\\Projects\\diffusers\\ProgramData\\sample_new"
# dataset = CustomDataset(path)
# data = dataset[0]

# # Generate VP Matrix
# vp_matrix = data['projection_matrix'] @ data['view_matrix']
# model = VPmatrixEncoder()
# vp_matrix_tensor = vp_matrix.float().unsqueeze(0)

# # Model Testing
# model = VPmatrixEncoder()
# output = model(vp_matrix_tensor)

# print("Input shape:", vp_matrix_tensor.shape)  # Expected: (1, 1, 4, 4)
# print("Output shape:", output.shape)  # Expected: (1, 77, 77)


# path=r'F:\Projects\diffusers\Project\PoseCtrl\dataSet\standardVertex.txt'
# raw_base_points=load_base_points(path)
# points = VPmatrixPoints(raw_base_points)
# with torch.no_grad():
#     base_points=points(data['view_matrix'].unsqueeze(0), data['projection_matrix'].unsqueeze(0))
# print(base_points.shape)