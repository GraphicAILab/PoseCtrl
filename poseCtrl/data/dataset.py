import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from pathlib import Path
from matplotlib import pyplot as plt
import re
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and txt files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),  
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])
        ])
        self.transform_feature = transform or transforms.Compose([
            transforms.Resize((512, 512)),  
            transforms.ToTensor(), 
        ])
        self.samples = []

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                data_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
                image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg')) and f.lower().startswith('capture')]

                data_files = sorted(
                    [f for f in os.listdir(folder_path) if f.endswith('.txt')],
                    key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else float('inf')
                )
                image_files = sorted(
                    [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg')) and f.lower().startswith('capture')],
                    key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else float('inf')
                )

                feature_file = os.path.join(folder_path, "feature.png")
                if not os.path.exists(feature_file):
                    raise FileNotFoundError(f"'{feature_file}' does not exist, please check again.")
                if len(data_files) == 134 and len(image_files) == 132:
                    projection_matrix_file = None
                    view_matrix_file = None
                    for data_file in data_files:
                        if 'projectionMatrix' in data_file:
                            projection_matrix_file = os.path.join(folder_path, data_file)
                        elif 'viewMatrix' in data_file:
                            view_matrix_file = os.path.join(folder_path, data_file)
                    image_files = [os.path.join(folder_path, img) for img in image_files]
                    if projection_matrix_file and view_matrix_file and image_files:
                        # 修改为每个图片与对应的矩阵文件配对
                        projection_matrices = self.read_matrices(projection_matrix_file)
                        view_matrices = self.read_matrices(view_matrix_file)
                        self.samples.extend([(proj, view, img, feature_file) for proj, view, img in zip(projection_matrices, view_matrices, image_files)])
                        # 添加调试信息
                        # print(f"Folder: {folder_name}, Number of images: {len(image_files)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        projection_matrix, view_matrix, image_file, feature_file = self.samples[idx]

        # 确保图片文件路径正确
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Image file not found: {image_file}")

        # 读取图像
        try:
            image = Image.open(image_file).convert('RGB')
        except IOError as e:
            raise IOError(f"Error opening image file {image_file}: {e}")
        
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Image file not found: {feature_file}")

        # 读取图像
        try:
            feature = Image.open(feature_file).convert('RGB')
        except IOError as e:
            raise IOError(f"Error opening image file {feature_file}: {e}")


        # 处理图像
        image = self.transform(image)  # **确保转换成 Tensor**
        feature = self.transform_feature(feature)
        # 确保矩阵是 Tensor
        projection_matrix = torch.tensor(projection_matrix, dtype=torch.float32)
        view_matrix = torch.tensor(view_matrix, dtype=torch.float32)

        # 确保 projection_matrix 和 view_matrix 形状正确
        if projection_matrix.shape != (4, 4):
            raise ValueError(f"Projection matrix shape is incorrect: {projection_matrix.shape}")
        if view_matrix.shape != (4, 4):
            raise ValueError(f"View matrix shape is incorrect: {view_matrix.shape}")

        sample = {
            'image': image,
            'projection_matrix': projection_matrix,
            'view_matrix': view_matrix,
            'feature': feature
        }
        return sample

    def read_matrices(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            matrices = []
            matrix = []
            for line in lines:
                if 'Capture' not in line:  # 跳过包含 'Capture' 的行
                    try:
                        row = list(map(float, line.strip().split()))
                        if len(row) == 4:  # 确保每一行有4个元素
                            matrix.append(row)
                            if len(matrix) == 4:  # 确保矩阵有4行
                                matrices.append(np.array(matrix))
                                matrix = []
                    except ValueError:
                        pass
            return matrices
        

def load_base_points(path):
    points = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' does not exist, please check again.")

    if path.endswith('.txt'):
        with open(path, 'r') as f:
            lines = f.readlines()  
            num_lines = len(lines)  
            if num_lines < 13860:
                for line in lines:
                    coords = list(map(float, line.strip().split()))
                    if len(coords) != 4:
                        raise ValueError(f"All points should have 4 coordinates, but found {len(coords)} in: {coords}")
                    points.append(coords)

                missing_points = 13860 - num_lines
                points.extend([[0, 0, 0, 0]] * missing_points)
            else:
                # 只取前 13860 个点
                for line in lines[:13860]:
                    coords = list(map(float, line.strip().split()))
                    if len(coords) != 4:
                        raise ValueError(f"All points should have 4 coordinates, but found {len(coords)} in: {coords}")
                    points.append(coords)

        points_tensor = torch.tensor(np.array(points, dtype=np.float32))
        return points_tensor

    else:
        pass  

# """ add 'set PYTHONPATH=F:/Projects/diffusers/Project' """
# train_dataset = CustomDataset("F:\\Projects\\diffusers\\ProgramData\\sample_new")

# train_dataloader = torch.utils.data.DataLoader(
#     train_dataset,
#     shuffle=True,
#     batch_size=32,
# )
# print(len(train_dataset))
# path=r'F:\Projects\diffusers\Project\PoseCtrl\dataSet\standardVertex.txt'
# base_points=load_base_points(path)
# print(base_points.shape)
import os
import re
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ResizeAndPad:
    """
    一个自定义的图像变换，它将图像的最长边调整到指定大小，
    然后用指定的颜色填充另一边，使图像变为正方形。
    """
    def __init__(self, output_size, fill_color=(255, 255, 255)):
        """
        Args:
            output_size (int): 期望的输出尺寸 (正方形的高和宽).
            fill_color (tuple): 用于填充的RGB颜色值.
        """
        self.output_size = output_size
        self.fill_color = fill_color

    def __call__(self, img):
        """
        Args:
            img (PIL.Image.Image): 输入的PIL图像.

        Returns:
            PIL.Image.Image: 经过调整和填充后的图像.
        """
        # 计算新的尺寸，同时保持宽高比
        w, h = img.size
        if w > h:
            new_w = self.output_size
            new_h = int(h * (self.output_size / w))
        else:
            new_h = self.output_size
            new_w = int(w * (self.output_size / h))
        
        # 调整图像大小
        resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 计算需要填充的量
        pad_left = (self.output_size - new_w) // 2
        pad_right = self.output_size - new_w - pad_left
        pad_top = (self.output_size - new_h) // 2
        pad_bottom = self.output_size - new_h - pad_top
        
        # 使用 transforms.Pad 进行填充
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        pad_transform = transforms.Pad(padding, fill=self.fill_color, padding_mode='constant')
        
        return pad_transform(resized_img)

class CustomDataset_v4(Dataset):
    """
    一个自定义的数据集加载器，它会根据一个主参数文件来加载图像、
    相机参数和文本描述。
    """
    def __init__(self, root_dir, camera_params_file, image_features_file, transform=None):
        """
        初始化数据集。
        """
        self.root_dir = root_dir
        # 如果没有提供transform，则使用包含自定义填充逻辑的默认图像变换
        self.transform = transform or transforms.Compose([
            ResizeAndPad(512, fill_color=(255, 255, 255)), # 使用新的自定义变换
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # 为RGB三通道进行归一化
        ])
        
        self.samples = []

        # 1. 加载图像特征（文本描述）
        try:
            with open(image_features_file, 'r', encoding='utf-8') as f:
                self.image_features = json.load(f)
        except FileNotFoundError:
            print(f"错误: 图像特征文件未找到于 '{image_features_file}'")
            self.image_features = {}
        except json.JSONDecodeError:
            print(f"错误: 解析JSON文件 '{image_features_file}' 失败")
            self.image_features = {}

        # 2. 解析相机参数文件并构建样本列表
        self._parse_camera_params(camera_params_file)

    def _parse_camera_params(self, file_path):
        """
        解析 camera_params.txt 文件来提取数据。
        """
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"错误: 相机参数文件未找到于 '{file_path}'")
            return

        current_image_path = None
        current_p_matrix = []
        current_v_matrix = []
        parsing_mode = None  # 可以是 'P', 'V', 或 None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检查是否是新的图像条目
            if '.jpg' in line or '.png' in line or '.webp' in line:
                # 如果我们已经处理完一个完整的条目，就保存它
                if current_image_path and len(current_p_matrix) == 4 and len(current_v_matrix) == 4:
                    self._add_sample(current_image_path, current_p_matrix, current_v_matrix)
                
                # 为下一个条目重置变量
                current_image_path = line.replace(':', '')
                current_p_matrix = []
                current_v_matrix = []
                parsing_mode = None
            elif line == 'P:':
                parsing_mode = 'P'
            elif line == 'V:':
                parsing_mode = 'V'
            # 解析矩阵行
            else:
                try:
                    # 使用正则表达式查找所有数字（包括负数和浮点数）
                    row_data = list(map(float, re.findall(r'-?\d+\.\d+(?:e-?\d+)?', line)))
                    if len(row_data) == 4:
                        if parsing_mode == 'P' and len(current_p_matrix) < 4:
                            current_p_matrix.append(row_data)
                        elif parsing_mode == 'V' and len(current_v_matrix) < 4:
                            current_v_matrix.append(row_data)
                except ValueError:
                    # 跳过无法解析的行
                    continue
        
        # 不要忘记添加文件中的最后一个条目
        if current_image_path and len(current_p_matrix) == 4 and len(current_v_matrix) == 4:
            self._add_sample(current_image_path, current_p_matrix, current_v_matrix)

    def _add_sample(self, image_path, p_matrix, v_matrix):
        """
        根据解析出的数据构建并添加一个样本。
        这个方法会忽略 image_path 中包含的任何父目录。
        """
        # 从 "batch_xxxxx/filename.ext" 中提取纯文件名 "filename.ext"
        image_filename = os.path.basename(image_path)
        
        # 将 root_dir 和提取出的文件名拼接成最终路径
        full_image_path = os.path.join(self.root_dir, image_filename)

        # 从字典中获取文本特征 (字典的键也是纯文件名)
        text_features = self.image_features.get(image_filename, [])
        text_prompt = ", ".join(text_features)

        # 仅当图像文件存在时才添加样本
        if os.path.exists(full_image_path):
            sample = {
                'image_path': full_image_path,
                'projection_matrix': np.array(p_matrix, dtype=np.float32),
                'view_matrix': np.array(v_matrix, dtype=np.float32),
                'text': text_prompt
            }
            self.samples.append(sample)
        else:
            print(f"警告: 图像文件 '{full_image_path}' 未找到，跳过此条目。")


    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取一个样本。
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples[idx]
        
        image_path = sample['image_path']
        
        try:
            image = Image.open(image_path).convert('RGB')
        except IOError as e:
            raise IOError(f"打开图像文件 '{image_path}' 时出错: {e}")

        # 应用图像变换
        image_tensor = self.transform(image)
        
        # 将numpy数组转换为torch张量
        p_matrix_tensor = torch.from_numpy(sample['projection_matrix'])
        v_matrix_tensor = torch.from_numpy(sample['view_matrix'])
        
        return {
            'image': image_tensor,
            'projection_matrix': p_matrix_tensor,
            'view_matrix': v_matrix_tensor,
            'text': sample['text']
        }

# if __name__ == '__main__':
#     # --- 使用示例 ---
#     ROOT_DIRECTORY = r'C:\Users\aw924\Downloads\image'
#     CAMERA_PARAMS_FILE = 'camera_params.txt'
#     IMAGE_FEATURES_FILE = 'image_features.txt'

#     # 3. 实例化数据集
#     custom_dataset = CustomDataset(
#         root_dir=ROOT_DIRECTORY,
#         camera_params_file=CAMERA_PARAMS_FILE,
#         image_features_file=IMAGE_FEATURES_FILE
#     )

#     # 4. 验证数据集
#     if len(custom_dataset) > 0:
#         print(f"成功加载数据集，共找到 {len(custom_dataset)} 个样本。")
        
#         first_sample = custom_dataset[0]
#         print("\n--- 第一个样本 ---")
#         print("Image shape:", first_sample['image'].shape) # 应该输出 torch.Size([3, 512, 512])
#         print("Text:", first_sample['text'])
#         print(len(custom_dataset))
#         print(first_sample['projection_matrix'], first_sample['view_matrix'])

#         from torch.utils.data import DataLoader
#         data_loader = DataLoader(custom_dataset, batch_size=2, shuffle=True, num_workers=0)
        
#         try:
#             first_batch = next(iter(data_loader))
#             print("\n--- 从 DataLoader 中获取的一个批次数据 ---")
#             print("Batch image shape:", first_batch['image'].shape) # 应该输出 torch.Size([2, 3, 512, 512])
#             print("Batch text:", first_batch['text'])
#         except Exception as e:
#             print(f"创建或迭代 DataLoader 时出错: {e}")
#     else:
#         print("未能从文件中加载任何样本。请检查文件内容和路径。")
class CombinedDataset_(Dataset):
    """
    A unified dataset loader that can load data from two different sources,
    each with its own structure and loading logic.
    """
    def __init__(self, path1=None, path2=None, camera_params_file_v4=None,
                 image_features_file_v4=None, transform=None):
        """
        Initializes the dataset by loading data from the provided paths.

        Args:
            path1 (str, optional): The root directory for the first dataset format (CustomDataset).
            path2 (str, optional): The root directory for the second dataset format (CustomDataset_v4).
            camera_params_file_v4 (str, optional): Path to the camera parameters file for path2.
            image_features_file_v4 (str, optional): Path to the image features (JSON) file for path2.
            transform (callable, optional): Optional transform to be applied to a sample.
        """
        self.transform = transform or transforms.Compose([
            ResizeAndPad(512, fill_color=(255, 255, 255)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # This transform is specific to the 'feature' image in the first dataset
        self.transform_feature = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        self.samples = []

        if path1:
            self._load_from_path1(path1)

        if path2 and camera_params_file_v4 and image_features_file_v4:
            self._load_from_path2(path2, camera_params_file_v4, image_features_file_v4)

    def _read_matrices_from_file(self, file_path):
        """Reads a file containing multiple 4x4 matrices."""
        with open(file_path, 'r') as file:
            lines = file.readlines()
            matrices = []
            matrix = []
            for line in lines:
                if 'Capture' not in line:
                    try:
                        row = list(map(float, line.strip().split()))
                        if len(row) == 4:
                            matrix.append(row)
                            if len(matrix) == 4:
                                matrices.append(np.array(matrix))
                                matrix = []
                    except ValueError:
                        continue
            return matrices

    def _load_from_path1(self, root_dir):
        """Loads data using the logic from the original CustomDataset."""
        print(f"Loading data from path1: {root_dir}")
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                data_files = sorted(
                    [f for f in os.listdir(folder_path) if f.endswith('.txt')],
                    key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else float('inf')
                )
                image_files = sorted(
                    [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg')) and f.lower().startswith('capture')],
                    key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else float('inf')
                )

                feature_file = os.path.join(folder_path, "feature.png")
                if not os.path.exists(feature_file):
                    print(f"Warning: 'feature.png' not found in {folder_path}. Skipping folder.")
                    continue

                projection_matrix_file = None
                view_matrix_file = None
                for data_file in data_files:
                    if 'projectionMatrix' in data_file:
                        projection_matrix_file = os.path.join(folder_path, data_file)
                    elif 'viewMatrix' in data_file:
                        view_matrix_file = os.path.join(folder_path, data_file)

                if projection_matrix_file and view_matrix_file and image_files:
                    projection_matrices = self._read_matrices_from_file(projection_matrix_file)
                    view_matrices = self._read_matrices_from_file(view_matrix_file)
                    
                    # Ensure the number of matrices matches the number of images
                    if len(projection_matrices) == len(view_matrices) == len(image_files):
                        for proj, view, img_name in zip(projection_matrices, view_matrices, image_files):
                            sample = {
                                'type': 'v1',
                                'image_path': os.path.join(folder_path, img_name),
                                'projection_matrix': proj,
                                'view_matrix': view,
                                'feature_path': feature_file,
                                'text': "highly detailed, anime"  # Default text
                            }
                            self.samples.append(sample)
                    else:
                        print(f"Warning: Mismatch in number of items in {folder_name}. Skipping.")


    def _load_from_path2(self, root_dir, camera_params_file, image_features_file):
        """Loads data using the logic from CustomDataset_v4."""
        print(f"Loading data from path2: {root_dir}")
        # 1. Load image features (text descriptions)
        try:
            with open(image_features_file, 'r', encoding='utf-8') as f:
                image_features = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading image features file '{image_features_file}': {e}")
            image_features = {}

        # 2. Parse camera parameters file
        try:
            with open(camera_params_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: Camera params file not found at '{camera_params_file}'")
            return

        current_image_path = None
        current_p_matrix = []
        current_v_matrix = []
        parsing_mode = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if any(ext in line for ext in ['.jpg', '.png', '.webp']):
                if current_image_path and len(current_p_matrix) == 4 and len(current_v_matrix) == 4:
                    self._add_v4_sample(root_dir, current_image_path, current_p_matrix, current_v_matrix, image_features)
                
                current_image_path = line.replace(':', '')
                current_p_matrix = []
                current_v_matrix = []
                parsing_mode = None
            elif line == 'P:':
                parsing_mode = 'P'
            elif line == 'V:':
                parsing_mode = 'V'
            else:
                try:
                    row_data = list(map(float, re.findall(r'-?\d+\.\d+(?:e-?\d+)?', line)))
                    if len(row_data) == 4:
                        if parsing_mode == 'P' and len(current_p_matrix) < 4:
                            current_p_matrix.append(row_data)
                        elif parsing_mode == 'V' and len(current_v_matrix) < 4:
                            current_v_matrix.append(row_data)
                except ValueError:
                    continue
        
        if current_image_path and len(current_p_matrix) == 4 and len(current_v_matrix) == 4:
            self._add_v4_sample(root_dir, current_image_path, current_p_matrix, current_v_matrix, image_features)

    def _add_v4_sample(self, root_dir, image_path, p_matrix, v_matrix, image_features):
        """Helper to construct and add a sample from path2."""
        image_filename = os.path.basename(image_path)
        full_image_path = os.path.join(root_dir, image_filename)
        
        text_prompt = ", ".join(image_features.get(image_filename, []))

        if os.path.exists(full_image_path):
            sample = {
                'type': 'v4',
                'image_path': full_image_path,
                'projection_matrix': np.array(p_matrix, dtype=np.float32),
                'view_matrix': np.array(v_matrix, dtype=np.float32),
                'text': text_prompt
            }
            self.samples.append(sample)
        else:
            print(f"Warning: Image file '{full_image_path}' not found. Skipping.")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the given index.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_data = self.samples[idx]
        
        try:
            image = Image.open(sample_data['image_path']).convert('RGB')
        except IOError as e:
            raise IOError(f"Error opening image file '{sample_data['image_path']}': {e}")
        
        # Apply the primary transform to the main image
        image_tensor = self.transform(image)

        # Convert matrices to tensors
        p_matrix_tensor = torch.tensor(sample_data['projection_matrix'], dtype=torch.float32)
        v_matrix_tensor = torch.tensor(sample_data['view_matrix'], dtype=torch.float32)

        # Prepare the final sample dictionary
        final_sample = {
            'image': image_tensor,
            'projection_matrix': p_matrix_tensor,
            'view_matrix': v_matrix_tensor,
            'text': sample_data['text']
        }

        # For v1 data, load and transform the 'feature' image
        if sample_data['type'] == 'v1':
            try:
                feature_image = Image.open(sample_data['feature_path']).convert('RGB')
                final_sample['feature'] = self.transform_feature(feature_image)
            except IOError as e:
                 raise IOError(f"Error opening feature file '{sample_data['feature_path']}': {e}")

        return final_sample
    

import os
import json
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# 假设 ResizeAndPad 是您自定义的转换，这里保留其引用
# from your_transforms import ResizeAndPad 

class CombinedDataset(Dataset):
    """
    一个统一的数据集加载器，可以从多个不同的数据源加载数据。
    - path1 使用 v1 格式。
    - path2, path3, path4, path6 使用 v4 格式。
    """
    def __init__(self, path1=None, path2=None, path3=None, path4=None, path5=None, transform=None):
        """
        通过加载所提供路径中的数据来初始化数据集。

        Args:
            path1 (str, optional): v1 数据集格式的根目录。
            path2 (str, optional): v4 数据集格式的根目录。
            path3 (str, optional): v4 数据集格式的根目录。
            path4 (str, optional): v4 数据集格式的根目录。
            path6 (str, optional): v4 数据集格式的根目录。
            transform (callable, optional): 应用于样本的可选转换。
        """
        # 如果未提供 transform，则使用默认的转换流程
        self.transform = transform or transforms.Compose([
            # 假设 ResizeAndPad 是一个有效的转换类
            # ResizeAndPad(512, fill_color=(255, 255, 255)),
            transforms.Resize((512, 512)), # 使用标准的Resize作为替代
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # 这个转换特定于 v1 数据集中的 'feature' 图像
        self.transform_feature = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        self.samples = []

        # 加载 v1 格式的数据
        if path1:
            self._load_from_path1(path1)

        # 将所有 v4 格式的路径集中处理
        v4_style_paths = {
            "path2": path2,
            "path3": path3,
            "path4": path4,
            "path5": path5
        }

        for path_name, root_path in v4_style_paths.items():
            if root_path:
                self._load_v4_style_data(root_path, path_name)

    def _read_matrices_from_file(self, file_path):
        """读取包含多个 4x4 矩阵的文件。"""
        # ... (此方法无需改动)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            matrices = []
            matrix = []
            for line in lines:
                if 'Capture' not in line:
                    try:
                        row = list(map(float, line.strip().split()))
                        if len(row) == 4:
                            matrix.append(row)
                            if len(matrix) == 4:
                                matrices.append(np.array(matrix))
                                matrix = []
                    except ValueError:
                        continue
            return matrices

    def _load_from_path1(self, root_dir):
        """使用 v1 数据集的逻辑加载数据。"""
        # ... (此方法无需改动)
        print(f"Loading data from path1: {root_dir}")
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                # ... (内部逻辑保持不变)
                data_files = sorted(
                    [f for f in os.listdir(folder_path) if f.endswith('.txt')],
                    key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else float('inf')
                )
                image_files = sorted(
                    [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg')) and f.lower().startswith('capture')],
                    key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else float('inf')
                )
                feature_file = os.path.join(folder_path, "feature.png")
                if not os.path.exists(feature_file):
                    print(f"Warning: 'feature.png' not found in {folder_path}. Skipping folder.")
                    continue
                projection_matrix_file = None
                view_matrix_file = None
                for data_file in data_files:
                    if 'projectionMatrix' in data_file:
                        projection_matrix_file = os.path.join(folder_path, data_file)
                    elif 'viewMatrix' in data_file:
                        view_matrix_file = os.path.join(folder_path, data_file)
                if projection_matrix_file and view_matrix_file and image_files:
                    projection_matrices = self._read_matrices_from_file(projection_matrix_file)
                    view_matrices = self._read_matrices_from_file(view_matrix_file)
                    if len(projection_matrices) == len(view_matrices) == len(image_files):
                        for proj, view, img_name in zip(projection_matrices, view_matrices, image_files):
                            sample = {
                                'type': 'v1',
                                'image_path': os.path.join(folder_path, img_name),
                                'projection_matrix': proj,
                                'view_matrix': view,
                                'feature_path': feature_file,
                                'text': "highly detailed, anime"
                            }
                            self.samples.append(sample)
                    else:
                        print(f"Warning: Mismatch in number of items in {folder_name}. Skipping.")

    def _load_v4_style_data(self, root_dir, path_name):
        """
        加载 v4 风格的数据集。
        假定 'camera_params.txt' 和 'image_features.json' 文件位于 root_dir 中。

        Args:
            root_dir (str): 数据集的根目录。
            path_name (str): 用于日志记录的路径名称 (例如 'path2')。
        """
        print(f"Loading data from {path_name}: {root_dir}")
        
        camera_params_file = os.path.join(root_dir, 'camera_params.txt')
        image_features_file = os.path.join(root_dir, 'image_features.txt')

        try:
            with open(image_features_file, 'r', encoding='utf-8') as f:
                image_features = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning for {path_name}: Error or file not found for image features '{image_features_file}': {e}. Proceeding without text prompts.")
            image_features = {}

        try:
            with open(camera_params_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error for {path_name}: Camera params file not found at '{camera_params_file}'. Skipping this data source.")
            return

        current_image_path = None
        current_p_matrix = []
        current_v_matrix = []
        parsing_mode = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if any(ext in line for ext in ['.jpg', '.png', '.webp']):
                if current_image_path and len(current_p_matrix) == 4 and len(current_v_matrix) == 4:
                    self._add_v4_sample(root_dir, current_image_path, current_p_matrix, current_v_matrix, image_features)
                current_image_path = line.replace(':', '')
                current_p_matrix, current_v_matrix, parsing_mode = [], [], None
            elif line == 'P:':
                parsing_mode = 'P'
            elif line == 'V:':
                parsing_mode = 'V'
            else:
                try:
                    row_data = list(map(float, re.findall(r'-?\d+\.\d+(?:e-?\d+)?', line)))
                    if len(row_data) == 4:
                        if parsing_mode == 'P' and len(current_p_matrix) < 4:
                            current_p_matrix.append(row_data)
                        elif parsing_mode == 'V' and len(current_v_matrix) < 4:
                            current_v_matrix.append(row_data)
                except ValueError:
                    continue
        
        if current_image_path and len(current_p_matrix) == 4 and len(current_v_matrix) == 4:
            self._add_v4_sample(root_dir, current_image_path, current_p_matrix, current_v_matrix, image_features)

    def _add_v4_sample(self, root_dir, image_path, p_matrix, v_matrix, image_features):
        """辅助函数，用于构建和添加一个 v4 风格的样本。"""
        # ... (此方法无需改动)
        image_filename = os.path.basename(image_path)
        full_image_path = os.path.join(root_dir, image_filename)
        text_prompt = ", ".join(image_features.get(image_filename, []))

        if os.path.exists(full_image_path):
            sample = {
                'type': 'v4',  # 所有这些路径的数据都标记为 v4 类型
                'image_path': full_image_path,
                'projection_matrix': np.array(p_matrix, dtype=np.float32),
                'view_matrix': np.array(v_matrix, dtype=np.float32),
                'text': text_prompt
            }
            self.samples.append(sample)
        else:
            print(f"Warning: Image file '{full_image_path}' not found. Skipping.")

    def __len__(self):
        """返回数据集中的样本总数。"""
        # ... (此方法无需改动)
        return len(self.samples)

    def __getitem__(self, idx):
        """从数据集中检索给定索引的样本。"""
        # ... (此方法无需改动)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_data = self.samples[idx]
        
        try:
            image = Image.open(sample_data['image_path']).convert('RGB')
        except IOError as e:
            raise IOError(f"Error opening image file '{sample_data['image_path']}': {e}")
        
        image_tensor = self.transform(image)
        p_matrix_tensor = torch.tensor(sample_data['projection_matrix'], dtype=torch.float32)
        v_matrix_tensor = torch.tensor(sample_data['view_matrix'], dtype=torch.float32)

        final_sample = {
            'image': image_tensor,
            'projection_matrix': p_matrix_tensor,
            'view_matrix': v_matrix_tensor,
            'text': sample_data['text']
        }

        # 对于 v1 类型的数据，加载并转换 'feature' 图像
        if sample_data['type'] == 'v1':
            try:
                feature_image = Image.open(sample_data['feature_path']).convert('RGB')
                final_sample['feature'] = self.transform_feature(feature_image)
            except IOError as e:
                raise IOError(f"Error opening feature file '{sample_data['feature_path']}': {e}")

        return final_sample