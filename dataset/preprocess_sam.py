"""
SAM特征预处理脚本
处理PIAD数据集的2D交互图像
提前提取所有数据的SAM特征并保存，训练时直接加载
"""
import os
import argparse
import torch
import numpy as np
import pickle
import json
import re
import sys
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

sys.path.append(".")
from utils.utils import read_yaml


class SAMPREprocessor:
    def __init__(self, sam_checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h", device=None, img_size=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """
        初始化SAM模型
        Args:
            sam_checkpoint: SAM模型权重路径
            model_type: SAM模型类型 (vit_b/vit_l/vit_h)
            device: GPU设备
            img_size: 图像resize尺寸，如 (224, 224)，None表示不resize
        """
        print(f"[SAM] Loading SAM {model_type}...")
        self.img_size = img_size
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.sam.eval()
        self.predictor = SamPredictor(self.sam)
    
    @torch.no_grad()
    def extract_features(self, image, bbox=None):
        """
        提取单张图像的SAM特征
        Args:
            image: PIL.Image或[H, W, 3] numpy数组
            bbox: 可选的边界框，格式为 [x_min, y_min, x_max, y_max]
                  如果提供，将使用边界框裁剪图像
        Returns:
            features: [256, H/16, W/16] SAM图像嵌入
        """
        # 处理PIL Image输入
        if isinstance(image, Image.Image):
            original_size = image.size  # (width, height)
            
            # 如果提供了bbox，先根据bbox裁剪图像
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                # 确保bbox在图像范围内
                x_min = max(0, int(x_min))
                y_min = max(0, int(y_min))
                x_max = min(original_size[0], int(x_max))
                y_max = min(original_size[1], int(y_max))
                
                # 裁剪图像到bbox区域
                image = image.crop((x_min, y_min, x_max, y_max))
            
            # 如果指定了图像尺寸，再resize
            if self.img_size is not None:
                image = image.resize(self.img_size, Image.Resampling.LANCZOS)
            
            image = np.array(image)
        
        # 处理不同格式的输入
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:
                image = image.permute(1, 2, 0).cpu().numpy()
            else:
                image = image.cpu().numpy()
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
        
        # 确保图像是uint8格式
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
        
        self.predictor.set_image(image)
        features = self.predictor.get_image_embedding()  # [1, 256, H/16, W/16]
        return features.squeeze(0)


def main(cfg_path="config/train_stage1.yaml"):
    cfg = read_yaml(cfg_path)
    dataset_cfg = cfg["dataset"]
    sam_cfg = cfg.get("sam", {})
    
    # 创建输出目录
    sam_feature_dir = dataset_cfg.get("sam_feature_dir", 
        os.path.join(dataset_cfg["data_root"], "sam_features"))
    os.makedirs(sam_feature_dir, exist_ok=True)
    print(f"[INFO] SAM features will be saved to: {sam_feature_dir}")
    
    gpu_id = cfg["train"]["gpu"]
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # 获取图像尺寸配置，用于统一特征尺寸
    img_size = sam_cfg.get("img_size", (224, 224))
    print(f"[INFO] Resizing images to {img_size}")
    
    # 初始化SAM处理器
    sam_processor = SAMPREprocessor(
        sam_checkpoint=sam_cfg.get("checkpoint", "sam_vit_h_4b8939.pth"),
        model_type=sam_cfg.get("model_type", "vit_h"),
        device=device,
        img_size=img_size
    )
    
    # 处理PIAD数据集的2D交互图像
    if dataset_cfg["category"] == "piad":
        # 处理训练集
        train_img_index_path = os.path.join(dataset_cfg["data_root"], 
            f"{dataset_cfg['setting']}_train_img_index.pkl")
        process_piad_images(train_img_index_path, sam_processor, sam_feature_dir, "train",
                           data_root=dataset_cfg["data_root"], setting=dataset_cfg["setting"])
        
        # 处理测试集
        test_img_index_path = os.path.join(dataset_cfg["data_root"], 
            f"{dataset_cfg['setting']}_test_img_index.pkl")
        process_piad_images(test_img_index_path, sam_processor, sam_feature_dir, "test",
                           data_root=dataset_cfg["data_root"], setting=dataset_cfg["setting"])
    
    print(f"[INFO] SAM feature extraction complete!")


def load_bounding_boxes(data_root, setting, split):
    """加载PIAD数据集的Bounding_Box信息
    
    从 /{data_root}/{setting}/Bounding_Box/{split}/【物体】/ 目录下的JSON文件加载Bounding_Box
    JSON格式为LabelMe格式，包含shapes数组，其中label为"object"的是物体边界框
    
    Returns:
        dict: 图像路径到bounding box的映射，格式为 {img_path: [x_min, y_min, x_max, y_max]}
    """
    bbox_index = {}
    
    # Bounding_Box目录结构: {data_root}/{setting}/Bounding_Box/{split}/{object_class}/*.json
    bbox_base_dir = os.path.join(data_root, setting, "Bounding_Box", split)
    
    if not os.path.exists(bbox_base_dir):
        print(f"[WARN] Bounding_Box directory not found: {bbox_base_dir}")
        return bbox_index
    
    # 遍历所有物体类别目录
    for obj_class in os.listdir(bbox_base_dir):
        obj_dir = os.path.join(bbox_base_dir, obj_class)
        if not os.path.isdir(obj_dir):
            continue
        
        # 遍历该类别下的所有JSON文件
        for json_file in os.listdir(obj_dir):
            if not json_file.endswith(".json"):
                continue
            
            json_path = os.path.join(obj_dir, json_file)
            try:
                with open(json_path, "r") as f:
                    labelme_data = json.load(f)
                
                # 解析LabelMe格式的JSON
                # 获取图像路径
                image_path = labelme_data.get("imagePath", "")
                if not image_path:
                    # 如果没有imagePath，从JSON文件名推断
                    img_name = json_file.replace(".json", ".jpg")
                    image_path = img_name
                
                # 从shapes数组中找到label为"object"的边界框
                bbox = None
                shapes = labelme_data.get("shapes", [])
                for shape in shapes:
                    if shape.get("label") == "object" and shape.get("shape_type") == "rectangle":
                        points = shape.get("points", [])
                        if len(points) >= 2:
                            # 矩形的两个对角点
                            x1, y1 = points[0]
                            x2, y2 = points[1]
                            # 计算bbox: [x_min, y_min, x_max, y_max]
                            x_min = min(x1, x2)
                            y_min = min(y1, y2)
                            x_max = max(x1, x2)
                            y_max = max(y1, y2)
                            bbox = [x_min, y_min, x_max, y_max]
                            break
                
                if bbox is not None:
                    # 将相对路径转换为绝对路径
                    if not os.path.isabs(image_path):
                        # imagePath是相对路径，相对于JSON文件所在目录
                        abs_image_path = os.path.normpath(os.path.join(obj_dir, image_path))
                        bbox_index[abs_image_path] = bbox
                    else:
                        bbox_index[image_path] = bbox
                    
            except Exception as e:
                print(f"[WARN] Failed to parse {json_path}: {e}")
                continue
    
    print(f"[INFO] Loaded {len(bbox_index)} bounding boxes from {bbox_base_dir}")
    return bbox_index


def find_bbox_for_image(img_path, bbox_index):
    """根据图像路径查找对应的Bounding_Box
    
    支持基于文件名的匹配
    
    Args:
        img_path: 图像路径
        bbox_index: bbox索引字典
    
    Returns:
        list or None: [x_min, y_min, x_max, y_max] 或 None
    """
    # 首先尝试精确匹配
    if img_path in bbox_index:
        return bbox_index[img_path]
    
    # 尝试基于文件名的匹配
    img_name = os.path.basename(img_path)
    if img_name in bbox_index:
        return bbox_index[img_name]
    
    # 尝试去掉扩展名的匹配
    img_name_no_ext = os.path.splitext(img_name)[0]
    for key in bbox_index.keys():
        key_name_no_ext = os.path.splitext(os.path.basename(key))[0]
        if key_name_no_ext == img_name_no_ext:
            return bbox_index[key]
    
    return None


def process_piad_images(img_index_path, sam_processor, output_dir, split_name, data_root=None, setting=None):
    """处理PIAD数据集的2D交互图像，提取SAM特征"""
    # 加载图像索引文件
    if not os.path.exists(img_index_path):
        print(f"[WARN] Image index file not found: {img_index_path}")
        return
    
    with open(img_index_path, "rb") as f:
        img_index = pickle.load(f)
    
    # 加载Bounding_Box信息（如果提供了数据集根目录）
    bbox_index = {}
    if data_root and setting:
        bbox_index = load_bounding_boxes(data_root, setting.capitalize(), split_name.capitalize())
    
    # 收集所有图像路径
    all_image_paths = []
    for key, paths in img_index.items():
        all_image_paths.extend(paths)
    
    total_images = len(all_image_paths)
    print(f"[INFO] Found {total_images} images in {split_name} split")
    
    # 创建图像到特征的映射字典
    img_to_feature = {}
    
    for idx, img_path in enumerate(all_image_paths):
        if idx % 50 == 0:
            print(f"[Progress {split_name}] {idx}/{total_images}")
        
        try:
            # 打开图像
            image = Image.open(img_path).convert("RGB")
            
            # 获取对应的Bounding_Box（如果存在）
            bbox = find_bbox_for_image(img_path, bbox_index)
            
            # 提取SAM特征（使用Bounding_Box裁剪）
            sam_feat = sam_processor.extract_features(image, bbox)
            
            # 保存到映射字典
            img_to_feature[img_path] = sam_feat
            
        except Exception as e:
            print(f"[ERROR] Failed to process {img_path}: {e}")
            continue
    
    # 保存特征映射
    save_path = os.path.join(output_dir, f"{split_name}_sam_features_dict.pt")
    torch.save(img_to_feature, save_path)
    print(f"[INFO] Saved {len(img_to_feature)} SAM features to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_stage1.yaml")
    opt = parser.parse_args()
    main(opt.config)