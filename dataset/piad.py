import os
import random
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from dataset.data_utils import normalize_point_cloud, CLASSES, AFFORDANCES, VIEWPOINTS

# ------------------------------
# SAM Segmenter (Singleton)
# ------------------------------

class SAMSegmenter:
    """
    Singleton wrapper for Segment Anything Model (SAM) for object segmentation.
    Uses the vit_h large model for better accuracy.
    """
    _instance = None
    
    def __new__(cls, device=None):
        if cls._instance is None:
            cls._instance = super(SAMSegmenter, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self, device=None):
        if self._initialized:
            return
        
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load SAM model (vit_h for better accuracy)
            self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(self.device)
            self.predictor = SamPredictor(self.sam)
            self._initialized = True
            print(f"[SAM] Loaded successfully on {self.device}")
        except ImportError:
            raise RuntimeError("Please install segment-anything: pip install segment-anything")
        except FileNotFoundError:
            raise RuntimeError("SAM checkpoint not found. Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
    
    def segment_object(self, image_pil):
        """
        Segment the main object in the image using SAM.
        Returns the original image with background zeroed out.
        
        Args:
            image_pil (PIL.Image): Input RGB image
        Returns:
            tuple: (masked_image_tensor, mask_tensor)
                   masked_image: [3, H, W] with background zeroed out
                   mask: [1, H, W] binary mask (1=object, 0=background)
        """
        if not self._initialized:
            self.initialize()
        
        # Convert PIL to numpy (SAM expects numpy array)
        image_np = np.array(image_pil)
        
        # Use automatic mask generation (no prompts needed)
        from segment_anything import SamAutomaticMaskGenerator
        
        mask_generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        
        masks = mask_generator.generate(image_np)
        
        if not masks:
            # Fallback: return original image with full mask
            print("[SAM] No masks found, returning original image")
            transform = T.Compose([T.ToTensor()])
            return transform(image_pil), torch.ones(1, image_pil.size[1], image_pil.size[0])
        
        # Select the largest mask (assumed to be the main object)
        largest_mask = max(masks, key=lambda x: x['area'])
        mask_np = largest_mask['segmentation']
        
        # Apply mask to image
        image_np_masked = image_np.copy()
        image_np_masked[~mask_np] = 0  # Zero out background
        
        # Convert back to tensor
        transform = T.Compose([T.ToTensor()])
        masked_tensor = transform(Image.fromarray(image_np_masked))
        mask_tensor = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)
        
        return masked_tensor, mask_tensor


# Global SAM instance
sam_segmenter = SAMSegmenter()

# ------------------------------
# PIAD Dataset
# ------------------------------

class PiadDataset(Dataset):
    """
    PIAD: Point-based Interactive Affordance Dataset.

    Each sample contains:
        - normalized point cloud (N, 3)
        - object class ID
        - binary affordance mask
        - 12 viewpoint-conditioned affordance questions
        - affordance label ID
        - ground truth affordance mask
    """

    def __init__(self, split: str = "train", setting: str = "seen", data_root: str = "piad_dataset",
                 use_image: bool = False, img_size: int = 224, use_sam: bool = False):
        """
        Args:
            split (str): "train" or "test"
            setting (str): "seen" or "unseen"
            data_root (str): path to PIAD dataset root
            use_image (bool): if True, also return a real interaction image sampled
                from the (class, affordance) pool built by piad_process.build_image_index.
                Default False keeps the original 6-tuple output untouched.
            img_size (int): square size the interaction image is resized to.
            use_sam (bool): if True, apply SAM segmentation to isolate the object.
        """
        self.split = split
        self.setting = setting
        self.data_root = data_root
        self.use_image = use_image
        self.use_sam = use_sam
        
        # Initialize SAM if needed
        if self.use_sam and self.use_image:
            sam_segmenter.initialize()

        # Build class and affordance name → index mappings
        self.class_to_idx = {cls.lower(): i for i, cls in enumerate(CLASSES)}
        self.aff_to_idx = {aff: i for i, aff in enumerate(AFFORDANCES)}

        # Load annotations (contains point cloud + labels)
        anno_path = os.path.join(data_root, f"{setting}_{split}.pkl")
        with open(anno_path, "rb") as f:
            self.annotations = pickle.load(f)

        # Load affordance rephrasing table
        self.questions = pd.read_csv(os.path.join(data_root, "Affordance-Question.csv"))

        # Optionally load the (class, affordance) -> [image paths] sidecar index
        self.img_index = None
        if self.use_image:
            idx_path = os.path.join(data_root, f"{setting}_{split}_img_index.pkl")
            with open(idx_path, "rb") as f:
                self.img_index = pickle.load(f)
            self.img_transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        print(f"[PIAD] Loaded {split} split ({len(self.annotations)} samples, "
              f"setting={setting}, use_image={use_image}, use_sam={use_sam})")

    # ------------------------------------------------------------------
    def _sample_question(self, object_name: str, affordance: str) -> str:
        """
        Retrieve one random rephrased question for an object-affordance pair.
        Training randomly samples from 15 variants; test uses 'Question0'.

        Args:
            object_name (str): object category name
            affordance (str): affordance type
        Returns:
            str: question text
        """
        qid = f"Question{np.random.randint(1, 15)}" if self.split == "train" else "Question0"
        row = self.questions.loc[
            (self.questions["Object"] == object_name) & (self.questions["Affordance"] == affordance),
            [qid]
        ]
        if not row.empty:
            return row.iloc[0][qid]
        raise ValueError(f"No question found for {object_name}-{affordance}")

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        """
        Retrieve one PIAD sample.
        Returns:
            point_input (np.ndarray): (3, N) normalized point cloud
            class_id (int): object class index
            binary_mask (np.ndarray): binary affordance mask (0/1)
            questions (tuple[str]): 12 rephrased affordance questions
            affordance_id (int): affordance label index
            gt_mask (np.ndarray): original affordance mask
        """
        data = self.annotations[idx]
        obj_class = data["class"]
        affordance = data["affordance"]
        gt_mask = data["mask"]
        points = data["point"]

        # Normalize point cloud
        points, _, _ = normalize_point_cloud(points)

        # Convert mask to binary
        binary_mask = (gt_mask > 0).astype(np.uint8)

        # Retrieve affordance question
        question = self._sample_question(obj_class, affordance)

        # Construct viewpoint-prefixed questions
        questions = tuple(
            f"This is a depth map of a {obj_class} viewed {vp}. {question}"
            for vp in VIEWPOINTS
        )

        # Convert to model input format
        point_input = points.T  # shape: (3, N)
        class_id = self.class_to_idx[obj_class.lower()]
        affordance_id = self.aff_to_idx[affordance]

        if self.use_image:
            image_pil = self._sample_image(obj_class.lower(), affordance)
            if self.use_sam:
                image = self._apply_sam_segmentation(image_pil)
            else:
                image = self.img_transform(image_pil)
            return point_input, class_id, binary_mask, questions, affordance_id, gt_mask, image

        return point_input, class_id, binary_mask, questions, affordance_id, gt_mask

    # ------------------------------------------------------------------
    def _sample_image(self, obj_class: str, affordance: str):
        """
        Sample one interaction image from the (class, affordance) pool as an
        affordance-localization prior. Falls back to any image of the same class
        if the exact pair is missing.

        Returns:
            PIL.Image: Original image (before SAM processing)
        """
        paths = self.img_index.get((obj_class, affordance))
        if not paths:
            paths = [p for (c, _a), ps in self.img_index.items() if c == obj_class for p in ps]
        if not paths:
            raise KeyError(f"No interaction image for class={obj_class}, affordance={affordance}")
        path = random.choice(paths) if self.split == "train" else paths[0]
        return Image.open(path).convert("RGB")  # Return PIL image for potential SAM processing
    
    def _apply_sam_segmentation(self, image_pil):
        """
        Apply SAM segmentation to isolate the object from background.

        Args:
            image_pil (PIL.Image): Original RGB image
        Returns:
            torch.Tensor: Segmented image tensor [3, H, W] with background zeroed
        """
        try:
            masked_tensor, _ = sam_segmenter.segment_object(image_pil)
            # Apply normalization
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            return normalize(masked_tensor)
        except Exception as e:
            print(f"[SAM] Error processing image: {e}")
            # Fallback: return normalized original image
            return self.img_transform(image_pil)

    # ------------------------------------------------------------------
    def __len__(self):
        """Return dataset size."""
        return len(self.annotations)