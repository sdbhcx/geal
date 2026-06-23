import os
import random
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

from dataset.data_utils import normalize_point_cloud, CLASSES, AFFORDANCES, VIEWPOINTS

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
                 use_image: bool = False, img_size: int = 224):
        """
        Args:
            split (str): "train" or "test"
            setting (str): "seen" or "unseen"
            data_root (str): path to PIAD dataset root
            use_image (bool): if True, also return a real interaction image sampled
                from the (class, affordance) pool built by piad_process.build_image_index.
                Default False keeps the original 6-tuple output untouched.
            img_size (int): square size the interaction image is resized to.
        """
        self.split = split
        self.setting = setting
        self.data_root = data_root
        self.use_image = use_image

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
              f"setting={setting}, use_image={use_image})")

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
            image = self._sample_image(obj_class.lower(), affordance)
            return point_input, class_id, binary_mask, questions, affordance_id, gt_mask, image

        return point_input, class_id, binary_mask, questions, affordance_id, gt_mask

    # ------------------------------------------------------------------
    def _sample_image(self, obj_class: str, affordance: str):
        """
        Sample one interaction image from the (class, affordance) pool as an
        affordance-localization prior. Falls back to any image of the same class
        if the exact pair is missing.
        """
        paths = self.img_index.get((obj_class, affordance))
        if not paths:
            paths = [p for (c, _a), ps in self.img_index.items() if c == obj_class for p in ps]
        if not paths:
            raise KeyError(f"No interaction image for class={obj_class}, affordance={affordance}")
        path = random.choice(paths) if self.split == "train" else paths[0]
        return self.img_transform(Image.open(path).convert("RGB"))

    # ------------------------------------------------------------------
    def __len__(self):
        """Return dataset size."""
        return len(self.annotations)