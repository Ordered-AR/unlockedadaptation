import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class RandARStyleDataset(Dataset):
    """
    Dataset for RandAR-style data with subdirectories per class.

    Structure:
    data_path/
      ├── 0/
      │   ├── sample1.npy
      │   ├── sample2.npy
      ├── 1/
      │   ├── sample1.npy
      ...
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        self.classes = sorted(
            [
                int(c)
                for c in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, c))
            ]
        )

        print(f"Loading RandAR-style dataset from {root_dir}...")
        print(f"Found {len(self.classes)} classes")

        # Collect all samples
        for class_id in self.classes:
            class_dir = os.path.join(root_dir, str(class_id))
            for fname in sorted(os.listdir(class_dir)):
                if fname.endswith(".npy") or fname.endswith(".npz"):
                    sample_path = os.path.join(class_dir, fname)
                    self.samples.append((sample_path, class_id))

        print(f"Loaded {len(self.samples)} samples from {len(self.classes)} classes")
        self.flip = False  # For compatibility with existing code
        self.feature_dir = root_dir  # For compatibility
        self.aug_feature_dir = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, class_id = self.samples[idx]

        # Load features (latent codes)
        if sample_path.endswith(".npz"):
            npz = np.load(sample_path)
            if "latents" in npz:
                features = npz["latents"]
            else:
                first_key = list(npz.keys())[0]
                features = npz[first_key]
        else:
            features = np.load(sample_path)

        # Create label (class_id)
        label = np.array(class_id, dtype=np.int64)

        return torch.from_numpy(features), torch.from_numpy(label)


class CustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.flip = "flip" in self.feature_dir

        aug_feature_dir = feature_dir.replace("ten_crop/", "ten_crop_105/")
        aug_label_dir = label_dir.replace("ten_crop/", "ten_crop_105/")
        if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        # self.feature_files = sorted(os.listdir(feature_dir))
        # self.label_files = sorted(os.listdir(label_dir))
        # TODO: make it configurable
        self.feature_files = [f"{i}.npy" for i in range(1281167)]
        self.label_files = [f"{i}.npy" for i in range(1281167)]

    def __len__(self):
        assert len(self.feature_files) == len(
            self.label_files
        ), "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):
        if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
            feature_dir = self.aug_feature_dir
            label_dir = self.aug_label_dir
        else:
            feature_dir = self.feature_dir
            label_dir = self.label_dir

        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]

        features = np.load(os.path.join(feature_dir, feature_file))
        if self.flip:
            aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
            features = features[:, aug_idx]
        labels = np.load(os.path.join(label_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)


def build_imagenet(args, transform):
    return ImageFolder(args.data_path, transform=transform)


def build_imagenet_code(args):
    """
    Build ImageNet dataset from pre-extracted codes.

    Supports two data formats:
    1. LlamaGen flat format: {code_path}/imagenet{size}_codes/{0.npy, 1.npy, ...}
                            {code_path}/imagenet{size}_labels/{0.npy, 1.npy, ...}

    2. RandAR subdirectory format: {code_path}/{0/, 1/, 2/, ...}/{samples.npy}
    """
    # Try standard LlamaGen naming first
    feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
    label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"

    # Check if we have LlamaGen format
    has_llamagen_format = os.path.exists(feature_dir) and os.path.exists(label_dir)

    if not has_llamagen_format:
        # Try alternative LlamaGen naming patterns
        if args.code_path.endswith("_codes"):
            feature_dir = args.code_path
            label_dir = args.code_path.replace("_codes", "_labels")
            has_llamagen_format = os.path.exists(feature_dir) and os.path.exists(
                label_dir
            )
        else:
            alternative_patterns = [
                (
                    f"{args.code_path}/imagenet-llamagen-adm-{args.image_size}_codes",
                    f"{args.code_path}/imagenet-llamagen-adm-{args.image_size}_labels",
                ),
                (f"{args.code_path}_codes", f"{args.code_path}_labels"),
            ]
            for feat_dir, lab_dir in alternative_patterns:
                if os.path.exists(feat_dir) and os.path.exists(lab_dir):
                    feature_dir = feat_dir
                    label_dir = lab_dir
                    has_llamagen_format = True
                    break

    # If we have LlamaGen format, use CustomDataset
    if has_llamagen_format:
        # Verify it's actually flat format (has numbered .npy files)
        test_files = [f"{i}.npy" for i in range(5)]
        if all(os.path.exists(os.path.join(feature_dir, f)) for f in test_files):
            print(
                f"Loading LlamaGen flat format dataset from:\n  Features: {feature_dir}\n  Labels: {label_dir}"
            )
            return CustomDataset(feature_dir, label_dir)

    # Check if we have RandAR subdirectory format
    data_dir = args.code_path
    if data_dir.endswith("_codes"):
        # Path already points to the data directory
        pass
    elif os.path.exists(
        f"{args.code_path}/imagenet-llamagen-adm-{args.image_size}_codes"
    ):
        data_dir = f"{args.code_path}/imagenet-llamagen-adm-{args.image_size}_codes"

    # Check if this is subdirectory format
    if os.path.exists(data_dir):
        # Check if it has class subdirectories (0/, 1/, etc.)
        contents = os.listdir(data_dir)
        subdirs = [
            d
            for d in contents
            if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()
        ]
        if len(subdirs) > 0:
            print(
                f"Detected RandAR subdirectory format with {len(subdirs)} class directories"
            )
            return RandARStyleDataset(data_dir)

    # If we get here, we couldn't find the data
    raise FileNotFoundError(
        f"Could not find dataset in either format:\n\n"
        f"LlamaGen flat format (tried):\n"
        f"  Features: {feature_dir}\n"
        f"  Labels: {label_dir}\n\n"
        f"RandAR subdirectory format (tried):\n"
        f"  Data dir: {data_dir}\n\n"
        f"Current --code-path: {args.code_path}\n\n"
        f"Expected structures:\n"
        f"  LlamaGen: {{code_path}}/imagenet{args.image_size}_codes/{{0.npy, 1.npy, ...}}\n"
        f"  RandAR: {{code_path}}/{{0/, 1/, 2/, ...}}/{{samples.npy}}"
    )
