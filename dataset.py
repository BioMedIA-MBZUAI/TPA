import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from pathlib import Path
from random import randint

# import open_clip
import random


class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        video_dir,
        split="train",
        num_classes=2,
        model_type="video",
        pair_frames_with_text=False,
        num_prompts_per_class=1,
    ):
        self.split = split
        self.video_dir = Path(video_dir)
        self.num_classes = num_classes
        self.model_type = model_type
        self._validate_inputs(df)
        self.pair_frames_with_text = pair_frames_with_text
        self.total_prompts = num_classes * num_prompts_per_class
        if self.split == "train":
            self.num_prompts_per_class = num_prompts_per_class
        else:
            self.num_prompts_per_class = 1

        self.total_prompts = self.num_classes * self.num_prompts_per_class
        self.multiclips_training = False
        self.transform = None
        # Setup data and transforms
        self._setup_data(df)
        self._setup_transforms()

    def _validate_inputs(self, df):
        """Validate all input parameters"""
        if not self.video_dir.exists():
            raise ValueError(f"Video directory {self.video_dir} does not exist")

        if not isinstance(self.num_classes, int) or self.num_classes < 2:
            raise ValueError(
                f"num_classes must be an integer >= 2, got {self.num_classes}"
            )

        if "label" not in df.columns:
            raise ValueError("DataFrame must contain 'label' column")

        if not all(label < self.num_classes for label in df["label"]):
            raise ValueError("Labels in dataframe exceed num_classes")

    def _setup_data(self, df):
        """Setup dataset data"""
        self.data = df.copy()
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645])
        self.std = torch.tensor([0.22803, 0.22145, 0.216989])

        # Extract video name from path
        self.data["videoname"] = self.data["video"].apply(
            lambda x: (
                Path(x).name[Path(x).name.find("REQ") :]
                if "REQ" in Path(x).name
                else Path(x).name
            )
        )

    def _setup_transforms(self):
        """Setup data transforms based on split and model type"""
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645])
        self.std = torch.tensor([0.22803, 0.22145, 0.216989])

        if self.model_type == "video":
            if self.split == "train":
                self.transform = transforms.Compose(
                    [
                        transforms.RandomRotation(degrees=15),
                        transforms.RandomResizedCrop(
                            224, scale=(0.8, 1.0), ratio=(1.0, 1.0), antialias=True
                        ),
                        transforms.Lambda(
                            lambda x: x / 255.0 if torch.max(x) > 1 else x
                        ),
                        transforms.Normalize(mean=self.mean, std=self.std),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.Resize((224, 224), antialias=True),
                        transforms.Lambda(
                            lambda x: x / 255.0 if torch.max(x) > 1 else x
                        ),
                        transforms.Normalize(mean=self.mean, std=self.std),
                    ]
                )
        elif self.model_type == "foundation":
            normalize_mean = (0.48145466, 0.4578275, 0.40821073)
            normalize_std = (0.26862954, 0.26130258, 0.27577711)
            if self.split == "train":
                self.transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            224,
                            scale=(0.8, 1.0),
                            ratio=(1.0, 1.0),
                            interpolation=InterpolationMode.BICUBIC,
                            antialias=True,
                        ),
                        transforms.ToTensor(),
                        transforms.Normalize(normalize_mean, normalize_std),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.Resize(
                            224, interpolation=InterpolationMode.BICUBIC, antialias=True
                        ),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(normalize_mean, normalize_std),
                    ]
                )

    def _tensor_to_pil(self, tensor):
        """Convert a tensor to PIL Image"""
        if tensor.max() <= 1.0:
            tensor = tensor * 255
        tensor = tensor.byte()
        return transforms.ToPILImage()(tensor)

    def _resolve_video_tensor(self, loaded, video_path):
        """Extract the underlying video tensor from the stored object."""
        if isinstance(loaded, dict):
            candidate_keys = ["video", "frames", "tensor", "data"]
            video = None
            for key in candidate_keys:
                value = loaded.get(key)
                if value is None:
                    continue
                value_tensor = torch.as_tensor(value)
                if value_tensor.ndim >= 3:
                    video = value_tensor
                    break
            if video is None:
                for value in loaded.values():
                    value_tensor = torch.as_tensor(value)
                    if value_tensor.ndim >= 3:
                        video = value_tensor
                        break
            if video is None:
                raise TypeError(
                    f"Loaded dictionary from {video_path} but none of the entries is a tensor-like video"
                )
        elif torch.is_tensor(loaded):
            video = loaded
        else:
            try:
                video = torch.as_tensor(loaded)
            except Exception as conversion_error:
                raise TypeError(
                    f"Unsupported data type {type(loaded)} when loading {video_path}"
                ) from conversion_error
        return self._standardize_video_tensor(video, video_path)

    def _standardize_video_tensor(self, video, video_path):
        """Return video shaped as [T, C, H, W]."""
        if not torch.is_tensor(video):
            video = torch.as_tensor(video)

        if video.ndim == 4:
            if video.shape[1] <= 4 and video.shape[0] > 4:
                # Already [T, C, H, W]
                pass
            elif video.shape[0] <= 4 and video.shape[1] > 4:
                # Likely [C, T, H, W]
                video = video.permute(1, 0, 2, 3)
            elif video.shape[-1] <= 4:
                # Likely [T, H, W, C]
                video = video.permute(0, 3, 1, 2)
            else:
                raise ValueError(
                    f"Unable to infer channel dimension for tensor from {video_path} with shape {tuple(video.shape)}"
                )
        elif video.ndim == 3:
            # Assume [T, H, W] (single channel)
            video = video.unsqueeze(1)
        else:
            raise ValueError(
                f"Unsupported tensor shape {tuple(video.shape)} when loading {video_path}"
            )

        return video.contiguous().float()

    def __len__(self):
        return len(self.data)

    def sliding_window(self, video, window_size=16, stride=8, num_clips=3):
        """Generate a fixed number of clips using sliding window."""
        total_frames = video.shape[0]
        clips = []
        selected_frames = []
        if total_frames <= window_size:
            # If video is shorter than window_size, repeat the video
            repeats = (window_size + total_frames - 1) // total_frames
            video = video.repeat(repeats, 1, 1, 1)[:window_size]
            clips = [video] * num_clips
            selected_frames = [list(range(window_size))] * num_clips
        else:
            # Calculate possible start indices
            possible_starts = list(range(0, total_frames - window_size + 1, stride))

            if len(possible_starts) < num_clips:
                # If we have fewer possible clips than desired, repeat some
                possible_starts = (
                    possible_starts
                    * ((num_clips + len(possible_starts) - 1) // len(possible_starts))
                )[:num_clips]
            else:
                # If we have more possible clips than desired, sample evenly
                indices = torch.linspace(0, len(possible_starts) - 1, num_clips).long()
                possible_starts = [possible_starts[i] for i in indices]

            # Extract clips
            for start in possible_starts:
                clip = video[start : start + window_size]
                clips.append(clip)
                selected_frames.append(list(range(start, start + window_size)))
        return torch.stack(clips), selected_frames  # Shape: [num_clips, T, C, H, W]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_name = row["videoname"]
        label = row["label"]

        # Load tensor
        tensor_name = video_name.replace(".mp4", ".pt")
        video_path = self.video_dir / tensor_name

        try:
            loaded = torch.load(video_path)
            video = self._resolve_video_tensor(loaded, video_path)

            if self.split == "train":
                if self.multiclips_training:
                    video, selected_frames = self.sliding_window(
                        video, window_size=16, stride=8, num_clips=3
                    )
                else:
                    start = randint(0, video.shape[0] - 16)
                    video = video[start : start + 16, ...]
                    selected_frames = [list(range(start, start + 16))]
            else:
                video, selected_frames = self.sliding_window(
                    video, window_size=16, stride=8, num_clips=3
                )

            if self.transform:
                if self.model_type == "foundation":
                    if self.split == "train" and not self.multiclips_training:
                        frame_tensors = []
                        for t in range(video.size(0)):
                            pil_img = self._tensor_to_pil(video[t])
                            transformed_img = self.transform(pil_img)
                            frame_tensors.append(transformed_img)
                        video = torch.stack(frame_tensors)
                    else:
                        transformed_clips = []
                        for clip in video:
                            frame_tensors = []
                            for t in range(clip.size(0)):
                                pil_img = self._tensor_to_pil(clip[t])
                                transformed_img = self.transform(pil_img)
                                frame_tensors.append(transformed_img)
                            transformed_clips.append(torch.stack(frame_tensors))
                        video = torch.stack(transformed_clips)
                else:
                    if self.split == "train":
                        video = self.transform(video)
                    else:
                        transformed_clips = []
                        for clip in video:
                            transformed_clips.append(self.transform(clip))
                        video = torch.stack(transformed_clips)

            if self.split == "train":
                video = video.permute((1, 0, 2, 3))  # [C, T, H, W]
            else:
                video = video.permute((0, 2, 1, 3, 4))  # [num_clips, C, T, H, W]

            if self.pair_frames_with_text:
                if self.split == "train":
                    # print(f"num_prompts_per_class:{self.num_prompts_per_class-1}")
                    pos_idx = random.randint(0, self.num_prompts_per_class - 1)
                    pos_text_idx = label * self.num_prompts_per_class + pos_idx

                    neg_class = random.choice(
                        [c for c in range(self.num_classes) if c != label]
                    )
                    neg_idx = random.randint(0, self.num_prompts_per_class - 1)
                    neg_text_idx = neg_class * self.num_prompts_per_class + neg_idx

                    return video, label, 1, video_name, pos_text_idx, neg_text_idx
                elif self.split == "val":
                    pos_text_idx = label
                    neg_class = random.choice(
                        [c for c in range(self.num_classes) if c != label]
                    )
                    neg_text_idx = neg_class

                    return video, label, 1, video_name, pos_text_idx, neg_text_idx

            return video, label, None, video_name, None, None

        except Exception as e:
            print(f"Failed to load {video_path}")
            print(f"Original video name: {row['video']}")
            print(f"Processed video name: {video_name}")
            raise RuntimeError(f"Error loading {video_path}: {str(e)}")


def custom_collate(batch):
    """Custom collate function to handle variable number of clips."""
    if len(batch[0]) == 7:
        videos, true_labels, labels, names, _, text_idx, neg_idx = zip(*batch)
    elif len(batch[0]) == 6:
        videos, true_labels, labels, names, text_idx, neg_idx = zip(*batch)
    else:
        raise ValueError(f"Unexpected sample structure with {len(batch[0])} elements")

    is_multiclip = isinstance(videos[0], torch.Tensor) and len(videos[0].shape) == 5

    if is_multiclip:
        all_clips = torch.cat(videos, dim=0)
        true_labels = torch.tensor(
            [
                true_label
                for true_label, video in zip(true_labels, videos)
                for _ in range(video.shape[0])
            ]
        )
        if labels[0] is not None:
            labels = torch.tensor(
                [
                    label
                    for label, video in zip(labels, videos)
                    for _ in range(video.shape[0])
                ]
            )
        else:
            labels = None
        if text_idx[0] is not None:
            text_idx = torch.tensor(
                [
                    idx
                    for idx, video in zip(text_idx, videos)
                    for _ in range(video.shape[0])
                ]
            )
        else:
            text_idx = None
        if neg_idx[0] is not None:
            neg_idx = torch.tensor(
                [
                    idx
                    for idx, video in zip(neg_idx, videos)
                    for _ in range(video.shape[0])
                ]
            )
        else:
            neg_idx = None
        return all_clips, true_labels, labels, names, text_idx, neg_idx

    # Single clip batches
    videos = torch.stack(videos)
    true_labels = torch.tensor(true_labels)
    labels = torch.tensor(labels) if labels[0] is not None else None
    text_idx = torch.tensor(text_idx) if text_idx[0] is not None else None
    neg_idx = torch.tensor(neg_idx) if neg_idx[0] is not None else None

    return videos, true_labels, labels, names, text_idx, neg_idx
