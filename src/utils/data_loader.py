#!/usr/bin/env python3
"""
VisDrone Dataset Loader for HiveMind Vision Project
Handles both detection (DET) and tracking (MOT) datasets with proper preprocessing.

Usage:
    # Detection dataset
    det_dataset = VisDroneDetection('data/datasets/visdrone/VisDrone2019-DET-train')
    det_loader = DataLoader(det_dataset, batch_size=32, shuffle=True)
    
    # Tracking dataset  
    mot_dataset = VisDroneTracking('data/datasets/visdrone/VisDrone2019-MOT-train')
    mot_loader = DataLoader(mot_dataset, batch_size=8, shuffle=True)
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisDroneDetection(Dataset):
    """
    VisDrone Detection Dataset for object detection training/inference.
    
    VisDrone annotation format:
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    
    Classes:
    0: ignored, 1: pedestrian, 2: people, 3: bicycle, 4: car, 5: van, 
    6: truck, 7: tricycle, 8: awning-tricycle, 9: bus, 10: motor
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[A.Compose] = None,
        filter_empty: bool = True,
        min_bbox_area: int = 100,
        exclude_ignored: bool = True
    ):
        """
        Args:
            data_dir: Path to VisDrone dataset directory (contains images/ and annotations/)
            transform: Albumentations transform pipeline
            filter_empty: Remove images with no valid annotations
            min_bbox_area: Minimum bounding box area in pixels
            exclude_ignored: Exclude class 0 (ignored) annotations
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'images'
        self.annotations_dir = self.data_dir / 'annotations'
        
        if not self.images_dir.exists() or not self.annotations_dir.exists():
            raise FileNotFoundError(f"Images or annotations directory not found in {data_dir}")
        
        self.filter_empty = filter_empty
        self.min_bbox_area = min_bbox_area
        self.exclude_ignored = exclude_ignored
        
        # VisDrone class mapping (excluding ignored class 0)
        self.class_names = {
            1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van',
            6: 'truck', 7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor'
        }
        
        # Map original classes to 0-based indices for training
        self.class_mapping = {k: i for i, k in enumerate(self.class_names.keys())}
        self.num_classes = len(self.class_names)
        
        # Default transform if none provided
        if transform is None:
            self.transform = A.Compose([
                A.Resize(640, 640),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels']
            ))
        else:
            self.transform = transform
        
        # Load and validate dataset
        self._load_dataset()
        
        logger.info(f"VisDrone Detection Dataset loaded: {len(self.samples)} samples")
        logger.info(f"Classes: {list(self.class_names.values())}")
    
    def _load_dataset(self):
        """Load and validate all image-annotation pairs."""
        self.samples = []
        
        # Get all image files
        image_files = list(self.images_dir.glob('*.jpg'))
        
        for img_path in image_files:
            ann_path = self.annotations_dir / (img_path.stem + '.txt')
            
            if not ann_path.exists():
                logger.warning(f"Annotation not found for {img_path.name}")
                continue
            
            # Parse annotations
            bboxes, labels = self._parse_annotation(ann_path)
            
            # Filter empty images if requested
            if self.filter_empty and len(bboxes) == 0:
                continue
            
            self.samples.append({
                'image_path': img_path,
                'annotation_path': ann_path,
                'bboxes': bboxes,
                'labels': labels
            })
    
    def _parse_annotation(self, ann_path: Path) -> Tuple[List[List[float]], List[int]]:
        """
        Parse VisDrone annotation file.
        
        Returns:
            bboxes: List of [x1, y1, x2, y2] in Pascal VOC format
            labels: List of class indices (0-based)
        """
        bboxes = []
        labels = []
        
        try:
            with open(ann_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 8:
                        continue
                    
                    # Parse VisDrone format: left, top, width, height, score, class, truncation, occlusion
                    left, top, width, height = map(int, parts[:4])
                    score = float(parts[4])
                    class_id = int(parts[5])
                    truncation = int(parts[6])
                    occlusion = int(parts[7])
                    
                    # Filter by conditions
                    if self.exclude_ignored and class_id == 0:
                        continue
                    
                    if class_id not in self.class_names:
                        continue
                    
                    # Calculate bbox area
                    if width * height < self.min_bbox_area:
                        continue
                    
                    # Convert to Pascal VOC format (x1, y1, x2, y2)
                    x1, y1 = left, top
                    x2, y2 = left + width, top + height
                    
                    # Map to 0-based class index
                    class_idx = self.class_mapping[class_id]
                    
                    bboxes.append([x1, y1, x2, y2])
                    labels.append(class_idx)
        
        except Exception as e:
            logger.warning(f"Error parsing annotation {ann_path}: {e}")
            return [], []
        
        return bboxes, labels
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Returns:
            Dict containing:
                - image: Tensor of shape (C, H, W)
                - bboxes: Tensor of shape (N, 4) in Pascal VOC format
                - labels: Tensor of shape (N,) with class indices
                - image_id: Original image filename
        """
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(sample['image_path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        bboxes = sample['bboxes'].copy()
        labels = sample['labels'].copy()
        
        # Apply transforms
        if self.transform and len(bboxes) > 0:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=labels
            )
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['class_labels']
        elif self.transform:
            # Image only transform for empty annotations
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image = transformed['image']
            bboxes = []
            labels = []
        
        # Convert to tensors
        if len(bboxes) > 0:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        
        return {
            'image': image,
            'bboxes': bboxes,
            'labels': labels,
            'image_id': sample['image_path'].stem
        }


class VisDroneTracking(Dataset):
    """
    VisDrone Multi-Object Tracking Dataset for sequence-based training.
    
    MOT annotation format:
    <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
    """
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 8,
        transform: Optional[A.Compose] = None,
        stride: int = 1,
        load_gt: bool = True
    ):
        """
        Args:
            data_dir: Path to MOT dataset directory (contains sequences/ and annotations/)
            sequence_length: Number of consecutive frames per sample
            transform: Albumentations transform pipeline
            stride: Frame sampling stride
            load_gt: Whether to load ground truth annotations
        """
        self.data_dir = Path(data_dir)
        self.sequences_dir = self.data_dir / 'sequences'
        self.annotations_dir = self.data_dir / 'annotations'
        
        if not self.sequences_dir.exists():
            raise FileNotFoundError(f"Sequences directory not found in {data_dir}")
        
        self.sequence_length = sequence_length
        self.stride = stride
        self.load_gt = load_gt
        
        # Class mapping same as detection
        self.class_names = {
            1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van',
            6: 'truck', 7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor'
        }
        self.class_mapping = {k: i for i, k in enumerate(self.class_names.keys())}
        
        # Default transform
        if transform is None:
            self.transform = A.Compose([
                A.Resize(640, 640),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = transform
        
        # Load sequences
        self._load_sequences()
        
        logger.info(f"VisDrone Tracking Dataset loaded: {len(self.samples)} sequence samples")
        logger.info(f"Total sequences: {len(self.sequences)}")
    
    def _load_sequences(self):
        """Load all available sequences and create frame samples."""
        self.sequences = {}
        self.samples = []
        
        # Get all sequence directories
        sequence_dirs = [d for d in self.sequences_dir.iterdir() if d.is_dir()]
        
        for seq_dir in sequence_dirs:
            seq_name = seq_dir.name
            
            # Get all frame images
            frame_files = sorted(list(seq_dir.glob('*.jpg')))
            if len(frame_files) == 0:
                continue
            
            # Load ground truth if available
            gt_data = {}
            if self.load_gt and self.annotations_dir.exists():
                gt_file = self.annotations_dir / f"{seq_name}.txt"
                if gt_file.exists():
                    gt_data = self._parse_mot_annotation(gt_file)
            
            self.sequences[seq_name] = {
                'frames': frame_files,
                'gt': gt_data
            }
            
            # Create sequence samples
            num_frames = len(frame_files)
            for start_idx in range(0, num_frames - self.sequence_length + 1, self.stride):
                self.samples.append({
                    'sequence': seq_name,
                    'start_frame': start_idx,
                    'end_frame': start_idx + self.sequence_length
                })
    
    def _parse_mot_annotation(self, gt_file: Path) -> Dict[int, List]:
        """
        Parse MOT format ground truth file.
        
        Returns:
            Dict mapping frame_id to list of detections
        """
        gt_data = {}
        
        try:
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 10:
                        continue
                    
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    left, top, width, height = map(float, parts[2:6])
                    conf = float(parts[6])
                    
                    # Convert to Pascal VOC format
                    x1, y1 = left, top
                    x2, y2 = left + width, top + height
                    
                    if frame_id not in gt_data:
                        gt_data[frame_id] = []
                    
                    gt_data[frame_id].append({
                        'track_id': track_id,
                        'bbox': [x1, y1, x2, y2],
                        'conf': conf
                    })
        
        except Exception as e:
            logger.warning(f"Error parsing MOT annotation {gt_file}: {e}")
        
        return gt_data
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sequence sample.
        
        Returns:
            Dict containing:
                - frames: Tensor of shape (T, C, H, W) where T is sequence_length
                - gt_bboxes: List of tensors, one per frame (N_t, 4)
                - gt_track_ids: List of tensors, one per frame (N_t,)
                - sequence_id: Sequence name
                - frame_ids: List of frame indices
        """
        sample = self.samples[idx]
        seq_name = sample['sequence']
        start_frame = sample['start_frame']
        end_frame = sample['end_frame']
        
        sequence = self.sequences[seq_name]
        frame_files = sequence['frames'][start_frame:end_frame]
        gt_data = sequence['gt']
        
        frames = []
        gt_bboxes = []
        gt_track_ids = []
        frame_ids = []
        
        for i, frame_file in enumerate(frame_files):
            # Load frame
            image = cv2.imread(str(frame_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transform
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            frames.append(image)
            
            # Get ground truth for this frame
            frame_idx = start_frame + i + 1  # MOT frames are 1-indexed
            frame_ids.append(frame_idx)
            
            if frame_idx in gt_data:
                bboxes = []
                track_ids = []
                for detection in gt_data[frame_idx]:
                    bboxes.append(detection['bbox'])
                    track_ids.append(detection['track_id'])
                
                gt_bboxes.append(torch.tensor(bboxes, dtype=torch.float32))
                gt_track_ids.append(torch.tensor(track_ids, dtype=torch.long))
            else:
                gt_bboxes.append(torch.zeros((0, 4), dtype=torch.float32))
                gt_track_ids.append(torch.zeros((0,), dtype=torch.long))
        
        # Stack frames
        frames = torch.stack(frames)
        
        return {
            'frames': frames,
            'gt_bboxes': gt_bboxes,
            'gt_track_ids': gt_track_ids,
            'sequence_id': seq_name,
            'frame_ids': frame_ids
        }


def create_detection_transforms(image_size: int = 640, train: bool = True) -> A.Compose:
    """Create detection training/validation transforms."""
    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels']
        ))


def create_tracking_transforms(image_size: int = 640, train: bool = True) -> A.Compose:
    """Create tracking training/validation transforms."""
    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.3),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def detection_collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """Custom collate function for detection dataset with variable number of objects."""
    images = torch.stack([item['image'] for item in batch])
    
    # Keep bboxes and labels as lists since they have variable lengths
    bboxes = [item['bboxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    
    return {
        'images': images,
        'bboxes': bboxes,
        'labels': labels,
        'image_ids': image_ids
    }


def tracking_collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """Custom collate function for tracking dataset."""
    frames = torch.stack([item['frames'] for item in batch])
    
    # Keep variable-length data as lists
    gt_bboxes = [item['gt_bboxes'] for item in batch]
    gt_track_ids = [item['gt_track_ids'] for item in batch]
    sequence_ids = [item['sequence_id'] for item in batch]
    frame_ids = [item['frame_ids'] for item in batch]
    
    return {
        'frames': frames,
        'gt_bboxes': gt_bboxes,
        'gt_track_ids': gt_track_ids,
        'sequence_ids': sequence_ids,
        'frame_ids': frame_ids
    }


# Example usage and testing
if __name__ == "__main__":
    # Test detection dataset
    try:
        det_dataset = VisDroneDetection(
            'data/datasets/visdrone/VisDrone2019-DET-train',
            transform=create_detection_transforms(train=True)
        )
        det_loader = DataLoader(
            det_dataset, 
            batch_size=4, 
            shuffle=True, 
            collate_fn=detection_collate_fn,
            num_workers=2
        )
        
        print(f"Detection dataset: {len(det_dataset)} samples")
        print(f"Classes: {det_dataset.class_names}")
        
        # Test one batch
        batch = next(iter(det_loader))
        print(f"Batch images shape: {batch['images'].shape}")
        print(f"Number of objects in batch: {[len(bbox) for bbox in batch['bboxes']]}")
        
    except Exception as e:
        print(f"Detection dataset test failed: {e}")
    
    # Test tracking dataset
    try:
        mot_dataset = VisDroneTracking(
            'data/datasets/visdrone/VisDrone2019-MOT-train',
            sequence_length=5,
            transform=create_tracking_transforms(train=True)
        )
        mot_loader = DataLoader(
            mot_dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=tracking_collate_fn,
            num_workers=2
        )
        
        print(f"Tracking dataset: {len(mot_dataset)} sequence samples")
        
        # Test one batch
        batch = next(iter(mot_loader))
        print(f"Batch frames shape: {batch['frames'].shape}")
        print(f"Sequence IDs: {batch['sequence_ids']}")
        
    except Exception as e:
        print(f"Tracking dataset test failed: {e}")