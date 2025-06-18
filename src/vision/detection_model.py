#!/usr/bin/env python3
"""
YOLOv8 Detection Model for HiveMind Vision Project
Optimized for real-time drone swarm object detection and consensus sharing.

Features:
- YOLOv8 fine-tuned on VisDrone dataset
- Edge-optimized inference for Jetson hardware
- Feature extraction for consensus algorithms
- Real-time processing with confidence filtering
- TensorRT optimization support
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import time
import logging
from dataclasses import dataclass
import onnx
import tensorrt as trt
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Single object detection result."""
    bbox: List[float]  # [x1, y1, x2, y2] in pixel coordinates
    confidence: float
    class_id: int
    class_name: str
    features: Optional[np.ndarray] = None  # Deep features for consensus
    track_id: Optional[int] = None  # For tracking integration

@dataclass
class DetectionResult:
    """Complete detection result for a frame."""
    detections: List[Detection]
    inference_time: float
    image_shape: Tuple[int, int]  # (height, width)
    timestamp: float
    drone_id: Optional[str] = None

class HiveMindYOLO(nn.Module):
    """
    YOLOv8 wrapper optimized for HiveMind drone swarm detection.
    
    Features:
    - VisDrone-specific class mapping
    - Feature extraction for consensus
    - Edge device optimization
    - Real-time inference pipeline
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'auto',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        enable_feature_extraction: bool = True
    ):
        """
        Initialize HiveMind YOLO model.
        
        Args:
            model_path: Path to trained model weights
            device: Device for inference ('auto', 'cpu', 'cuda', 'mps')
            conf_threshold: Confidence threshold for filtering detections
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections per image
            enable_feature_extraction: Extract deep features for consensus
        """
        super().__init__()
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.enable_feature_extraction = enable_feature_extraction
        
        # VisDrone class mapping
        self.class_names = {
            0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van',
            5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'
        }
        self.num_classes = len(self.class_names)
        
        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        # Load model
        self._load_model(model_path)
        
        # Performance tracking
        self.inference_times = []
        self.total_detections = 0
        
        logger.info(f"HiveMind YOLO initialized on {self.device}")
        logger.info(f"Classes: {list(self.class_names.values())}")
    
    def _load_model(self, model_path: Optional[str]):
        """Load YOLOv8 model with VisDrone configuration."""
        if model_path and Path(model_path).exists():
            # Load custom trained model
            self.model = YOLO(model_path)
            logger.info(f"Loaded custom model from {model_path}")
        else:
            # Start with pretrained YOLOv8n for drone applications
            self.model = YOLO('yolov8n.pt')
            logger.info("Loaded YOLOv8n pretrained model")
            logger.warning("Consider training on VisDrone dataset for optimal performance")
        
        # Configure model
        self.model.to(self.device)
        self.model.conf = self.conf_threshold
        self.model.iou = self.iou_threshold
        self.model.max_det = self.max_detections
        
        # Enable feature extraction if requested
        if self.enable_feature_extraction:
            self._setup_feature_extraction()
    
    def _setup_feature_extraction(self):
        """Setup hooks for deep feature extraction."""
        self.features = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                # Store features from backbone
                if isinstance(output, torch.Tensor):
                    self.features[name] = output.detach()
            return hook
        
        # Register hooks on backbone layers for feature extraction
        # This will be used for consensus algorithms
        backbone = self.model.model.model[:-1]  # Exclude head
        for i, layer in enumerate(backbone):
            if i in [6, 8]:  # Extract from key layers
                layer.register_forward_hook(hook_fn(f'backbone_{i}'))
    
    def preprocess_image(
        self, 
        image: Union[np.ndarray, str, Path],
        target_size: int = 640
    ) -> Tuple[np.ndarray, Tuple[int, int], float]:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (numpy array, file path, or Path object)
            target_size: Target size for inference
            
        Returns:
            processed_image: Preprocessed image
            original_shape: Original (height, width)
            scale_factor: Scaling factor applied
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Already RGB, keep as is
            pass
        else:
            raise ValueError("Unsupported image format")
        
        original_shape = image.shape[:2]  # (height, width)
        
        # Resize while maintaining aspect ratio
        h, w = original_shape
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        
        processed = cv2.copyMakeBorder(
            resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        return processed, original_shape, scale
    
    def postprocess_detections(
        self,
        results,
        original_shape: Tuple[int, int],
        scale_factor: float,
        target_size: int = 640
    ) -> List[Detection]:
        """
        Postprocess YOLOv8 results to Detection objects.
        
        Args:
            results: YOLOv8 results object
            original_shape: Original image (height, width)
            scale_factor: Scale factor from preprocessing
            target_size: Target inference size
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        if len(results) == 0 or results[0].boxes is None:
            return detections
        
        boxes = results[0].boxes
        
        # Extract detection data
        if len(boxes) == 0:
            return detections
        
        # Get bounding boxes, confidences, and classes
        bboxes = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        
        # Calculate padding offset (assuming center padding)
        h_orig, w_orig = original_shape
        new_h, new_w = int(h_orig * scale_factor), int(w_orig * scale_factor)
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        top_pad, left_pad = pad_h // 2, pad_w // 2
        
        for i, (bbox, conf, cls_id) in enumerate(zip(bboxes, confidences, class_ids)):
            # Convert back to original image coordinates
            x1, y1, x2, y2 = bbox
            
            # Remove padding
            x1 = max(0, x1 - left_pad)
            y1 = max(0, y1 - top_pad)
            x2 = max(0, x2 - left_pad)
            y2 = max(0, y2 - top_pad)
            
            # Scale back to original size
            x1 = x1 / scale_factor
            y1 = y1 / scale_factor
            x2 = x2 / scale_factor
            y2 = y2 / scale_factor
            
            # Clip to image bounds
            x1 = max(0, min(x1, w_orig))
            y1 = max(0, min(y1, h_orig))
            x2 = max(0, min(x2, w_orig))
            y2 = max(0, min(y2, h_orig))
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Map class ID to VisDrone classes
            if cls_id < len(self.class_names):
                class_name = self.class_names[cls_id]
            else:
                class_name = f'unknown_{cls_id}'
            
            # Extract features if enabled
            features = None
            if self.enable_feature_extraction and self.features:
                features = self._extract_roi_features(bbox, scale_factor)
            
            detection = Detection(
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                confidence=float(conf),
                class_id=cls_id,
                class_name=class_name,
                features=features
            )
            
            detections.append(detection)
        
        return detections
    
    def _extract_roi_features(self, bbox: np.ndarray, scale_factor: float) -> np.ndarray:
        """Extract ROI features from backbone for consensus algorithms."""
        if not self.features:
            return None
        
        # Use features from a key backbone layer
        feature_map = None
        for name, feat in self.features.items():
            if 'backbone_6' in name:  # Use features from layer 6
                feature_map = feat
                break
        
        if feature_map is None:
            return None
        
        # ROI pooling on feature map
        # This is a simplified version - in practice you'd use proper ROI pooling
        try:
            b, c, h, w = feature_map.shape
            x1, y1, x2, y2 = bbox
            
            # Scale bbox to feature map coordinates
            feat_scale = h / 640  # Assuming 640px input
            fx1 = int(x1 * feat_scale * scale_factor)
            fy1 = int(y1 * feat_scale * scale_factor)
            fx2 = int(x2 * feat_scale * scale_factor)
            fy2 = int(y2 * feat_scale * scale_factor)
            
            # Extract and pool features
            roi_feat = feature_map[0, :, fy1:fy2, fx1:fx2]
            if roi_feat.numel() > 0:
                # Global average pooling
                pooled_feat = torch.mean(roi_feat, dim=[1, 2])
                return pooled_feat.cpu().numpy()
        except Exception as e:
            logger.debug(f"Feature extraction failed: {e}")
        
        return None
    
    def detect(
        self, 
        image: Union[np.ndarray, str, Path],
        drone_id: Optional[str] = None
    ) -> DetectionResult:
        """
        Run object detection on input image.
        
        Args:
            image: Input image
            drone_id: Optional drone identifier
            
        Returns:
            DetectionResult object with all detections
        """
        start_time = time.time()
        
        # Clear previous features
        if self.enable_feature_extraction:
            self.features.clear()
        
        # Preprocess
        processed_img, original_shape, scale_factor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            results = self.model(processed_img, verbose=False)
        
        # Postprocess
        detections = self.postprocess_detections(
            results, original_shape, scale_factor
        )
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.total_detections += len(detections)
        
        return DetectionResult(
            detections=detections,
            inference_time=inference_time,
            image_shape=original_shape,
            timestamp=time.time(),
            drone_id=drone_id
        )
    
    def detect_batch(
        self,
        images: List[Union[np.ndarray, str, Path]],
        drone_id: Optional[str] = None
    ) -> List[DetectionResult]:
        """
        Run batch detection for multiple images.
        
        Args:
            images: List of input images
            drone_id: Optional drone identifier
            
        Returns:
            List of DetectionResult objects
        """
        results = []
        
        for image in images:
            result = self.detect(image, drone_id)
            results.append(result)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'fps': 1.0 / np.mean(self.inference_times),
            'total_detections': self.total_detections,
            'avg_detections_per_frame': self.total_detections / len(self.inference_times)
        }
    
    def export_to_onnx(self, output_path: str, input_size: int = 640):
        """Export model to ONNX format for deployment."""
        try:
            success = self.model.export(
                format='onnx',
                imgsz=input_size,
                dynamic=False,
                simplify=True
            )
            if success:
                logger.info(f"Model exported to ONNX: {output_path}")
            return success
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False
    
    def export_to_tensorrt(self, output_path: str, input_size: int = 640):
        """Export model to TensorRT for optimized inference on Jetson."""
        try:
            success = self.model.export(
                format='engine',
                imgsz=input_size,
                device=self.device,
                half=True,  # FP16 for faster inference
                dynamic=False,
                simplify=True,
                workspace=4  # 4GB workspace
            )
            if success:
                logger.info(f"Model exported to TensorRT: {output_path}")
            return success
        except Exception as e:
            logger.error(f"TensorRT export failed: {e}")
            return False
    
    def train_on_visdrone(
        self,
        train_data_path: str,
        val_data_path: str,
        epochs: int = 100,
        batch_size: int = 16,
        output_dir: str = "runs/train"
    ) -> str:
        """
        Fine-tune model on VisDrone dataset.
        
        Args:
            train_data_path: Path to training dataset
            val_data_path: Path to validation dataset
            epochs: Number of training epochs
            batch_size: Training batch size
            output_dir: Output directory for training results
            
        Returns:
            Path to best trained model
        """
        logger.info("Starting VisDrone training...")
        
        # Create training configuration
        train_config = {
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': 640,
            'device': self.device,
            'workers': 8,
            'project': output_dir,
            'name': 'hivemind_yolo',
            'save_period': 10,
            'val': True,
            'plots': True,
            'verbose': True
        }
        
        try:
            # Start training
            results = self.model.train(
                data={
                    'train': train_data_path,
                    'val': val_data_path,
                    'nc': self.num_classes,
                    'names': list(self.class_names.values())
                },
                **train_config
            )
            
            best_model_path = results.save_dir / 'weights' / 'best.pt'
            logger.info(f"Training completed. Best model saved to: {best_model_path}")
            
            return str(best_model_path)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


def load_pretrained_hivemind_yolo(
    model_path: str,
    device: str = 'auto',
    **kwargs
) -> HiveMindYOLO:
    """
    Load a pretrained HiveMind YOLO model.
    
    Args:
        model_path: Path to trained model weights
        device: Device for inference
        **kwargs: Additional arguments for HiveMindYOLO
        
    Returns:
        Loaded HiveMindYOLO model
    """
    return HiveMindYOLO(model_path=model_path, device=device, **kwargs)


def create_yolo_for_edge_device(
    model_path: str,
    optimize_for_jetson: bool = True
) -> HiveMindYOLO:
    """
    Create YOLO model optimized for edge devices like Jetson Xavier.
    
    Args:
        model_path: Path to model weights
        optimize_for_jetson: Apply Jetson-specific optimizations
        
    Returns:
        Edge-optimized HiveMindYOLO model
    """
    config = {
        'conf_threshold': 0.3,  # Slightly higher threshold for edge
        'iou_threshold': 0.5,
        'max_detections': 50,  # Fewer detections for faster processing
        'enable_feature_extraction': True
    }
    
    if optimize_for_jetson:
        config.update({
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        })
    
    model = HiveMindYOLO(model_path=model_path, **config)
    
    logger.info("Created edge-optimized YOLO model")
    return model


# Example usage and testing
if __name__ == "__main__":
    # Test basic detection
    print("🚁 Testing HiveMind YOLO Detection Model")
    
    # Initialize model
    model = HiveMindYOLO(
        conf_threshold=0.25,
        enable_feature_extraction=True
    )
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Run detection
    result = model.detect(dummy_image, drone_id="drone_001")
    
    print(f"Detections: {len(result.detections)}")
    print(f"Inference time: {result.inference_time:.3f}s")
    print(f"FPS: {1/result.inference_time:.1f}")
    
    # Print detection details
    for i, det in enumerate(result.detections):
        print(f"Detection {i}: {det.class_name} ({det.confidence:.2f}) at {det.bbox}")
        if det.features is not None:
            print(f"  Features shape: {det.features.shape}")
    
    # Performance stats
    stats = model.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    print("✅ HiveMind YOLO test completed!")