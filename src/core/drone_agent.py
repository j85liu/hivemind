#!/usr/bin/env python3
"""
Drone Agent Core for HiveMind Vision Project
Individual drone processing unit that handles local detection, communication, and consensus.

Features:
- Real-time camera processing with YOLOv8
- Mesh network communication with other drones
- Local state management and tracking
- Consensus participation for target validation
- Performance monitoring and health checks
"""

import asyncio
import time
import threading
import queue
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import logging
from pathlib import Path
import json
import uuid
from collections import defaultdict, deque

# Import our detection model
from detection_model import HiveMindYOLO, Detection, DetectionResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DroneState(Enum):
    """Drone operational states."""
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    CONSENSUS_VOTING = "consensus_voting"
    TARGET_CONFIRMED = "target_confirmed"
    ERROR = "error"
    OFFLINE = "offline"

@dataclass
class DronePosition:
    """3D position and orientation of drone."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0  # Altitude
    yaw: float = 0.0  # Heading in degrees
    pitch: float = 0.0
    roll: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class CameraFrame:
    """Camera frame with metadata."""
    image: np.ndarray
    timestamp: float
    frame_id: int
    position: Optional[DronePosition] = None
    camera_params: Optional[Dict] = None

@dataclass
class TargetReport:
    """Target detection report for consensus."""
    target_id: str
    detection: Detection
    drone_id: str
    position: DronePosition
    confidence_score: float
    timestamp: float
    consensus_votes: Dict[str, float] = field(default_factory=dict)
    confirmed: bool = False

class DroneAgent:
    """
    Individual drone agent for HiveMind swarm system.
    
    Handles:
    - Local object detection
    - Communication with other drones
    - Consensus participation
    - State management
    - Performance monitoring
    """
    
    def __init__(
        self,
        drone_id: str,
        detection_model_path: Optional[str] = None,
        camera_source: int = 0,
        detection_threshold: float = 0.3,
        consensus_threshold: float = 0.7,
        max_detection_age: float = 5.0,
        position: Optional[DronePosition] = None
    ):
        """
        Initialize drone agent.
        
        Args:
            drone_id: Unique identifier for this drone
            detection_model_path: Path to trained detection model
            camera_source: Camera source (0 for default, or video path)
            detection_threshold: Minimum confidence for local detections
            consensus_threshold: Minimum consensus score for target confirmation
            max_detection_age: Maximum age of detections in seconds
            position: Initial drone position
        """
        self.drone_id = drone_id
        self.state = DroneState.INITIALIZING
        self.position = position or DronePosition()
        
        # Detection configuration
        self.detection_threshold = detection_threshold
        self.consensus_threshold = consensus_threshold
        self.max_detection_age = max_detection_age
        
        # Initialize detection model
        self.detector = HiveMindYOLO(
            model_path=detection_model_path,
            conf_threshold=detection_threshold,
            enable_feature_extraction=True
        )
        
        # Camera setup
        self.camera_source = camera_source
        self.camera = None
        self.camera_active = False
        
        # Processing threads and queues
        self.frame_queue = queue.Queue(maxsize=10)
        self.detection_queue = queue.Queue(maxsize=50)
        self.communication_queue = queue.Queue(maxsize=100)
        
        # State tracking
        self.local_detections = {}  # target_id -> TargetReport
        self.peer_detections = {}   # drone_id -> List[TargetReport]
        self.confirmed_targets = {}  # target_id -> TargetReport
        self.frame_counter = 0
        
        # Performance tracking
        self.stats = {
            'frames_processed': 0,
            'detections_made': 0,
            'targets_confirmed': 0,
            'consensus_participations': 0,
            'avg_processing_time': 0.0,
            'last_update': time.time()
        }
        
        # Threading control
        self.running = False
        self.threads = []
        
        # Callbacks for external integration
        self.callbacks = {
            'on_target_confirmed': [],
            'on_detection': [],
            'on_state_change': [],
            'on_error': []
        }
        
        logger.info(f"Drone Agent {self.drone_id} initialized")
    
    def add_callback(self, event: str, callback: Callable):
        """Add callback for drone events."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _trigger_callback(self, event: str, *args, **kwargs):
        """Trigger registered callbacks for an event."""
        for callback in self.callbacks.get(event, []):
            try:
                callback(self, *args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
    
    def start(self):
        """Start drone agent processing."""
        if self.running:
            logger.warning(f"Drone {self.drone_id} already running")
            return
        
        logger.info(f"Starting drone agent {self.drone_id}")
        self.running = True
        
        # Initialize camera
        self._setup_camera()
        
        # Start processing threads
        self.threads = [
            threading.Thread(target=self._camera_thread, daemon=True),
            threading.Thread(target=self._detection_thread, daemon=True),
            threading.Thread(target=self._consensus_thread, daemon=True),
            threading.Thread(target=self._communication_thread, daemon=True),
            threading.Thread(target=self._health_monitor_thread, daemon=True)
        ]
        
        for thread in self.threads:
            thread.start()
        
        self.state = DroneState.READY
        self._trigger_callback('on_state_change', self.state)
        
        logger.info(f"Drone agent {self.drone_id} started successfully")
    
    def stop(self):
        """Stop drone agent processing."""
        logger.info(f"Stopping drone agent {self.drone_id}")
        self.running = False
        
        # Close camera
        if self.camera:
            self.camera.release()
            self.camera_active = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        self.state = DroneState.OFFLINE
        self._trigger_callback('on_state_change', self.state)
        
        logger.info(f"Drone agent {self.drone_id} stopped")
    
    def _setup_camera(self):
        """Initialize camera capture."""
        try:
            self.camera = cv2.VideoCapture(self.camera_source)
            if not self.camera.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_source}")
            
            # Set camera properties for optimal performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            self.camera_active = True
            logger.info(f"Camera initialized for drone {self.drone_id}")
            
        except Exception as e:
            logger.error(f"Camera setup failed for drone {self.drone_id}: {e}")
            self.state = DroneState.ERROR
            self._trigger_callback('on_error', f"Camera setup failed: {e}")
    
    def _camera_thread(self):
        """Camera capture thread."""
        while self.running and self.camera_active:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning(f"Failed to read frame from camera for drone {self.drone_id}")
                    continue
                
                # Create camera frame object
                camera_frame = CameraFrame(
                    image=frame,
                    timestamp=time.time(),
                    frame_id=self.frame_counter,
                    position=self.position
                )
                
                # Add to processing queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(camera_frame)
                    self.frame_counter += 1
                except queue.Full:
                    # Drop oldest frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(camera_frame)
                    except queue.Empty:
                        pass
                
                time.sleep(0.01)  # Small delay to prevent CPU overload
                
            except Exception as e:
                logger.error(f"Camera thread error for drone {self.drone_id}: {e}")
                self._trigger_callback('on_error', f"Camera error: {e}")
    
    def _detection_thread(self):
        """Object detection processing thread."""
        while self.running:
            try:
                # Get frame from queue
                camera_frame = self.frame_queue.get(timeout=1.0)
                
                start_time = time.time()
                
                # Run detection
                detection_result = self.detector.detect(
                    camera_frame.image,
                    drone_id=self.drone_id
                )
                
                # Process detections
                self._process_detections(detection_result, camera_frame)
                
                # Update performance stats
                processing_time = time.time() - start_time
                self._update_processing_stats(processing_time)
                
                # Update state
                if self.state == DroneState.READY and len(detection_result.detections) > 0:
                    self.state = DroneState.ACTIVE
                    self._trigger_callback('on_state_change', self.state)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Detection thread error for drone {self.drone_id}: {e}")
                self._trigger_callback('on_error', f"Detection error: {e}")
    
    def _process_detections(self, detection_result: DetectionResult, camera_frame: CameraFrame):
        """Process detection results and create target reports."""
        current_time = time.time()
        
        for detection in detection_result.detections:
            # Generate target ID based on detection and position
            target_id = self._generate_target_id(detection, camera_frame.position)
            
            # Create target report
            target_report = TargetReport(
                target_id=target_id,
                detection=detection,
                drone_id=self.drone_id,
                position=camera_frame.position,
                confidence_score=detection.confidence,
                timestamp=current_time
            )
            
            # Store local detection
            self.local_detections[target_id] = target_report
            
            # Add to communication queue for sharing
            self.communication_queue.put({
                'type': 'detection',
                'data': target_report,
                'timestamp': current_time
            })
            
            # Trigger detection callback
            self._trigger_callback('on_detection', target_report)
        
        # Update stats
        self.stats['frames_processed'] += 1
        self.stats['detections_made'] += len(detection_result.detections)
        
        # Clean up old detections
        self._cleanup_old_detections(current_time)
    
    def _generate_target_id(self, detection: Detection, position: DronePosition) -> str:
        """Generate unique target ID based on detection and location."""
        # Simple target ID generation - in practice, this would be more sophisticated
        bbox_center_x = (detection.bbox[0] + detection.bbox[2]) / 2
        bbox_center_y = (detection.bbox[1] + detection.bbox[3]) / 2
        
        # Create ID based on class, position, and rough location
        target_id = f"{detection.class_name}_{position.x:.1f}_{position.y:.1f}_{bbox_center_x:.0f}_{bbox_center_y:.0f}"
        return target_id
    
    def _consensus_thread(self):
        """Consensus processing thread."""
        while self.running:
            try:
                # Check for targets that need consensus validation
                current_time = time.time()
                
                for target_id, target_report in self.local_detections.items():
                    if target_report.confirmed:
                        continue
                    
                    # Calculate consensus score
                    consensus_score = self._calculate_consensus_score(target_id)
                    
                    if consensus_score >= self.consensus_threshold:
                        # Target confirmed by consensus
                        target_report.confirmed = True
                        self.confirmed_targets[target_id] = target_report
                        
                        # Update stats and trigger callback
                        self.stats['targets_confirmed'] += 1
                        self._trigger_callback('on_target_confirmed', target_report)
                        
                        logger.info(f"Target {target_id} confirmed by consensus (score: {consensus_score:.2f})")
                
                time.sleep(0.1)  # Consensus check interval
                
            except Exception as e:
                logger.error(f"Consensus thread error for drone {self.drone_id}: {e}")
                self._trigger_callback('on_error', f"Consensus error: {e}")
    
    def _calculate_consensus_score(self, target_id: str) -> float:
        """Calculate consensus score for a target across all drones."""
        if target_id not in self.local_detections:
            return 0.0
        
        local_report = self.local_detections[target_id]
        total_confidence = local_report.confidence_score
        vote_count = 1
        
        # Check peer reports for similar targets
        for drone_id, peer_reports in self.peer_detections.items():
            for peer_report in peer_reports:
                if self._targets_match(local_report, peer_report):
                    total_confidence += peer_report.confidence_score
                    vote_count += 1
        
        # Calculate weighted consensus score
        if vote_count > 1:
            consensus_score = total_confidence / vote_count
            # Boost score based on number of confirming drones
            consensus_score *= min(1.0 + (vote_count - 1) * 0.1, 1.5)
        else:
            consensus_score = total_confidence
        
        return min(consensus_score, 1.0)
    
    def _targets_match(self, report1: TargetReport, report2: TargetReport) -> bool:
        """Check if two target reports refer to the same object."""
        # Simple matching based on class and approximate position
        if report1.detection.class_name != report2.detection.class_name:
            return False
        
        # Check if positions are close (simplified - would use proper coordinate transformation)
        pos_distance = np.sqrt(
            (report1.position.x - report2.position.x) ** 2 +
            (report1.position.y - report2.position.y) ** 2
        )
        
        return pos_distance < 50.0  # Within 50 meters
    
    def _communication_thread(self):
        """Communication thread for mesh network interaction."""
        # Placeholder for mesh network communication
        # This will be implemented when we add the communication module
        while self.running:
            try:
                # Process outgoing messages
                if not self.communication_queue.empty():
                    message = self.communication_queue.get(timeout=1.0)
                    # TODO: Send message via mesh network
                    logger.debug(f"Would send message: {message['type']}")
                
                # TODO: Receive and process incoming messages from other drones
                
                time.sleep(0.05)  # Communication check interval
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Communication thread error for drone {self.drone_id}: {e}")
    
    def _health_monitor_thread(self):
        """Health monitoring and statistics thread."""
        while self.running:
            try:
                current_time = time.time()
                
                # Update performance stats
                self.stats['last_update'] = current_time
                
                # Check system health
                if current_time - self.stats['last_update'] > 10.0:
                    logger.warning(f"Drone {self.drone_id} health check: No recent activity")
                
                # Log periodic status
                if self.stats['frames_processed'] % 100 == 0 and self.stats['frames_processed'] > 0:
                    fps = self.stats['frames_processed'] / (current_time - self.stats.get('start_time', current_time))
                    logger.info(f"Drone {self.drone_id} status: {fps:.1f} FPS, "
                              f"{self.stats['detections_made']} detections, "
                              f"{self.stats['targets_confirmed']} confirmed targets")
                
                time.sleep(5.0)  # Health check interval
                
            except Exception as e:
                logger.error(f"Health monitor error for drone {self.drone_id}: {e}")
    
    def _cleanup_old_detections(self, current_time: float):
        """Remove old detections that are past max age."""
        expired_targets = []
        
        for target_id, target_report in self.local_detections.items():
            if current_time - target_report.timestamp > self.max_detection_age:
                expired_targets.append(target_id)
        
        for target_id in expired_targets:
            del self.local_detections[target_id]
            if target_id in self.confirmed_targets:
                del self.confirmed_targets[target_id]
    
    def _update_processing_stats(self, processing_time: float):
        """Update processing performance statistics."""
        if self.stats['avg_processing_time'] == 0:
            self.stats['avg_processing_time'] = processing_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats['avg_processing_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.stats['avg_processing_time']
            )
    
    def update_position(self, position: DronePosition):
        """Update drone position."""
        self.position = position
    
    def get_status(self) -> Dict[str, Any]:
        """Get current drone status and statistics."""
        return {
            'drone_id': self.drone_id,
            'state': self.state.value,
            'position': {
                'x': self.position.x,
                'y': self.position.y,
                'z': self.position.z,
                'yaw': self.position.yaw
            },
            'stats': self.stats.copy(),
            'local_detections': len(self.local_detections),
            'confirmed_targets': len(self.confirmed_targets),
            'camera_active': self.camera_active,
            'running': self.running
        }
    
    def get_confirmed_targets(self) -> List[TargetReport]:
        """Get list of confirmed targets."""
        return list(self.confirmed_targets.values())
    
    def force_target_confirmation(self, target_id: str) -> bool:
        """Force confirmation of a target (for testing/emergency)."""
        if target_id in self.local_detections:
            target_report = self.local_detections[target_id]
            target_report.confirmed = True
            self.confirmed_targets[target_id] = target_report
            self._trigger_callback('on_target_confirmed', target_report)
            return True
        return False


class DroneSwarmCoordinator:
    """
    Coordinator for managing multiple drone agents.
    Useful for testing and simulation.
    """
    
    def __init__(self):
        self.drones = {}
        self.running = False
    
    def add_drone(self, drone_agent: DroneAgent):
        """Add a drone to the swarm."""
        self.drones[drone_agent.drone_id] = drone_agent
        logger.info(f"Added drone {drone_agent.drone_id} to swarm")
    
    def start_swarm(self):
        """Start all drones in the swarm."""
        logger.info(f"Starting swarm with {len(self.drones)} drones")
        for drone in self.drones.values():
            drone.start()
        self.running = True
    
    def stop_swarm(self):
        """Stop all drones in the swarm."""
        logger.info("Stopping drone swarm")
        for drone in self.drones.values():
            drone.stop()
        self.running = False
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get status of all drones in the swarm."""
        return {
            'total_drones': len(self.drones),
            'running': self.running,
            'drone_statuses': {
                drone_id: drone.get_status() 
                for drone_id, drone in self.drones.items()
            }
        }


# Example usage and testing
if __name__ == "__main__":
    print("🚁 Testing HiveMind Drone Agent")
    
    # Create drone agent
    drone = DroneAgent(
        drone_id="test_drone_001",
        camera_source=0,  # Use default camera
        detection_threshold=0.3
    )
    
    # Add callback for target confirmations
    def on_target_confirmed(agent, target_report):
        print(f"🎯 Target confirmed: {target_report.detection.class_name} "
              f"(confidence: {target_report.confidence_score:.2f})")
    
    def on_detection(agent, target_report):
        print(f"👁️ Detection: {target_report.detection.class_name} "
              f"at {target_report.detection.bbox}")
    
    drone.add_callback('on_target_confirmed', on_target_confirmed)
    drone.add_callback('on_detection', on_detection)
    
    try:
        # Start drone
        drone.start()
        
        # Run for a test period
        print("Running drone agent for 30 seconds...")
        time.sleep(30)
        
        # Print final status
        status = drone.get_status()
        print(f"Final status: {status}")
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Stop drone
        drone.stop()
        print("✅ Drone agent test completed!")