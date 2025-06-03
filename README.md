# HiveMind Vision: Project Memory & Implementation Guide

## Core System Architecture

**Mission**: Real-time visual consensus among 100-1000 autonomous drones for distributed surveillance and target engagement. Each drone processes visual data locally, shares compressed features via mesh network, and participates in Byzantine fault-tolerant consensus for target identification.

**Key Innovation**: Distributed computer vision with consensus tracking - multiple drones confirm target identity before engagement authorization through hierarchical coordination (Squad leaders: 10 drones → Battalion leaders: 100 drones → Command center).

## Technical Stack & Dependencies

### Core Technologies
- **Computer Vision**: OpenCV, PyTorch, SIFT/ORB + deep features (ResNet/ViT backbone)
- **Multi-Agent RL**: Stable-Baselines3, MADDPG/PPO for coordination
- **Communication**: ZeroMQ for mesh networking, Protocol Buffers for serialization
- **Simulation**: AirSim (primary), Gazebo + ROS (hardware testing)
- **Edge Deployment**: TensorRT, ONNX for Jetson Xavier/Orin optimization

### Key Algorithms
- **Consensus**: Byzantine fault tolerance (handles 30% drone failures)
- **SLAM**: Distributed mapping with feature sharing and map merging
- **Tracking**: Multi-object tracking across multiple camera views with triangulation
- **Coordination**: Hierarchical task allocation with dynamic role assignment

## Essential Datasets & Models

### Primary Datasets (Download Priority)
1. **VisDrone-DET**: 1.44GB trainset + 0.35GB val/test-dev (object detection foundation)
2. **VisDrone-MOT**: 11.15GB trainset + val + test-dev (multi-object tracking for consensus)
3. **AirSim Synthetic**: Custom scenarios via API (infinite training data)

### Pretrained Models
- **Detection**: YOLOv8/DETR fine-tuned on VisDrone
- **Features**: ViT-Base for visual embeddings
- **RL**: MADDPG checkpoints for multi-agent coordination

## File Structure (Implementation-Ready)

```
hivemind_vision/
├── requirements.txt                  # torch>=2.0, opencv-python, airsim, zmq, protobuf
├── config/
│   ├── swarm_config.yaml            # Drone count, formation params, communication settings
│   ├── models_config.yaml           # Model architectures, checkpoints, hyperparams
│   └── datasets_config.yaml         # Data paths, preprocessing pipelines
│
├── src/
│   ├── core/
│   │   ├── swarm_manager.py         # Main: SwarmManager.coordinate_mission()
│   │   ├── drone_agent.py           # DroneAgent.process_frame() + consensus voting
│   │   ├── consensus_engine.py      # ByzantineConsensus.vote_on_target()
│   │   └── hierarchical_control.py  # Squad/Battalion leader assignment logic
│   │
│   ├── vision/
│   │   ├── detection_model.py       # YOLOv8 wrapper, load_pretrained_yolo()
│   │   ├── feature_extractor.py     # extract_sift_features(), extract_deep_features()
│   │   ├── multi_view_tracker.py    # track_across_drones(), triangulate_position()
│   │   ├── consensus_tracker.py     # ConsensusTracker.merge_detections()
│   │   └── distributed_slam.py      # DistributedSLAM.update_global_map()
│   │
│   ├── communication/
│   │   ├── mesh_network.py          # ZMQ mesh setup, DroneNetwork.broadcast()
│   │   ├── message_protocol.py      # Protobuf schemas: DetectionMsg, ConsensusMsg
│   │   ├── compression.py           # compress_features(), adaptive_bitrate()
│   │   └── fault_tolerance.py       # handle_node_failure(), reroute_messages()
│   │
│   ├── ml/
│   │   ├── marl_trainer.py          # MADDPG training loop, load_sb3_policy()
│   │   ├── model_optimization.py    # tensorrt_optimize(), quantize_for_jetson()
│   │   └── federated_learning.py    # FederatedTrainer.aggregate_gradients()
│   │
│   ├── simulation/
│   │   ├── airsim_env.py           # AirSimMultiDrone.setup_swarm()
│   │   ├── scenario_generator.py    # generate_surveillance_mission()
│   │   └── evaluation_metrics.py    # calculate_swarm_metrics(), consensus_accuracy()
│   │
│   └── utils/
│       ├── data_loader.py          # VisDroneDataset, MultiDroneDataLoader
│       ├── visualization.py        # plot_swarm_formation(), show_consensus_heatmap()
│       └── config_parser.py        # load_config(), validate_swarm_params()
│
├── data/
│   ├── visdrone/                   # VisDrone-DET + MOT datasets
│   ├── models/                     # Pretrained weights, optimized models
│   └── scenarios/                  # AirSim mission configurations
│
├── experiments/
│   ├── consensus_validation.py     # Test Byzantine consensus under failures
│   ├── scaling_benchmark.py        # Performance vs swarm size (5→50→100 drones)
│   └── real_world_trial.py         # Hardware deployment scripts
│
└── deployment/
    ├── jetson_setup.sh             # Edge device configuration
    ├── docker/                     # Containerized swarm services
    └── kubernetes/                 # Distributed deployment manifests
```

## Implementation Phases (Development Order)

### Phase 1: Core Vision + Basic Swarm (Weeks 1-4)
**Files to implement first:**
```python
# Week 1: Basic detection
src/vision/detection_model.py      # YOLOv8 on VisDrone
src/utils/data_loader.py          # VisDrone dataset loading
src/simulation/airsim_env.py      # 5-drone AirSim setup

# Week 2: Communication
src/communication/mesh_network.py  # ZMQ basic messaging
src/communication/message_protocol.py # Detection sharing protocol

# Week 3: Simple consensus
src/core/drone_agent.py           # Individual drone logic
src/core/consensus_engine.py      # Basic voting mechanism

# Week 4: Integration
src/core/swarm_manager.py         # Orchestrate 5-drone detection sharing
```

### Phase 2: Advanced Consensus + SLAM (Weeks 5-8)
```python
src/vision/multi_view_tracker.py    # Cross-drone tracking
src/vision/distributed_slam.py      # Shared mapping
src/communication/fault_tolerance.py # Handle drone failures
```

### Phase 3: MARL + Optimization (Weeks 9-12)
```python
src/ml/marl_trainer.py              # Multi-agent coordination
src/ml/model_optimization.py        # Edge deployment prep
experiments/scaling_benchmark.py    # Test 50+ drone swarms
```

## Key Implementation Details

### Data Flow Architecture
```python
# Each drone runs this loop:
def drone_main_loop():
    frame = camera.capture()
    detections = detection_model.predict(frame)
    features = feature_extractor.extract(detections)
    
    # Share with swarm
    mesh_network.broadcast(DetectionMsg(features, position, timestamp))
    
    # Receive from others
    peer_detections = mesh_network.receive_all()
    
    # Consensus voting
    consensus_targets = consensus_engine.vote(detections, peer_detections)
    
    # Update global state
    distributed_slam.update_map(consensus_targets)
```

### Critical Parameters
```yaml
# config/swarm_config.yaml
swarm:
  size: 10                    # Start small, scale to 100+
  communication_range: 1000   # meters
  consensus_threshold: 0.7    # 70% agreement required
  update_frequency: 10        # Hz
  
vision:
  detection_confidence: 0.5
  feature_dim: 512           # Deep feature size
  max_targets_per_drone: 20
  
networking:
  message_timeout: 100       # ms
  compression_ratio: 0.1     # Feature compression
  byzantine_tolerance: 0.3   # Handle 30% failures
```

### Performance Targets
- **Latency**: <100ms end-to-end detection→consensus→action
- **Accuracy**: 90%+ target detection in VisDrone test-dev
- **Scalability**: Linear performance 5→100 drones
- **Fault Tolerance**: Function with 30% drone failures
- **Edge Deployment**: <50ms inference on Jetson Xavier

## Hardware Requirements

### Development
- **GPU**: RTX 4080+ (for training/simulation)
- **RAM**: 32GB+ (large swarm simulations)
- **Storage**: 200GB+ SSD (datasets + models)

### Deployment (per drone)
- **Compute**: Jetson Xavier NX/Orin
- **Camera**: Stereo vision, 4K capability
- **Communication**: 4G/5G + WiFi mesh
- **Storage**: 64GB+ eUFS

This condensed guide contains all essential information for implementing HiveMind Vision without extraneous details, optimized for development reference and coding implementation.