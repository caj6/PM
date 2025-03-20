# Projection mapping

## Interactive Projection Mapping

### Overview

This project integrates Blender 3D rendering with real-time face detection using OpenCV and an Intel RealSense camera to create an interactive holographic projection mapping system. The system dynamically adjusts virtual windows on objects based on the observer’s position, enhancing the illusion of looking through a real window into a virtual world.

### Features

- Blender 3D Integration

 * Clears previous objects and imports 3D models.

 * Ensures the model fits within a designated holographic cube.

 * Applies emissive noise textures for a flickering, dynamic effect.

 * Uses a strategically placed camera and lighting system for optimal projection alignment.

- OpenCV & Intel RealSense Face Detection

 * Captures real-time video using an Intel RealSense camera.

 * Detects and tracks human faces using deep learning-based models.

 * Analyzes position data to refine image alignment in Blender.

 * Adjusts the holographic projection dynamically based on the observer’s viewpoint.

- Real-Time Adjustments & Performance Optimization

 * Utilizes CNN-based face detection for high accuracy with minimal latency.

 * Leverages holography transformations for precise projection alignment.

 * Implements multi-threading for high frame rates and smooth, responsive effects.

## System Architecture

### Blender Scene Setup

* Import and position the 3D model.

 * Apply emissive textures for a holographic effect.

 * Align the camera and lighting for accurate projection.

### RealSense Face Detection

 * Capture video streams in real-time.

 * Process frames using CNN-based face detection.

 * Extract face position data for dynamic adjustments.

### Projection Alignment

 * Calculate observer position and adjust the projection.

 * Optimize rendering for minimal lag and smooth visualization.

## Installation & Setup

### Prerequisites

 - Blender (4.3)
 - Python 3.11
 - OpenCV (pip install opencv-python)
 - Intel RealSense SDK (pip install pyrealsense2)
 - Blender Python API

### Steps to Run

1. Clone this repository:

2. Install required dependencies:

3. Run the face detection module:

4. Start the Blender projection system:

## Future Improvements

- Enhancing real-time tracking accuracy with improved deep learning models.

- Implementing more advanced shaders for realistic holographic effects.

- Adding multi-user tracking for a more immersive experience.

## Acknowledgments

- Blender API, OpenCV and Intel RealSense community for real-time face tracking resources.

- Blender developers for providing a powerful rendering platform.

- Deep learning advancements for efficient real-time face detection.

## 📢 Contributions & Feedback

We welcome contributions! Feel free to open issues or submit pull requests to enhance this project.
