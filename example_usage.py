#!/usr/bin/env python3
"""
Example usage of the new FrameProcessor architecture.
This demonstrates how to use the refactored code.
"""

from main import FrameProcessor, Equirectangular2PinholeProcessor, VideoProcessor
import cv2
import numpy as np

def example_basic_usage():
    """Basic example of using the new processor architecture."""
    print("Basic Usage Example")
    print("-" * 40)
    
    # Step 1: Create a frame processor
    processor = Equirectangular2PinholeProcessor(
        fov=90.0,
        output_width=1920,
        output_height=1080
    )
    
    # Step 2: Set viewing parameters
    processor.set_parameter('yaw', 45.0)    # Look 45 degrees to the right
    processor.set_parameter('pitch', 10.0)   # Look 10 degrees up
    processor.set_parameter('roll', 0.0)     # No roll
    
    # Step 3: Create a sample equirectangular frame
    sample_frame = np.zeros((720, 1440, 3), dtype=np.uint8)
    sample_frame[:, :, 1] = 128  # Green background
    
    # Step 4: Process the frame
    processed_frame = processor.process(sample_frame)
    
    print(f"Input frame shape: {sample_frame.shape}")
    print(f"Output frame shape: {processed_frame.shape}")
    print(f"Current parameters: {processor.get_parameters()}")
    
    return processed_frame

def example_interactive_video():
    """Example of using with video input."""
    print("\nInteractive Video Example")
    print("-" * 40)
    
    # Create processor with desired output resolution
    frame_processor = Equirectangular2PinholeProcessor(
        fov=90.0,
        output_width=1280,
        output_height=720
    )
    
    # Set initial viewing direction
    frame_processor.set_parameter('yaw', 0.0)
    frame_processor.set_parameter('pitch', 0.0)
    frame_processor.set_parameter('roll', 0.0)
    
    # Use with video file (replace with your video path)
    video_path = "your_360_video.mp4"
    video_path = 0  # Use "0" for webcam input
    
    try:
        # Create video processor
        video_processor = VideoProcessor(video_path, frame_processor)
        
        print(f"Video loaded: {video_processor.width}x{video_processor.height}")
        print("Starting interactive viewer...")
        print("Use A/D for yaw, W/S for pitch, Q/E for roll, Z/X for FOV")
        print("Press ESC to exit")
        
        # Start interactive viewer
        video_processor.interactive_viewer()
        
    except Exception as e:
        print(f"Could not load video: {e}")
        print("Make sure you have a valid video file or camera connected.")

def example_parameter_animation():
    """Example of animating parameters programmatically."""
    print("\nParameter Animation Example")
    print("-" * 40)
    
    # Create processor
    processor = Equirectangular2PinholeProcessor(
        fov=90.0,
        output_width=640,
        output_height=480
    )
    
    # Create sample frame
    sample_frame = np.zeros((720, 1440, 3), dtype=np.uint8)
    sample_frame[:, :, 2] = 255  # Red background
    
    # Animate yaw from 0 to 360 degrees
    for yaw in range(0, 360, 45):
        processor.set_parameter('yaw', float(yaw))
        processed_frame = processor.process(sample_frame)
        
        print(f"Yaw: {yaw}°, processed frame shape: {processed_frame.shape}")
        
        # In a real application, you would save or display the frame here
        # cv2.imshow('Animated View', processed_frame)
        # cv2.waitKey(100)
    
    print("Animation complete!")

def example_custom_processor():
    """Example of creating a custom processor."""
    print("\nCustom Processor Example")
    print("-" * 40)
    
    class SimpleColorProcessor(FrameProcessor):
        """A simple processor that just applies a color filter."""
        
        def __init__(self):
            super().__init__()
            self._parameters = {
                'red_factor': 1.0,
                'green_factor': 1.0,
                'blue_factor': 1.0
            }
        
        def process(self, frame: np.ndarray) -> np.ndarray:
            """Apply color scaling to the frame."""
            processed = frame.copy().astype(np.float32)
            
            processed[:, :, 0] *= self._parameters['blue_factor']
            processed[:, :, 1] *= self._parameters['green_factor']
            processed[:, :, 2] *= self._parameters['red_factor']
            
            return np.clip(processed, 0, 255).astype(np.uint8)
    
    # Create and use the custom processor
    color_processor = SimpleColorProcessor()
    color_processor.set_parameter('red_factor', 1.5)
    color_processor.set_parameter('green_factor', 0.5)
    color_processor.set_parameter('blue_factor', 0.8)
    
    # Create test frame
    test_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
    
    # Process
    result = color_processor.process(test_frame)
    
    print(f"Custom processor parameters: {color_processor.get_parameters()}")
    print(f"Input frame mean: {test_frame.mean()}")
    print(f"Output frame mean: {result.mean()}")

def main():
    """Run all examples."""
    print("=" * 60)
    print("New FrameProcessor Architecture Examples")
    print("=" * 60)
    
    # Run examples
    # example_basic_usage()
    # example_parameter_animation()
    # example_custom_processor()
    
    # Interactive example (commented out by default)
    example_interactive_video()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("Uncomment example_interactive_video() to test with real video.")
    print("=" * 60)

if __name__ == "__main__":
    main()
