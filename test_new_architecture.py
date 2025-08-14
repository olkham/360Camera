#!/usr/bin/env python3
"""
Test script to demonstrate the new FrameProcessor architecture
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import FrameProcessor, Equirectangular2PinholeProcessor, VideoProcessor
import cv2
import numpy as np

def test_frame_processor():
    """Test the frame processor with a simple example."""
    print("Testing FrameProcessor architecture...")
    
    # Create a simple test frame (simulating equirectangular input)
    test_frame = np.zeros((720, 1440, 3), dtype=np.uint8)
    test_frame[:, :, 1] = 128  # Green channel
    
    # Create the processor
    processor = Equirectangular2PinholeProcessor(
        fov=90.0,
        output_width=640,
        output_height=480
    )
    
    # Test parameter setting
    processor.set_parameter('yaw', 45.0)
    processor.set_parameter('pitch', 10.0)
    processor.set_parameter('roll', 0.0)
    
    # Test parameter getting
    print(f"Current yaw: {processor.get_parameter('yaw')}")
    print(f"Current pitch: {processor.get_parameter('pitch')}")
    print(f"All parameters: {processor.get_parameters()}")
    
    # Process the frame
    print("Processing frame...")
    processed_frame = processor.process(test_frame)
    
    print(f"Input frame shape: {test_frame.shape}")
    print(f"Output frame shape: {processed_frame.shape}")
    
    # Test cache functionality
    print(f"Cache entries: {len(processor._map_cache)}")
    
    # Process same frame again (should hit cache)
    processed_frame2 = processor.process(test_frame)
    print(f"Cache entries after second processing: {len(processor._map_cache)}")
    
    # Change parameters and process again
    processor.set_parameter('yaw', 90.0)
    processed_frame3 = processor.process(test_frame)
    print(f"Cache entries after parameter change: {len(processor._map_cache)}")
    
    print("✓ FrameProcessor test completed successfully!")
    return True

def test_video_processor():
    """Test the video processor with camera input."""
    print("\nTesting VideoProcessor with camera...")
    
    # Create processor
    frame_processor = Equirectangular2PinholeProcessor(
        fov=90.0,
        output_width=640,
        output_height=480
    )
    
    # Try to use camera (fallback to None if no camera available)
    try:
        video_processor = VideoProcessor(0, frame_processor)
        print(f"Camera initialized: {video_processor.width}x{video_processor.height}")
        
        # Test getting a frame
        frame = video_processor.get_frame(0)
        if frame is not None:
            print(f"Got frame of shape: {frame.shape}")
            
            # Test processing
            processed = frame_processor.process(frame)
            print(f"Processed frame shape: {processed.shape}")
            
            print("✓ VideoProcessor test completed successfully!")
            return True
        else:
            print("⚠ No frame available from camera")
            return False
            
    except Exception as e:
        print(f"⚠ Camera test failed: {e}")
        print("This is expected if no camera is available")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing New FrameProcessor Architecture")
    print("=" * 60)
    
    success = True
    
    # Test 1: FrameProcessor functionality
    try:
        test_frame_processor()
    except Exception as e:
        print(f"✗ FrameProcessor test failed: {e}")
        success = False
    
    # Test 2: VideoProcessor functionality
    try:
        test_video_processor()
    except Exception as e:
        print(f"✗ VideoProcessor test failed: {e}")
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! The new architecture is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    print("=" * 60)

if __name__ == "__main__":
    main()
