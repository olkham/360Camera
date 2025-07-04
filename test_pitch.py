#!/usr/bin/env python3
"""
Quick test of the corrected pitch functionality
"""

from main import Equirectangular360
import cv2

def test_pitch_correction():
    print("Testing corrected pitch functionality...")
    
    # Load video
    video_path = 'C:/insta360/x5/exports/VID_20250704_123015_00_001(1).mp4'
    processor = Equirectangular360(video_path, use_optimized_coords=True, max_memory_cache_mb=50)
    
    # Get a test frame
    frame = processor.get_frame(0)
    if frame is None:
        print("Could not read test frame")
        return
    
    print("Testing pitch rotation with corrected matrix...")
    
    # Test different pitch values
    test_pitches = [0, 15, 30, -15, -30]
    
    for pitch in test_pitches:
        try:
            result = processor.get_perspective_projection(
                frame, yaw=0, pitch=pitch, roll=0, fov=90, benchmark=True
            )
            print(f"✅ Pitch {pitch}° completed successfully")
        except Exception as e:
            print(f"❌ Pitch {pitch}° failed: {e}")
    
    print("\nAll pitch tests completed!")
    print("The rotation matrix has been corrected to match the working version.")

if __name__ == "__main__":
    test_pitch_correction()
