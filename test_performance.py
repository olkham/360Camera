#!/usr/bin/env python3
"""
Quick performance test for the optimized coordinate generation
"""

import time
import numpy as np
from main import Equirectangular360, generate_mapping_jit_ultra, generate_mapping_jit_ultra_parallel

def test_coordinate_performance():
    print("=" * 60)
    print("COORDINATE GENERATION PERFORMANCE TEST")
    print("=" * 60)
    
    # Test different output resolutions
    resolutions = [
        (1280, 720),   # HD
        (1920, 1080),  # Full HD  
        (2560, 1440),  # 2K
        (3840, 2160),  # 4K
    ]
    
    # Test parameters
    yaw, pitch, roll, fov = 45, 30, 0, 90
    frame_shape = (3840, 7680)  # Example 360 video frame
    iterations = 3
    
    for width, height in resolutions:
        print(f"\nTesting {width}x{height} ({width*height:,} pixels)")
        
        # Calculate constants for JIT functions
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch) 
        roll_rad = np.radians(roll)
        fov_rad = np.radians(fov)
        
        focal_length = width / (2 * np.tan(fov_rad / 2))
        cx = width * 0.5
        cy = height * 0.5
        
        # Create rotation matrix elements
        cos_r, sin_r = np.cos(roll_rad), np.sin(roll_rad)
        cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        
        R00 = cos_y * cos_r - sin_y * sin_p * sin_r
        R01 = -cos_y * sin_r - sin_y * sin_p * cos_r
        R02 = sin_y * cos_p
        R10 = cos_p * sin_r
        R11 = cos_p * cos_r
        R12 = -sin_p
        R20 = -sin_y * cos_r - cos_y * sin_p * sin_r
        R21 = sin_y * sin_r - cos_y * sin_p * cos_r
        R22 = cos_y * cos_p
        
        # Test ultra-optimized version
        times = []
        for i in range(iterations):
            start = time.perf_counter()
            if width * height > 1000000:
                pixel_x, pixel_y = generate_mapping_jit_ultra_parallel(
                    width, height, focal_length, cx, cy,
                    R00, R01, R02, R10, R11, R12, R20, R21, R22,
                    frame_shape[0], frame_shape[1]
                )
            else:
                pixel_x, pixel_y = generate_mapping_jit_ultra(
                    width, height, focal_length, cx, cy,
                    R00, R01, R02, R10, R11, R12, R20, R21, R22,
                    frame_shape[0], frame_shape[1]
                )
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times) * 1000
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000
        
        pixels_per_sec = (width * height) / (avg_time / 1000)
        
        method = "parallel" if width * height > 1000000 else "serial"
        print(f"  Method: {method}")
        print(f"  Time: {avg_time:.2f}ms (min: {min_time:.2f}ms, max: {max_time:.2f}ms)")
        print(f"  Throughput: {pixels_per_sec/1e6:.1f} million pixels/second")
        print(f"  Memory: {(pixel_x.nbytes + pixel_y.nbytes)/1e6:.1f} MB")

if __name__ == "__main__":
    test_coordinate_performance()
