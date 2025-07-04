#!/usr/bin/env python3
"""
Test large-scale cache prepopulation with 1° step sizes using disk-only mode
"""

from main import Equirectangular360
import time
import psutil
import os

def test_large_prepopulation():
    print("Testing large-scale cache prepopulation with 1° step sizes...")
    print("Memory before starting:", psutil.virtual_memory().used / (1024**3), "GB")
    
    # Create processor with reasonable memory limit
    processor = Equirectangular360(
        "C:/insta360/x5/exports/VID_20250704_123015_00_001(1).mp4", 
        use_optimized_coords=True,
        max_memory_cache_mb=500  # 500MB memory limit
    )
    
    # Clear any existing cache
    processor.clear_disk_cache()
    
    print(f"Starting prepopulation...")
    print(f"Memory limit: {processor.max_memory_cache_mb}MB")
    
    start_time = time.perf_counter()
    
    # Test with a smaller subset first - 10° steps instead of 1°
    processor.prepopulate_cache(
        output_width=1920,
        output_height=1080,
        fov_values=[90],
        yaw_range=(0, 360),           # Full yaw range  
        yaw_step=10,                  # 36 yaw values
        pitch_range=(-90, 90),        # Full pitch range
        pitch_step=10,                # 19 pitch values 
        roll_range=(0, 1),            # Only roll=0
        roll_step=30,
        disk_only_mode=True           # Use disk-only mode
    )
    
    total_time = time.perf_counter() - start_time
    
    print(f"\nPrepopulation completed in {total_time:.1f} seconds")
    print(f"Memory after completion:", psutil.virtual_memory().used / (1024**3), "GB")
    print(f"Final cache status: {processor.get_cache_info()}")
    print(f"Disk usage: {processor.get_disk_cache_stats()}")
    
    # Test cache access performance
    print("\n=== Testing cache access performance ===")
    dummy_frame = processor.get_frame(0)
    
    # Test a few random access patterns
    test_cases = [
        (0, 0, 0, 90),      # Should be cached
        (10, 10, 0, 90),    # Should be cached  
        (5, 5, 0, 90),      # Should NOT be cached (not on 10° grid)
        (20, -20, 0, 90),   # Should be cached
    ]
    
    for yaw, pitch, roll, fov in test_cases:
        start = time.perf_counter()
        result = processor.get_perspective_projection(
            dummy_frame, yaw, pitch, roll, fov, benchmark=True
        )
        duration = time.perf_counter() - start
        print(f"Access ({yaw}, {pitch}, {roll}, {fov}): {duration*1000:.2f}ms")

if __name__ == "__main__":
    test_large_prepopulation()
