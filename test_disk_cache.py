#!/usr/bin/env python3
"""
Test script for the new disk-based caching system
"""

from main import Equirectangular360
import time
import os

def test_disk_cache():
    # Use a smaller memory limit to force disk caching
    processor = Equirectangular360(
        "C:/insta360/x5/exports/VID_20250704_123015_00_001(1).mp4", 
        use_optimized_coords=True,
        max_memory_cache_mb=100  # Small memory limit
    )
    
    print("Testing disk-based cache system...")
    print(f"Memory limit: {processor.max_memory_cache_mb}MB")
    print()
    
    # Test 1: Generate a few cache entries
    print("=== Test 1: Generating cache entries ===")
    dummy_frame = processor.get_frame(0)
    
    test_angles = [
        (0, 0, 0, 90),
        (45, 0, 0, 90),
        (90, 0, 0, 90),
        (0, 30, 0, 90),
        (0, 0, 0, 120),
    ]
    
    for yaw, pitch, roll, fov in test_angles:
        start = time.perf_counter()
        result = processor.get_perspective_projection(
            dummy_frame, yaw, pitch, roll, fov, benchmark=True
        )
        duration = time.perf_counter() - start
        print(f"  Generated ({yaw}, {pitch}, {roll}, {fov}): {duration*1000:.2f}ms")
        print(f"  Cache status: {processor.get_cache_info()}")
        print()
    
    # Test 2: Access cached entries
    print("=== Test 2: Accessing cached entries ===")
    for yaw, pitch, roll, fov in test_angles:
        start = time.perf_counter()
        result = processor.get_perspective_projection(
            dummy_frame, yaw, pitch, roll, fov, benchmark=True
        )
        duration = time.perf_counter() - start
        print(f"  Accessed ({yaw}, {pitch}, {roll}, {fov}): {duration*1000:.2f}ms")
    print()
    
    # Test 3: Disk cache statistics
    print("=== Test 3: Disk cache statistics ===")
    print(processor.get_disk_cache_stats())
    print(processor.get_cache_info())
    print()
    
    # Test 4: Memory flushing
    print("=== Test 4: Testing memory flushing ===")
    print("Generating many entries to trigger memory flushing...")
    
    for yaw in range(0, 180, 15):  # 12 entries
        for pitch in range(-30, 31, 15):  # 5 entries
            result = processor.get_perspective_projection(
                dummy_frame, yaw, pitch, 0, 90, benchmark=False
            )
    
    print(f"After generating many entries:")
    print(processor.get_cache_info())
    print(processor.get_disk_cache_stats())
    print()
    
    # Test 5: Small-scale prepopulation
    print("=== Test 5: Small-scale prepopulation ===")
    processor.prepopulate_cache(
        yaw_range=(0, 36), yaw_step=6,      # 6 yaw values
        pitch_range=(-30, 31), pitch_step=15, # 5 pitch values  
        roll_range=(0, 1), roll_step=30,    # 1 roll value
        fov_values=[90],                    # 1 FOV value
        disk_only_mode=True                 # Use disk-only mode
    )
    
    print("Final cache status:")
    print(processor.get_cache_info())
    print(processor.get_disk_cache_stats())

if __name__ == "__main__":
    test_disk_cache()
