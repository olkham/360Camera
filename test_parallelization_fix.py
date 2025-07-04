#!/usr/bin/env python3
"""
Test script to verify that the parallelization warning is fixed
"""

import numpy as np
import time
import warnings
from main import generate_mapping_jit_ultra_parallel, generate_mapping_jit_ultra_parallel_safe

def test_parallelization_warning():
    """Test if the parallel function still generates warnings"""
    print("Testing parallelization improvements...")
    print("=" * 50)
    
    # Test parameters
    output_width, output_height = 1920, 1080
    focal_length = 800.0
    cx, cy = output_width / 2, output_height / 2
    frame_width, frame_height = 3840, 1920
    
    # Identity rotation matrix (no rotation)
    R = np.eye(3)
    R00, R01, R02 = R[0, 0], R[0, 1], R[0, 2]
    R10, R11, R12 = R[1, 0], R[1, 1], R[1, 2]
    R20, R21, R22 = R[2, 0], R[2, 1], R[2, 2]
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        print("Testing new flattened parallel approach...")
        start_time = time.perf_counter()
        
        # First call (compilation)
        pixel_x1, pixel_y1 = generate_mapping_jit_ultra_parallel(
            output_width, output_height, focal_length, cx, cy,
            R00, R01, R02, R10, R11, R12, R20, R21, R22,
            frame_height, frame_width
        )
        
        compile_time = time.perf_counter() - start_time
        print(f"Compilation time: {compile_time:.3f}s")
        
        # Second call (execution)
        start_time = time.perf_counter()
        pixel_x1, pixel_y1 = generate_mapping_jit_ultra_parallel(
            output_width, output_height, focal_length, cx, cy,
            R00, R01, R02, R10, R11, R12, R20, R21, R22,
            frame_height, frame_width
        )
        execution_time = time.perf_counter() - start_time
        print(f"Execution time: {execution_time:.3f}s ({execution_time*1000:.1f}ms)")
        
        # Check for Numba warnings
        numba_warnings = [warning for warning in w if 'numba' in str(warning.message).lower()]
        perf_warnings = [warning for warning in w if 'performance' in str(warning.message).lower()]
        
        print(f"\nWarnings captured: {len(w)}")
        print(f"Numba-related warnings: {len(numba_warnings)}")
        print(f"Performance warnings: {len(perf_warnings)}")
        
        if numba_warnings:
            print("\nNumba warnings found:")
            for warning in numba_warnings:
                print(f"  - {warning.message}")
        else:
            print("\n✓ No Numba warnings detected!")
        
        if perf_warnings:
            print("\nPerformance warnings found:")
            for warning in perf_warnings:
                print(f"  - {warning.message}")
        else:
            print("✓ No performance warnings detected!")
    
    # Test the serial fallback for comparison
    print("\n" + "=" * 50)
    print("Testing serial fallback (for comparison)...")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        start_time = time.perf_counter()
        pixel_x2, pixel_y2 = generate_mapping_jit_ultra_parallel_safe(
            output_width, output_height, focal_length, cx, cy,
            R00, R01, R02, R10, R11, R12, R20, R21, R22,
            frame_height, frame_width
        )
        execution_time_serial = time.perf_counter() - start_time
        print(f"Serial execution time: {execution_time_serial:.3f}s ({execution_time_serial*1000:.1f}ms)")
        
        # Check warnings
        numba_warnings = [warning for warning in w if 'numba' in str(warning.message).lower()]
        print(f"Serial version warnings: {len(numba_warnings)}")
    
    # Verify results are identical
    diff_x = np.abs(pixel_x1 - pixel_x2).max()
    diff_y = np.abs(pixel_y1 - pixel_y2).max()
    print(f"\nResults comparison:")
    print(f"Max difference in pixel_x: {diff_x}")
    print(f"Max difference in pixel_y: {diff_y}")
    print(f"Results identical: {diff_x < 1e-6 and diff_y < 1e-6}")
    
    # Performance comparison
    if execution_time > 0 and execution_time_serial > 0:
        speedup = execution_time_serial / execution_time
        print(f"\nSpeedup from parallelization: {speedup:.2f}x")
        if speedup < 1.1:
            print("⚠️  Warning: Parallel version not significantly faster")
        elif speedup > 1.5:
            print("✓ Good parallelization speedup!")
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    if not numba_warnings and not perf_warnings:
        print("✅ SUCCESS: No Numba parallelization warnings!")
    else:
        print("❌ ISSUES: Still getting warnings")
    
    print(f"Parallel execution: {execution_time*1000:.1f}ms")
    print(f"Serial execution: {execution_time_serial*1000:.1f}ms")

if __name__ == "__main__":
    test_parallelization_warning()
