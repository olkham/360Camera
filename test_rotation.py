#!/usr/bin/env python3
"""
Test script to verify the rotation matrix calculations match
"""

import numpy as np
import math

def test_rotation_matrix_consistency():
    print("Testing rotation matrix consistency...")
    
    # Test angles
    roll_deg, pitch_deg, yaw_deg = 15, 30, 45
    roll_rad = math.radians(roll_deg)
    pitch_rad = math.radians(pitch_deg)
    yaw_rad = math.radians(yaw_deg)
    
    # Original method (working)
    cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)
    R_roll = np.array([[cos_r, -sin_r, 0],
                      [sin_r, cos_r, 0],
                      [0, 0, 1]], dtype=np.float64)
    
    cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
    R_pitch = np.array([[1, 0, 0],
                       [0, cos_p, -sin_p],
                       [0, sin_p, cos_p]], dtype=np.float64)
    
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    R_yaw = np.array([[cos_y, 0, sin_y],
                     [0, 1, 0],
                     [-sin_y, 0, cos_y]], dtype=np.float64)
    
    # Combined rotation matrix (order matters!)
    R_original = R_yaw @ R_pitch @ R_roll
    
    # New ultra-fast method (corrected)
    R00 = cos_y * cos_r + sin_y * sin_r * sin_p
    R01 = cos_y * (-sin_r) + sin_y * cos_r * sin_p
    R02 = sin_y * cos_p
    R10 = sin_r * cos_p
    R11 = cos_r * cos_p
    R12 = -sin_p
    R20 = -sin_y * cos_r + cos_y * sin_r * sin_p
    R21 = -sin_y * (-sin_r) + cos_y * cos_r * sin_p
    R22 = cos_y * cos_p
    
    R_new = np.array([[R00, R01, R02],
                     [R10, R11, R12],
                     [R20, R21, R22]])
    
    # Compare matrices
    diff = np.abs(R_original - R_new)
    max_diff = np.max(diff)
    
    print(f"Test angles: roll={roll_deg}°, pitch={pitch_deg}°, yaw={yaw_deg}°")
    print(f"Original matrix:\n{R_original}")
    print(f"New matrix:\n{R_new}")
    print(f"Max difference: {max_diff:.10f}")
    
    if max_diff < 1e-10:
        print("✅ Rotation matrices match perfectly!")
        return True
    else:
        print("❌ Rotation matrices differ!")
        return False

def test_point_rotation():
    """Test rotation on actual points"""
    print("\nTesting point rotation...")
    
    # Test point
    test_point = np.array([1.0, 0.5, 0.8])
    
    # Test angles
    roll_deg, pitch_deg, yaw_deg = 0, 30, 0  # Test pitch specifically
    roll_rad = math.radians(roll_deg)
    pitch_rad = math.radians(pitch_deg)
    yaw_rad = math.radians(yaw_deg)
    
    # Original method
    cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)
    R_roll = np.array([[cos_r, -sin_r, 0],
                      [sin_r, cos_r, 0],
                      [0, 0, 1]])
    
    cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
    R_pitch = np.array([[1, 0, 0],
                       [0, cos_p, -sin_p],
                       [0, sin_p, cos_p]])
    
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    R_yaw = np.array([[cos_y, 0, sin_y],
                     [0, 1, 0],
                     [-sin_y, 0, cos_y]])
    
    R_original = R_yaw @ R_pitch @ R_roll
    result_original = R_original @ test_point
    
    # New method
    R00 = cos_y * cos_r + sin_y * sin_r * sin_p
    R01 = cos_y * (-sin_r) + sin_y * cos_r * sin_p
    R02 = sin_y * cos_p
    R10 = sin_r * cos_p
    R11 = cos_r * cos_p
    R12 = -sin_p
    R20 = -sin_y * cos_r + cos_y * sin_r * sin_p
    R21 = -sin_y * (-sin_r) + cos_y * cos_r * sin_p
    R22 = cos_y * cos_p
    
    # Apply rotation manually (as in JIT function)
    x, y, z = test_point[0], test_point[1], test_point[2]
    x_rot = R00 * x + R01 * y + R02 * z
    y_rot = R10 * x + R11 * y + R12 * z
    z_rot = R20 * x + R21 * y + R22 * z
    
    result_new = np.array([x_rot, y_rot, z_rot])
    
    print(f"Test point: {test_point}")
    print(f"Original result: {result_original}")
    print(f"New result: {result_new}")
    print(f"Difference: {np.abs(result_original - result_new)}")
    print(f"Max difference: {np.max(np.abs(result_original - result_new)):.10f}")

if __name__ == "__main__":
    test_rotation_matrix_consistency()
    test_point_rotation()
