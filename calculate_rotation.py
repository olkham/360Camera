#!/usr/bin/env python3
"""
Calculate the correct combined rotation matrix elements
"""

import numpy as np
import math

def calculate_correct_rotation_matrix():
    """Calculate the correct combined rotation matrix elements"""
    
    # Symbolic calculation for R_yaw @ R_pitch @ R_roll
    # Let's use specific values first to verify
    roll_rad = math.radians(15)
    pitch_rad = math.radians(30)
    yaw_rad = math.radians(45)
    
    cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)
    cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    
    # Individual matrices
    R_roll = np.array([[cos_r, -sin_r, 0],
                      [sin_r, cos_r, 0],
                      [0, 0, 1]])
    
    R_pitch = np.array([[1, 0, 0],
                       [0, cos_p, -sin_p],
                       [0, sin_p, cos_p]])
    
    R_yaw = np.array([[cos_y, 0, sin_y],
                     [0, 1, 0],
                     [-sin_y, 0, cos_y]])
    
    # Combined: R = R_yaw @ R_pitch @ R_roll
    R_combined = R_yaw @ R_pitch @ R_roll
    
    print("Individual matrices:")
    print(f"R_roll:\n{R_roll}")
    print(f"R_pitch:\n{R_pitch}")
    print(f"R_yaw:\n{R_yaw}")
    print(f"Combined R_yaw @ R_pitch @ R_roll:\n{R_combined}")
    
    # Extract elements
    print("\nCombined matrix elements:")
    print(f"R00 = {R_combined[0,0]:.10f}")
    print(f"R01 = {R_combined[0,1]:.10f}")
    print(f"R02 = {R_combined[0,2]:.10f}")
    print(f"R10 = {R_combined[1,0]:.10f}")
    print(f"R11 = {R_combined[1,1]:.10f}")
    print(f"R12 = {R_combined[1,2]:.10f}")
    print(f"R20 = {R_combined[2,0]:.10f}")
    print(f"R21 = {R_combined[2,1]:.10f}")
    print(f"R22 = {R_combined[2,2]:.10f}")
    
    # Now let's derive the symbolic form
    print("\nSymbolic derivation:")
    print("R_yaw @ R_pitch @ R_roll =")
    print("[[cos_y, 0, sin_y],     [[1, 0, 0],           [[cos_r, -sin_r, 0],")
    print(" [0, 1, 0],         @    [0, cos_p, -sin_p], @  [sin_r, cos_r, 0],")
    print(" [-sin_y, 0, cos_y]]     [0, sin_p, cos_p]]     [0, 0, 1]]")
    
    # Let's calculate step by step
    # First: R_pitch @ R_roll
    print("\nStep 1: R_pitch @ R_roll")
    R_pitch_roll = R_pitch @ R_roll
    print(f"R_pitch @ R_roll:\n{R_pitch_roll}")
    
    # Then: R_yaw @ (R_pitch @ R_roll)
    print("\nStep 2: R_yaw @ (R_pitch @ R_roll)")
    final = R_yaw @ R_pitch_roll
    print(f"Final result:\n{final}")
    
    # Manual calculation to verify
    print("\nManual element calculation:")
    # R_pitch @ R_roll elements:
    PR00 = cos_r
    PR01 = -sin_r
    PR02 = 0
    PR10 = sin_r * cos_p
    PR11 = cos_r * cos_p
    PR12 = -sin_p
    PR20 = sin_r * sin_p
    PR21 = cos_r * sin_p
    PR22 = cos_p
    
    print(f"R_pitch @ R_roll manual:")
    print(f"[{PR00:.6f}, {PR01:.6f}, {PR02:.6f}]")
    print(f"[{PR10:.6f}, {PR11:.6f}, {PR12:.6f}]")
    print(f"[{PR20:.6f}, {PR21:.6f}, {PR22:.6f}]")
    
    # R_yaw @ (R_pitch @ R_roll) elements:
    R00_final = cos_y * PR00 + sin_y * PR20
    R01_final = cos_y * PR01 + sin_y * PR21
    R02_final = cos_y * PR02 + sin_y * PR22
    R10_final = PR10
    R11_final = PR11
    R12_final = PR12
    R20_final = -sin_y * PR00 + cos_y * PR20
    R21_final = -sin_y * PR01 + cos_y * PR21
    R22_final = -sin_y * PR02 + cos_y * PR22
    
    print(f"\nFinal manual calculation:")
    print(f"R00 = cos_y * cos_r + sin_y * sin_r * sin_p = {R00_final:.10f}")
    print(f"R01 = cos_y * (-sin_r) + sin_y * cos_r * sin_p = {R01_final:.10f}")
    print(f"R02 = cos_y * 0 + sin_y * cos_p = {R02_final:.10f}")
    print(f"R10 = sin_r * cos_p = {R10_final:.10f}")
    print(f"R11 = cos_r * cos_p = {R11_final:.10f}")
    print(f"R12 = -sin_p = {R12_final:.10f}")
    print(f"R20 = -sin_y * cos_r + cos_y * sin_r * sin_p = {R20_final:.10f}")
    print(f"R21 = -sin_y * (-sin_r) + cos_y * cos_r * sin_p = {R21_final:.10f}")
    print(f"R22 = -sin_y * 0 + cos_y * cos_p = {R22_final:.10f}")

if __name__ == "__main__":
    calculate_correct_rotation_matrix()
