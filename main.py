import cv2
import numpy as np
import math
from numba import jit, njit
import time
from concurrent.futures import ThreadPoolExecutor

# Set Numba compilation cache for faster startup
import numba
numba.config.CACHE_DIR = './numba_cache'

class Equirectangular360:
    def __init__(self, video_path, use_optimized_coords=True, max_memory_cache_mb=100):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Optimization settings
        self.use_optimized_coords = use_optimized_coords
        self.max_memory_cache_mb = max_memory_cache_mb
        
        # Optimize video capture settings for speed
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent lag
        
        # Simple in-memory cache for coordinate mappings
        self._map_cache = {}
        self._cache_size_limit = 50  # Limit number of cached mappings
        
        # Frame cache for faster sequential access
        self._frame_cache = {}
        self._frame_cache_limit = 30  # Limit frame cache
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {self.width}x{self.height}, {self.fps} fps, {self.frame_count} frames")
    
    def get_perspective_projection(self, frame, yaw, pitch, roll, fov, output_width=1920, output_height=1080, benchmark=False):
        """
        Convert equirectangular frame to perspective projection (optimized)
        
        Args:
            frame: Input equirectangular frame
            yaw: Horizontal rotation in degrees (left/right)
            pitch: Vertical rotation in degrees (up/down)
            roll: Camera roll in degrees (tilt)
            fov: Field of view in degrees
            output_width: Output frame width
            output_height: Output frame height
            benchmark: If True, print timing information
        
        Returns:
            Perspective projected frame
        """
        start_time = time.perf_counter() if benchmark else None
        
        # Normalize angles for consistent caching
        norm_yaw, norm_pitch, norm_roll = self.normalize_angles(yaw, pitch, roll)
        
        # Create cache key for coordinate mapping using normalized angles
        cache_key = (norm_yaw, norm_pitch, norm_roll, fov, output_width, output_height, frame.shape[0], frame.shape[1])
        
        cache_start = time.perf_counter() if benchmark else None
        # Simple cache lookup
        if cache_key in self._map_cache:
            pixel_x, pixel_y = self._map_cache[cache_key]
            cache_hit = True
        else:
            # Generate new mapping using the fastest method
            pixel_x, pixel_y = self._generate_coordinate_mapping_ultra_fast(
                norm_yaw, norm_pitch, norm_roll, fov, output_width, output_height, frame.shape, benchmark
            )
            
            # Simple cache management - remove oldest if cache is full
            if len(self._map_cache) >= self._cache_size_limit:
                # Remove first (oldest) entry
                oldest_key = next(iter(self._map_cache))
                del self._map_cache[oldest_key]
            
            self._map_cache[cache_key] = (pixel_x, pixel_y)
            cache_hit = False
        
        cache_time = time.perf_counter() - cache_start if benchmark else None
        
        remap_start = time.perf_counter() if benchmark else None
        # Use optimized remapping
        output_img = cv2.remap(frame, pixel_x, pixel_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        remap_time = time.perf_counter() - remap_start if benchmark else None
        
        total_time = time.perf_counter() - start_time if benchmark else None
        
        if benchmark:
            cache_status = "HIT" if cache_hit else "MISS"
            angles_info = f"({yaw:.1f}°,{pitch:.1f}°,{roll:.1f}°) -> ({norm_yaw:.1f}°,{norm_pitch:.1f}°,{norm_roll:.1f}°)" if (yaw != norm_yaw or pitch != norm_pitch or roll != norm_roll) else f"({yaw:.1f}°,{pitch:.1f}°,{roll:.1f}°)"
            print(f"Projection Timing {angles_info} - Total: {total_time*1000:.2f}ms | Cache {cache_status}: {cache_time*1000:.2f}ms | Remap: {remap_time*1000:.2f}ms")
        
        return output_img
    
    def _generate_coordinate_mapping(self, yaw, pitch, roll, fov, output_width, output_height, frame_shape, benchmark=False):
        """Generate coordinate mapping for remapping (using original logic with detailed profiling)"""
        start_time = time.perf_counter() if benchmark else None
        profile = benchmark  # Enable detailed profiling when benchmarking
        
        # Convert angles to radians
        if profile: t1 = time.perf_counter()
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        fov_rad = math.radians(fov)
        if profile: angle_time = time.perf_counter() - t1
        
        # Calculate focal length
        if profile: t2 = time.perf_counter()
        focal_length = output_width / (2 * math.tan(fov_rad / 2))
        if profile: focal_time = time.perf_counter() - t2
        
        # Create coordinate grids
        if profile: t3 = time.perf_counter()
        x_grid, y_grid = np.meshgrid(np.arange(output_width), np.arange(output_height))
        if profile: meshgrid_time = time.perf_counter() - t3
        
        # Convert to normalized coordinates
        if profile: t4 = time.perf_counter()
        x_norm = (x_grid - output_width / 2) / focal_length
        y_norm = (y_grid - output_height / 2) / focal_length
        if profile: normalize_time = time.perf_counter() - t4
        
        # Create 3D direction vectors
        if profile: t5 = time.perf_counter()
        z = np.ones_like(x_norm)
        if profile: ones_time = time.perf_counter() - t5
        
        # Stack coordinates
        if profile: t6 = time.perf_counter()
        coords = np.stack([x_norm, y_norm, z], axis=-1)
        if profile: stack_time = time.perf_counter() - t6
        
        # Normalize direction vectors
        if profile: t7 = time.perf_counter()
        norm = np.linalg.norm(coords, axis=-1, keepdims=True)
        if profile: norm_calc_time = time.perf_counter() - t7
        
        if profile: t8 = time.perf_counter()
        coords = coords / norm
        if profile: norm_divide_time = time.perf_counter() - t8
        
        grid_time = time.perf_counter() - start_time if benchmark else None
        
        rotation_start = time.perf_counter() if benchmark else None
        # Apply rotations (in order: roll, pitch, yaw) - using original method
        coords = self.rotate_coords(coords, roll_rad, pitch_rad, yaw_rad, benchmark)
        rotation_time = time.perf_counter() - rotation_start if benchmark else None
        
        spherical_start = time.perf_counter() if benchmark else None
        
        # Convert to spherical coordinates
        if profile: t9 = time.perf_counter()
        x, y, z = coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]
        if profile: extract_time = time.perf_counter() - t9
        
        # Calculate spherical coordinates (original working method)
        if profile: t10 = time.perf_counter()
        theta = np.arctan2(x, z)  # Azimuth angle
        if profile: arctan2_time = time.perf_counter() - t10
        
        if profile: t11 = time.perf_counter()
        phi = np.arcsin(np.clip(y, -1, 1))  # Elevation angle (clamp to avoid domain errors)
        if profile: arcsin_time = time.perf_counter() - t11
        
        # Convert to equirectangular coordinates
        if profile: t12 = time.perf_counter()
        u = (theta + np.pi) / (2 * np.pi)  # 0 to 1
        v = (phi + np.pi/2) / np.pi        # 0 to 1
        if profile: equirect_time = time.perf_counter() - t12
        
        # Convert to pixel coordinates
        if profile: t13 = time.perf_counter()
        pixel_x = u * (frame_shape[1] - 1)
        pixel_y = v * (frame_shape[0] - 1)
        if profile: pixel_conv_time = time.perf_counter() - t13
        
        # Handle wrapping for x coordinates
        if profile: t14 = time.perf_counter()
        pixel_x = np.clip(pixel_x, 0, frame_shape[1] - 1)
        pixel_y = np.clip(pixel_y, 0, frame_shape[0] - 1)
        if profile: clip_time = time.perf_counter() - t14
        
        # Convert to float32 for OpenCV
        if profile: t15 = time.perf_counter()
        pixel_x = pixel_x.astype(np.float32)
        pixel_y = pixel_y.astype(np.float32)
        if profile: float32_time = time.perf_counter() - t15
        
        spherical_time = time.perf_counter() - spherical_start if benchmark else None
        
        total_time = time.perf_counter() - start_time if benchmark else None
        
        if benchmark:
            print(f"  Coordinate Generation - Total: {total_time*1000:.2f}ms")
            print(f"    Grid Setup: {grid_time*1000:.2f}ms | Rotation: {rotation_time*1000:.2f}ms | Spherical: {spherical_time*1000:.2f}ms")
            
            if profile:
                print(f"    Detailed Profile:")
                print(f"      Angles->radians: {angle_time*1000:.3f}ms")
                print(f"      Focal length: {focal_time*1000:.3f}ms") 
                print(f"      Meshgrid: {meshgrid_time*1000:.3f}ms")
                print(f"      Normalize coords: {normalize_time*1000:.3f}ms")
                print(f"      Create z=1: {ones_time*1000:.3f}ms")
                print(f"      Stack coords: {stack_time*1000:.3f}ms")
                print(f"      Norm calculation: {norm_calc_time*1000:.3f}ms")
                print(f"      Norm division: {norm_divide_time*1000:.3f}ms")
                print(f"      Extract x,y,z: {extract_time*1000:.3f}ms")
                print(f"      Arctan2: {arctan2_time*1000:.3f}ms")
                print(f"      Arcsin+clip: {arcsin_time*1000:.3f}ms")
                print(f"      Equirect coords: {equirect_time*1000:.3f}ms")
                print(f"      Pixel conversion: {pixel_conv_time*1000:.3f}ms")
                print(f"      Clipping: {clip_time*1000:.3f}ms")
                print(f"      Float32 cast: {float32_time*1000:.3f}ms")
        
        return pixel_x, pixel_y
    
    def _generate_coordinate_mapping_optimized(self, yaw, pitch, roll, fov, output_width, output_height, frame_shape, benchmark=False):
        """Optimized coordinate mapping generation with performance improvements"""
        start_time = time.perf_counter() if benchmark else None
        
        # Pre-calculate constants
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        half_fov = math.radians(fov) / 2
        focal_length = output_width / (2 * math.tan(half_fov))
        
        # Pre-calculate offsets
        cx = output_width * 0.5
        cy = output_height * 0.5
        inv_focal = 1.0 / focal_length
        
        grid_start = time.perf_counter() if benchmark else None
        
        # Use more efficient coordinate generation
        x_indices = np.arange(output_width, dtype=np.float32)
        y_indices = np.arange(output_height, dtype=np.float32)
        
        # Vectorized coordinate calculation
        x_norm = (x_indices - cx) * inv_focal
        y_norm = (y_indices - cy) * inv_focal
        
        # Create meshgrid more efficiently
        x_norm_grid = np.broadcast_to(x_norm[None, :], (output_height, output_width))
        y_norm_grid = np.broadcast_to(y_norm[:, None], (output_height, output_width))
        
        # Pre-allocate arrays
        coords = np.empty((output_height, output_width, 3), dtype=np.float32)
        coords[:, :, 0] = x_norm_grid
        coords[:, :, 1] = y_norm_grid
        coords[:, :, 2] = 1.0
        
        # Vectorized normalization
        norm_factor = 1.0 / np.sqrt(x_norm_grid*x_norm_grid + y_norm_grid*y_norm_grid + 1.0)
        coords[:, :, 0] *= norm_factor
        coords[:, :, 1] *= norm_factor
        coords[:, :, 2] *= norm_factor
        
        grid_time = time.perf_counter() - grid_start if benchmark else None
        
        rotation_start = time.perf_counter() if benchmark else None
        # Apply rotations using optimized method
        coords = self.rotate_coords_optimized(coords, roll_rad, pitch_rad, yaw_rad, benchmark)
        rotation_time = time.perf_counter() - rotation_start if benchmark else None
        
        spherical_start = time.perf_counter() if benchmark else None
        
        # Extract coordinates (avoid copying)
        x, y, z = coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]
        
        # Vectorized spherical coordinate calculation
        theta = np.arctan2(x, z)
        phi = np.arcsin(np.clip(y, -1.0, 1.0))
        
        # Direct pixel coordinate calculation
        inv_2pi = 1.0 / (2.0 * np.pi)
        inv_pi = 1.0 / np.pi
        
        pixel_x = (theta + np.pi) * inv_2pi * (frame_shape[1] - 1)
        pixel_y = (phi + np.pi * 0.5) * inv_pi * (frame_shape[0] - 1)
        
        # Clamp to valid ranges
        np.clip(pixel_x, 0, frame_shape[1] - 1, out=pixel_x)
        np.clip(pixel_y, 0, frame_shape[0] - 1, out=pixel_y)
        
        spherical_time = time.perf_counter() - spherical_start if benchmark else None
        
        total_time = time.perf_counter() - start_time if benchmark else None
        
        if benchmark:
            print(f"  Optimized Coordinate Generation - Total: {total_time*1000:.2f}ms | Grid: {grid_time*1000:.2f}ms | Rotation: {rotation_time*1000:.2f}ms | Spherical: {spherical_time*1000:.2f}ms")
        
        return pixel_x, pixel_y
    
    def _generate_coordinate_mapping_ultra_fast(self, yaw, pitch, roll, fov, output_width, output_height, frame_shape, benchmark=False):
        """Ultra-fast coordinate mapping generation using advanced Numba JIT compilation"""
        start_time = time.perf_counter() if benchmark else None
        
        # Convert to radians
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        fov_rad = math.radians(fov)
        
        # Pre-calculate constants
        focal_length = output_width / (2 * math.tan(fov_rad / 2))
        cx = output_width * 0.5
        cy = output_height * 0.5
        
        # Create rotation matrix elements
        cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)
        cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        
        # Combined rotation matrix elements (R = R_yaw @ R_pitch @ R_roll)
        # Correct symbolic calculation matching the working rotate_coords method
        R00 = cos_y * cos_r + sin_y * sin_r * sin_p
        R01 = cos_y * (-sin_r) + sin_y * cos_r * sin_p
        R02 = sin_y * cos_p
        R10 = sin_r * cos_p
        R11 = cos_r * cos_p
        R12 = -sin_p
        R20 = -sin_y * cos_r + cos_y * sin_r * sin_p
        R21 = -sin_y * (-sin_r) + cos_y * cos_r * sin_p
        R22 = cos_y * cos_p
        
        generation_start = time.perf_counter() if benchmark else None
        
        # Choose the fastest JIT implementation based on output size
        total_pixels = output_width * output_height
        
        if total_pixels > 1000000:  # Use ultra-optimized parallel for very large outputs
            pixel_x, pixel_y = generate_mapping_jit_ultra_parallel(
                output_width, output_height,
                focal_length, cx, cy,
                R00, R01, R02, R10, R11, R12, R20, R21, R22,
                frame_shape[0], frame_shape[1]
            )
        elif total_pixels > 500000:  # Use regular parallel for large outputs
            pixel_x, pixel_y = generate_mapping_jit_parallel(
                output_width, output_height,
                focal_length, cx, cy,
                R00, R01, R02, R10, R11, R12, R20, R21, R22,
                frame_shape[0], frame_shape[1]
            )
        elif total_pixels > 200000:  # Use ultra-optimized serial for medium outputs
            pixel_x, pixel_y = generate_mapping_jit_ultra(
                output_width, output_height,
                focal_length, cx, cy,
                R00, R01, R02, R10, R11, R12, R20, R21, R22,
                frame_shape[0], frame_shape[1]
            )
        else:  # Use regular serial for small outputs
            pixel_x, pixel_y = generate_mapping_jit(
                output_width, output_height,
                focal_length, cx, cy,
                R00, R01, R02, R10, R11, R12, R20, R21, R22,
                frame_shape[0], frame_shape[1]
            )
        
        generation_time = time.perf_counter() - generation_start if benchmark else None
        total_time = time.perf_counter() - start_time if benchmark else None
        
        if benchmark:
            method = "ultra-parallel" if total_pixels > 1000000 else "parallel" if total_pixels > 500000 else "ultra-serial" if total_pixels > 200000 else "serial"
            print(f"  Ultra-fast Coordinate Generation ({method}) - Total: {total_time*1000:.2f}ms | JIT Generation: {generation_time*1000:.2f}ms")
        
        return pixel_x, pixel_y
    
    def rotate_coords(self, coords, roll, pitch, yaw, benchmark=False):
        """Apply rotation matrices for roll, pitch, yaw (using original working method)"""
        matrix_start = time.perf_counter() if benchmark else None
        
        # Roll rotation (around z-axis)
        cos_r, sin_r = math.cos(roll), math.sin(roll)
        R_roll = np.array([[cos_r, -sin_r, 0],
                          [sin_r, cos_r, 0],
                          [0, 0, 1]], dtype=np.float64)
        
        # Pitch rotation (around x-axis)
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        R_pitch = np.array([[1, 0, 0],
                           [0, cos_p, -sin_p],
                           [0, sin_p, cos_p]], dtype=np.float64)
        
        # Yaw rotation (around y-axis)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        R_yaw = np.array([[cos_y, 0, sin_y],
                         [0, 1, 0],
                         [-sin_y, 0, cos_y]], dtype=np.float64)
        
        # Combined rotation matrix (order matters!)
        R = R_yaw @ R_pitch @ R_roll
        matrix_time = time.perf_counter() - matrix_start if benchmark else None
        
        multiply_start = time.perf_counter() if benchmark else None
        # Apply rotation
        original_shape = coords.shape
        coords_flat = coords.reshape(-1, 3)
        rotated_coords = (R @ coords_flat.T).T
        result = rotated_coords.reshape(original_shape)
        multiply_time = time.perf_counter() - multiply_start if benchmark else None
        
        if benchmark:
            print(f"    Rotation - Matrix: {matrix_time*1000:.2f}ms | Multiply: {multiply_time*1000:.2f}ms")
        
        return result
    
    def rotate_coords_optimized(self, coords, roll, pitch, yaw, benchmark=False):
        """Optimized rotation using the same logic as original but with vectorized operations"""
        matrix_start = time.perf_counter() if benchmark else None
        
        # Use the same matrix calculation as the original to ensure correctness
        # Roll rotation (around z-axis)
        cos_r, sin_r = math.cos(roll), math.sin(roll)
        R_roll = np.array([[cos_r, -sin_r, 0],
                          [sin_r, cos_r, 0],
                          [0, 0, 1]], dtype=np.float32)  # Use float32 for consistency
        
        # Pitch rotation (around x-axis)
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        R_pitch = np.array([[1, 0, 0],
                           [0, cos_p, -sin_p],
                           [0, sin_p, cos_p]], dtype=np.float32)
        
        # Yaw rotation (around y-axis)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        R_yaw = np.array([[cos_y, 0, sin_y],
                         [0, 1, 0],
                         [-sin_y, 0, cos_y]], dtype=np.float32)
        
        # Combined rotation matrix (same order as original!)
        R = R_yaw @ R_pitch @ R_roll
        
        matrix_time = time.perf_counter() - matrix_start if benchmark else None
        
        multiply_start = time.perf_counter() if benchmark else None
        
        # Apply rotation using vectorized operations (avoid reshaping for better performance)
        original_shape = coords.shape
        h, w = original_shape[:2]
        
        # Reshape for matrix multiplication but more efficiently
        coords_reshaped = coords.reshape(-1, 3)
        
        # Apply rotation (same as original but potentially faster due to float32)
        rotated_coords = (R @ coords_reshaped.T).T
        
        # Reshape back
        result = rotated_coords.reshape(original_shape)
        
        multiply_time = time.perf_counter() - multiply_start if benchmark else None
        
        if benchmark:
            print(f"    Optimized Rotation - Matrix: {matrix_time*1000:.2f}ms | Multiply: {multiply_time*1000:.2f}ms")
        
        return result
    
    def process_frame(self, frame_number, yaw, pitch, roll, fov, output_width=1920, output_height=1080):
        """Process a specific frame"""
        frame = self.get_frame(frame_number)
        
        if frame is None:
            return None
        
        return self.get_perspective_projection(frame, yaw, pitch, roll, fov, output_width, output_height)
    
    def create_video_projection(self, output_path, yaw, pitch, roll, fov, 
                               output_width=1920, output_height=1080, 
                               start_frame=0, end_frame=None, num_threads=4, use_preloading=True):
        """Create a video with perspective projection (optimized with threading and preloading)"""
        if end_frame is None:
            end_frame = self.frame_count
        
        # Pre-generate coordinate mapping for better performance
        dummy_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.get_perspective_projection(dummy_frame, yaw, pitch, roll, fov, output_width, output_height)
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (output_width, output_height))
        
        total_frames = min(end_frame, self.frame_count) - start_frame
        
        if use_preloading:
            # Process in chunks with preloading
            chunk_size = min(self._cache_size_limit, total_frames)
            
            for chunk_start in range(start_frame, min(end_frame, self.frame_count), chunk_size):
                chunk_end = min(chunk_start + chunk_size, min(end_frame, self.frame_count))
                
                # Preload chunk of frames
                print(f"Preloading frames {chunk_start} to {chunk_end-1}...")
                self.preload_frames(chunk_start, chunk_end - 1)
                
                # Process frames in this chunk
                for i in range(chunk_start, chunk_end):
                    frame = self.get_frame(i)
                    if frame is None:
                        continue
                    
                    # Project frame
                    projected_frame = self.get_perspective_projection(
                        frame, yaw, pitch, roll, fov, output_width, output_height
                    )
                    
                    out.write(projected_frame)
                    
                    # Progress indicator
                    progress = ((i - start_frame + 1) / total_frames) * 100
                    if i % 10 == 0:
                        print(f"Progress: {progress:.1f}% ({i - start_frame + 1}/{total_frames} frames)")
        else:
            # Original threading approach without preloading
            batch_size = max(1, num_threads * 2)
            
            for batch_start in range(start_frame, min(end_frame, self.frame_count), batch_size):
                batch_end = min(batch_start + batch_size, min(end_frame, self.frame_count))
                
                # Read frames in batch
                frames = []
                for i in range(batch_start, batch_end):
                    frame = self.get_frame(i)
                    if frame is not None:
                        frames.append((i, frame))
                
                # Process frames in parallel
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    def process_single_frame(frame_data):
                        frame_idx, frame = frame_data
                        return self.get_perspective_projection(
                            frame, yaw, pitch, roll, fov, output_width, output_height
                        )
                    
                    projected_frames = list(executor.map(process_single_frame, frames))
                
                # Write processed frames
                for projected_frame in projected_frames:
                    out.write(projected_frame)
                
                # Progress indicator
                progress = ((batch_end - start_frame) / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({batch_end - start_frame}/{total_frames} frames)")
        
        out.release()
        print(f"Video saved to: {output_path}")
    
    def interactive_viewer(self, benchmark=False):
        """Interactive viewer with keyboard controls"""
        yaw, pitch, roll, fov = 0, 0, 0, 90
        frame_idx = 0
        
        print("Controls:")
        print("A/D - Adjust yaw (left/right)")
        print("W/S - Adjust pitch (up/down)")
        print("Q/E - Adjust roll")
        print("Z/X - Adjust FOV")
        print("Left/Right arrows - Navigate frames")
        print("B - Toggle benchmark timing")
        print("P - Preload next 30 frames")
        print("C - Clear caches")
        print("I - Show cache info")
        print("ESC - Exit")
        
        while True:
            frame_start = time.perf_counter() if benchmark else None
            
            # Use optimized frame reading
            if benchmark:
                frame, read_info = self.get_frame(frame_idx, benchmark=True)
                read_time, cache_status = read_info if read_info else (0, "ERROR")
            else:
                frame = self.get_frame(frame_idx)
            
            if frame is None:
                break
            
            # Project frame
            projected = self.get_perspective_projection(frame, yaw, pitch, roll, fov, benchmark=benchmark)
            
            display_start = time.perf_counter() if benchmark else None
            # Display info
            info_text = f"Frame: {frame_idx} | Yaw: {yaw:.1f}° | Pitch: {pitch:.1f}° | Roll: {roll:.1f}° | FOV: {fov:.1f}°"
            cv2.putText(projected, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add cache info to display
            cache_info = f"Frame: {len(self._frame_cache)}/{self._frame_cache_limit} | Coord: {len(self._map_cache)}/{self._cache_size_limit}"
            cv2.putText(projected, cache_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('360 Video Projection', projected)
            display_time = time.perf_counter() - display_start if benchmark else None
            
            total_frame_time = time.perf_counter() - frame_start if benchmark else None
            
            if benchmark:
                fps = 1.0 / total_frame_time if total_frame_time > 0 else 0
                print(f"Frame Timing - Total: {total_frame_time*1000:.2f}ms ({fps:.1f} FPS) | Read {cache_status}: {read_time*1000:.2f}ms | Display: {display_time*1000:.2f}ms")
                print("-" * 80)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('a'):
                yaw -= 5
            elif key == ord('d'):
                yaw += 5
            elif key == ord('w'):
                pitch += 5
            elif key == ord('s'):
                pitch -= 5
            elif key == ord('q'):
                roll -= 5
            elif key == ord('e'):
                roll += 5
            elif key == ord('z'):
                fov = max(10, fov - 5)
            elif key == ord('x'):
                fov = min(120, fov + 5)
            elif key == ord('b'):
                benchmark = not benchmark
                print(f"Benchmark mode: {'ON' if benchmark else 'OFF'}")
            elif key == ord('p'):
                # Preload next frames
                end_frame = min(frame_idx + self._frame_cache_limit, self.frame_count - 1)
                self.preload_frames(frame_idx, end_frame)
            elif key == ord('c'):
                # Clear all caches
                self.clear_cache()
                print("All caches cleared")
            elif key == ord('i'):
                # Show cache info
                print(self.get_cache_info())
            elif key == 83:  # Right arrow
                frame_idx = min(frame_idx + 1, self.frame_count - 1)
            elif key == 81:  # Left arrow
                frame_idx = max(frame_idx - 1, 0)
            elif key == ord(' '):  # Space - auto play
                frame_idx = (frame_idx + 1) % self.frame_count
        
        cv2.destroyAllWindows()
    
    def clear_cache(self):
        """Clear coordinate and frame caches"""
        self._map_cache.clear()
        self._frame_cache.clear()
    
    def get_cache_info(self):
        """Get information about the caches"""
        frame_cache_mb = sum(frame.nbytes for frame in self._frame_cache.values()) / (1024 * 1024) if self._frame_cache else 0
        
        # Estimate coordinate cache size
        coord_cache_mb = 0
        if self._map_cache:
            sample_key = next(iter(self._map_cache.keys()))
            pixel_x, pixel_y = self._map_cache[sample_key]
            bytes_per_entry = pixel_x.nbytes + pixel_y.nbytes
            coord_cache_mb = (bytes_per_entry * len(self._map_cache)) / (1024 * 1024)
        
        return f"Frame cache: {len(self._frame_cache)} frames ({frame_cache_mb:.1f} MB) | Coord cache: {len(self._map_cache)} entries ({coord_cache_mb:.1f} MB)"
    
    def precompute_projection(self, yaw, pitch, roll, fov, output_width=1920, output_height=1080):
        """Precompute and cache coordinate mapping for given parameters"""
        dummy_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.get_perspective_projection(dummy_frame, yaw, pitch, roll, fov, output_width, output_height)
        
        # Show normalized angles
        norm_yaw, norm_pitch, norm_roll = self.normalize_angles(yaw, pitch, roll)
        if yaw != norm_yaw or pitch != norm_pitch or roll != norm_roll:
            print(f"Precomputed projection for yaw={yaw}, pitch={pitch}, roll={roll}, fov={fov}")
            print(f"  -> Normalized to yaw={norm_yaw:.1f}°, pitch={norm_pitch:.1f}°, roll={norm_roll:.1f}°, fov={fov}")
        else:
            print(f"Precomputed projection for yaw={yaw}, pitch={pitch}, roll={roll}, fov={fov}")
    
    def benchmark_performance(self, num_iterations=10, output_width=1920, output_height=1080):
        """Run comprehensive performance benchmarks"""
        print("=" * 80)
        print("PERFORMANCE BENCHMARK")
        print("=" * 80)
        
        # Get a sample frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret:
            print("Could not read frame for benchmarking")
            return
        
        print(f"Video: {frame.shape[1]}x{frame.shape[0]} -> Output: {output_width}x{output_height}")
        print(f"Running {num_iterations} iterations for each test...")
        print()
        
        # Test scenarios
        scenarios = [
            ("Static (0,0,0,90)", 0, 0, 0, 90),
            ("Yaw 45°", 45, 0, 0, 90),
            ("Pitch 45°", 0, 45, 0, 90),
            ("Combined (45,30,0,90)", 45, 30, 0, 90),
            ("Wide FOV (120°)", 0, 0, 0, 120),
            ("Narrow FOV (60°)", 0, 0, 0, 60),
            ("Angle normalization test (360,0,0,90)", 360, 0, 0, 90),  # Should hit same cache as (0,0,0,90)
            ("Angle normalization test (45,360,720,90)", 45, 360, 720, 90),  # Should normalize properly
        ]
        
        for scenario_name, yaw, pitch, roll, fov in scenarios:
            print(f"Testing {scenario_name}:")
            
            # Clear cache for fair comparison
            self.clear_cache()
            
            # First run (cache miss)
            start_time = time.perf_counter()
            projected = self.get_perspective_projection(frame, yaw, pitch, roll, fov, output_width, output_height, benchmark=True)
            first_run_time = time.perf_counter() - start_time
            
            # Subsequent runs (cache hit)
            cache_hit_times = []
            for i in range(num_iterations):
                start_time = time.perf_counter()
                projected = self.get_perspective_projection(frame, yaw, pitch, roll, fov, output_width, output_height)
                cache_hit_times.append(time.perf_counter() - start_time)
            
            avg_cache_hit = np.mean(cache_hit_times) * 1000
            std_cache_hit = np.std(cache_hit_times) * 1000
            min_cache_hit = np.min(cache_hit_times) * 1000
            max_cache_hit = np.max(cache_hit_times) * 1000
            
            print(f"  First run (cache miss): {first_run_time*1000:.2f}ms")
            print(f"  Cache hits avg: {avg_cache_hit:.2f}±{std_cache_hit:.2f}ms (min: {min_cache_hit:.2f}ms, max: {max_cache_hit:.2f}ms)")
            print(f"  Speedup: {first_run_time*1000/avg_cache_hit:.1f}x")
            print()
        
        # Memory usage
        cache_info = self.get_cache_info()
        print(f"Cache status: {cache_info}")
        
        # Estimate memory usage per cache entry
        if len(self._map_cache) > 0:
            sample_key = next(iter(self._map_cache.keys()))
            pixel_x, pixel_y = self._map_cache[sample_key]
            bytes_per_entry = pixel_x.nbytes + pixel_y.nbytes
            total_cache_mb = (bytes_per_entry * len(self._map_cache)) / (1024 * 1024)
            print(f"Estimated cache memory usage: {total_cache_mb:.1f} MB")
        
        print("=" * 80)
    
    def benchmark_coordinate_generation(self, output_width=1920, output_height=1080, iterations=5):
        """Compare performance between different coordinate generation methods"""
        print("=" * 80)
        print("COORDINATE GENERATION BENCHMARK")
        print("=" * 80)
        
        dummy_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        test_params = [
            (0, 0, 0, 90),     # Static
            (45, 30, 0, 90),   # Mixed angles
            (0, 0, 0, 120),    # Wide FOV
        ]
        
        total_pixels = output_width * output_height
        print(f"Output resolution: {output_width}x{output_height} ({total_pixels:,} pixels)")
        print(f"Testing with {iterations} iterations per method\n")
        
        for yaw, pitch, roll, fov in test_params:
            print(f"Testing angles: yaw={yaw}°, pitch={pitch}°, roll={roll}°, fov={fov}°")
            
            # Test ultra-fast version
            ultra_times = []
            for i in range(iterations):
                start = time.perf_counter()
                px1, py1 = self._generate_coordinate_mapping_ultra_fast(
                    yaw, pitch, roll, fov, output_width, output_height, dummy_frame.shape, benchmark=False
                )
                ultra_times.append(time.perf_counter() - start)
            
            ultra_avg = np.mean(ultra_times) * 1000
            print(f"  Ultra-fast:   {ultra_avg:.2f}ms ± {np.std(ultra_times)*1000:.2f}ms")
            
            # Calculate pixels per second
            pixels_per_second = total_pixels / (ultra_avg / 1000)
            print(f"  Throughput:   {pixels_per_second/1e6:.1f} million pixels/second")
            print()
        
        print("=" * 80)
    
    def test_rotation_consistency(self):
        """Test that original and optimized rotation give the same results"""
        print("Testing rotation consistency...")
        
        # Create test coordinates
        test_coords = np.random.rand(100, 100, 3).astype(np.float32)
        test_coords = test_coords / np.linalg.norm(test_coords, axis=-1, keepdims=True)
        
        # Test angles
        roll, pitch, yaw = math.radians(15), math.radians(30), math.radians(45)
        
        # Original rotation
        result1 = self.rotate_coords(test_coords.copy(), roll, pitch, yaw, benchmark=False)
        
        # Optimized rotation  
        result2 = self.rotate_coords_optimized(test_coords.copy(), roll, pitch, yaw, benchmark=False)
        
        # Check difference
        max_diff = np.max(np.abs(result1 - result2))
        
        print(f"Max difference between rotation methods: {max_diff:.8f}")
        
        if max_diff < 1e-6:
            print("✅ Rotation methods are consistent")
            return True
        else:
            print("❌ WARNING: Rotation methods differ significantly!")
            return False
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
    
    def get_frame(self, frame_number, benchmark=False):
        """Optimized frame reading with caching"""
        read_start = time.perf_counter() if benchmark else None
        
        # Check frame cache first
        if frame_number in self._frame_cache:
            frame = self._frame_cache[frame_number]
            cache_hit = True
        else:
            # Read from video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            
            if not ret:
                return None, None if benchmark else None
            
            # Cache the frame if we have space
            if len(self._frame_cache) < self._frame_cache_limit:
                self._frame_cache[frame_number] = frame.copy()
            elif len(self._frame_cache) >= self._frame_cache_limit:
                # Remove oldest cached frame to make space
                oldest_key = min(self._frame_cache.keys())
                del self._frame_cache[oldest_key]
                self._frame_cache[frame_number] = frame.copy()
            
            cache_hit = False
        
        read_time = time.perf_counter() - read_start if benchmark else None
        
        if benchmark:
            cache_status = "HIT" if cache_hit else "MISS"
            return frame, (read_time, cache_status)
        else:
            return frame
    
    def preload_frames(self, start_frame, end_frame):
        """Preload a range of frames into cache for faster access"""
        print(f"Preloading frames {start_frame} to {end_frame}...")
        
        # Clear existing cache
        self._frame_cache.clear()
        
        frames_to_load = min(end_frame - start_frame + 1, self._frame_cache_limit)
        
        for i in range(start_frame, start_frame + frames_to_load):
            if i >= self.frame_count:
                break
                
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            
            if ret:
                self._frame_cache[i] = frame.copy()
            
            if i % 10 == 0:
                print(f"  Loaded frame {i}/{start_frame + frames_to_load - 1}")
        
        print(f"Preloaded {len(self._frame_cache)} frames into cache")
    
    def normalize_angles(self, yaw, pitch, roll):
        """Normalize angles to canonical ranges for cache consistency"""
        # Normalize yaw to [0, 360)
        yaw = yaw % 360
        
        # Normalize pitch properly - only handle true overflow cases
        # First, normalize pitch to equivalent angle in [-180, 180)
        pitch = ((pitch + 180) % 360) - 180
        
        # Handle pitch overflow beyond valid range [-90, 90]
        if pitch > 90:
            # Pitch > 90: flip over the top
            pitch = 180 - pitch
            yaw = (yaw + 180) % 360
            roll = (roll + 180) % 360
        elif pitch < -90:
            # Pitch < -90: flip over the bottom
            pitch = -180 - pitch
            yaw = (yaw + 180) % 360
            roll = (roll + 180) % 360
        
        # Now pitch is guaranteed to be in [-90, 90]
        # Normalize roll to [0, 360)
        roll = roll % 360
        
        return yaw, pitch, roll

# Optimized functions using Numba for better performance
@jit(nopython=True, cache=True)
def fast_remap_coords(coords_flat, R_flat):
    """Fast coordinate rotation using numba"""
    result = np.zeros_like(coords_flat)
    for i in range(coords_flat.shape[0]):
        x, y, z = coords_flat[i, 0], coords_flat[i, 1], coords_flat[i, 2]
        result[i, 0] = R_flat[0] * x + R_flat[1] * y + R_flat[2] * z
        result[i, 1] = R_flat[3] * x + R_flat[4] * y + R_flat[5] * z
        result[i, 2] = R_flat[6] * x + R_flat[7] * y + R_flat[8] * z
    return result

@njit(cache=True, fastmath=True)
def generate_mapping_jit(output_width, output_height, focal_length, cx, cy,
                        R00, R01, R02, R10, R11, R12, R20, R21, R22,
                        frame_height, frame_width):
    """JIT-compiled coordinate mapping generation for maximum speed"""
    
    # Pre-allocate output arrays
    pixel_x = np.empty((output_height, output_width), dtype=np.float32)
    pixel_y = np.empty((output_height, output_width), dtype=np.float32)
    
    # Pre-calculate constants
    inv_focal = 1.0 / focal_length
    inv_2pi = 1.0 / (2.0 * np.pi)
    inv_pi = 1.0 / np.pi
    half_pi = np.pi * 0.5
    frame_width_minus_1 = frame_width - 1
    frame_height_minus_1 = frame_height - 1
    
    # Process each pixel
    for j in range(output_height):
        y_norm = (j - cy) * inv_focal
        for i in range(output_width):
            x_norm = (i - cx) * inv_focal
            
            # Normalize direction vector
            norm_factor = 1.0 / math.sqrt(x_norm * x_norm + y_norm * y_norm + 1.0)
            x_unit = x_norm * norm_factor
            y_unit = y_norm * norm_factor
            z_unit = norm_factor
            
            # Apply rotation matrix
            x_rot = R00 * x_unit + R01 * y_unit + R02 * z_unit
            y_rot = R10 * x_unit + R11 * y_unit + R12 * z_unit
            z_rot = R20 * x_unit + R21 * y_unit + R22 * z_unit
            
            # Convert to spherical coordinates
            theta = math.atan2(x_rot, z_rot)  # Azimuth
            phi = math.asin(max(-1.0, min(1.0, y_rot)))  # Elevation (clamped)
            
            # Convert to pixel coordinates
            u = (theta + np.pi) * inv_2pi
            v = (phi + half_pi) * inv_pi
            
            pixel_x[j, i] = max(0.0, min(frame_width_minus_1, u * frame_width_minus_1))
            pixel_y[j, i] = max(0.0, min(frame_height_minus_1, v * frame_height_minus_1))
    
    return pixel_x, pixel_y

@njit(cache=True, fastmath=True, parallel=True)
def generate_mapping_jit_parallel(output_width, output_height, focal_length, cx, cy,
                                 R00, R01, R02, R10, R11, R12, R20, R21, R22,
                                 frame_height, frame_width):
    """Parallel JIT-compiled coordinate mapping for multi-core systems"""
    
    # Pre-allocate output arrays
    pixel_x = np.empty((output_height, output_width), dtype=np.float32)
    pixel_y = np.empty((output_height, output_width), dtype=np.float32)
    
    # Pre-calculate constants
    inv_focal = 1.0 / focal_length
    inv_2pi = 1.0 / (2.0 * np.pi)
    inv_pi = 1.0 / np.pi
    half_pi = np.pi * 0.5
    frame_width_minus_1 = frame_width - 1
    frame_height_minus_1 = frame_height - 1
    
    # Process rows in parallel
    for j in range(output_height):
        y_norm = (j - cy) * inv_focal
        for i in range(output_width):
            x_norm = (i - cx) * inv_focal
            
            # Normalize direction vector
            norm_factor = 1.0 / math.sqrt(x_norm * x_norm + y_norm * y_norm + 1.0)
            x_unit = x_norm * norm_factor
            y_unit = y_norm * norm_factor
            z_unit = norm_factor
            
            # Apply rotation matrix
            x_rot = R00 * x_unit + R01 * y_unit + R02 * z_unit
            y_rot = R10 * x_unit + R11 * y_unit + R12 * z_unit
            z_rot = R20 * x_unit + R21 * y_unit + R22 * z_unit
            
            # Convert to spherical coordinates
            theta = math.atan2(x_rot, z_rot)
            phi = math.asin(max(-1.0, min(1.0, y_rot)))
            
            # Convert to pixel coordinates
            u = (theta + np.pi) * inv_2pi
            v = (phi + half_pi) * inv_pi
            
            pixel_x[j, i] = max(0.0, min(frame_width_minus_1, u * frame_width_minus_1))
            pixel_y[j, i] = max(0.0, min(frame_height_minus_1, v * frame_height_minus_1))
    
    return pixel_x, pixel_y

# Advanced optimized functions for maximum performance
@njit(cache=True, fastmath=True, inline='always')
def generate_mapping_jit_ultra(output_width, output_height, focal_length, cx, cy,
                              R00, R01, R02, R10, R11, R12, R20, R21, R22,
                              frame_height, frame_width):
    """Ultra-optimized coordinate mapping with memory layout optimization"""
    
    # Pre-allocate output arrays with optimal memory layout
    pixel_x = np.empty((output_height, output_width), dtype=np.float32)
    pixel_y = np.empty((output_height, output_width), dtype=np.float32)
    
    # Pre-calculate all constants (more than before)
    inv_focal = 1.0 / focal_length
    inv_2pi = 0.15915494309189535  # 1/(2*pi) precomputed
    inv_pi = 0.3183098861837907    # 1/pi precomputed
    half_pi = 1.5707963267948966   # pi/2 precomputed
    frame_width_f = float(frame_width - 1)
    frame_height_f = float(frame_height - 1)
    
    # Process each pixel with optimized inner loop
    for j in range(output_height):
        y_norm = (j - cy) * inv_focal
        y_norm_sq = y_norm * y_norm
        for i in range(output_width):
            x_norm = (i - cx) * inv_focal
            
            # Optimized normalization using precomputed y_norm_sq
            norm_factor = 1.0 / math.sqrt(x_norm * x_norm + y_norm_sq + 1.0)
            x_unit = x_norm * norm_factor
            y_unit = y_norm * norm_factor
            z_unit = norm_factor
            
            # Apply rotation matrix (unrolled for speed)
            x_rot = R00 * x_unit + R01 * y_unit + R02 * z_unit
            y_rot = R10 * x_unit + R11 * y_unit + R12 * z_unit
            z_rot = R20 * x_unit + R21 * y_unit + R22 * z_unit
            
            # Optimized spherical coordinate calculation
            theta = math.atan2(x_rot, z_rot)
            phi = math.asin(max(-1.0, min(1.0, y_rot)))
            
            # Direct pixel coordinate calculation with precomputed constants
            u = (theta + math.pi) * inv_2pi
            v = (phi + half_pi) * inv_pi
            
            # Final pixel coordinates with bounds checking
            pixel_x[j, i] = max(0.0, min(frame_width_f, u * frame_width_f))
            pixel_y[j, i] = max(0.0, min(frame_height_f, v * frame_height_f))
    
    return pixel_x, pixel_y

@njit(cache=True, fastmath=True, parallel=True)
def generate_mapping_jit_ultra_parallel(output_width, output_height, focal_length, cx, cy,
                                       R00, R01, R02, R10, R11, R12, R20, R21, R22,
                                       frame_height, frame_width):
    """Ultra-optimized parallel coordinate mapping for multi-core systems"""
    
    # Pre-allocate output arrays
    pixel_x = np.empty((output_height, output_width), dtype=np.float32)
    pixel_y = np.empty((output_height, output_width), dtype=np.float32)
    
    # Pre-calculate constants
    inv_focal = 1.0 / focal_length
    inv_2pi = 0.15915494309189535
    inv_pi = 0.3183098861837907
    half_pi = 1.5707963267948966
    frame_width_f = float(frame_width - 1)
    frame_height_f = float(frame_height - 1)
    pi = 3.141592653589793
    
    # Parallel processing with optimized inner loop
    for j in range(output_height):
        y_norm = (j - cy) * inv_focal
        y_norm_sq = y_norm * y_norm
        for i in range(output_width):
            x_norm = (i - cx) * inv_focal
            
            # Fast normalization
            norm_factor = 1.0 / math.sqrt(x_norm * x_norm + y_norm_sq + 1.0)
            x_unit = x_norm * norm_factor
            y_unit = y_norm * norm_factor
            z_unit = norm_factor
            
            # Matrix multiply
            x_rot = R00 * x_unit + R01 * y_unit + R02 * z_unit
            y_rot = R10 * x_unit + R11 * y_unit + R12 * z_unit
            z_rot = R20 * x_unit + R21 * y_unit + R22 * z_unit
            
            # Spherical conversion
            theta = math.atan2(x_rot, z_rot)
            phi = math.asin(max(-1.0, min(1.0, y_rot)))
            
            # Pixel mapping
            u = (theta + pi) * inv_2pi
            v = (phi + half_pi) * inv_pi
            
            pixel_x[j, i] = max(0.0, min(frame_width_f, u * frame_width_f))
            pixel_y[j, i] = max(0.0, min(frame_height_f, v * frame_height_f))
    
    return pixel_x, pixel_y

# Example usage
def main():
    # Replace with your video path
    video_path = "C:/insta360/x5/exports/VID_20250704_123015_00_001(1).mp4"
 
    # Create processor with ultra-fast coordinate generation and minimal memory cache
    processor = Equirectangular360(video_path, use_optimized_coords=True, max_memory_cache_mb=50)
    
    print(f"Video loaded: {processor.width}x{processor.height}")
    print("Ultra-fast coordinate generation enabled with JIT compilation")
    print("Minimal memory cache (no disk storage) for maximum speed")
    print()
    
    # Test coordinate generation performance first
    print("Testing coordinate generation performance...")
    processor.benchmark_coordinate_generation()
    print()
    
    # Option 1: Interactive viewer with real-time performance
    print("Starting interactive viewer...")
    print("Press 'B' during viewing to toggle benchmark mode")
    processor.interactive_viewer()
    
    # Option 2: Run performance benchmark
    # processor.benchmark_performance()
    
    # Option 3: Process single frame
    # frame = processor.process_frame(
    #     frame_number=0,
    #     yaw=0, pitch=0, roll=0, fov=90
    # )
    # if frame is not None:
    #     cv2.imshow('Projected Frame', frame)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

if __name__ == "__main__":
    main()