import cv2
import numpy as np
import math
from numba import jit, njit, prange
import time
from concurrent.futures import ThreadPoolExecutor

# Set Numba compilation cache for faster startup
import numba
numba.config.CACHE_DIR = './numba_cache'

class Equirectangular360:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
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
                fov = min(360, fov + 5)
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
    
    # Process rows in parallel using prange
    for j in prange(output_height):
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
    """Ultra-optimized parallel coordinate mapping using a flattened approach for better Numba parallelization"""
    
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
    
    # Flatten to 1D for better parallelization
    total_pixels = output_height * output_width
    
    # Use prange for parallel processing of individual pixels
    for idx in prange(total_pixels):
        # Convert 1D index back to 2D coordinates
        j = idx // output_width
        i = idx % output_width
        
        # Same calculation as before
        x_norm = (i - cx) * inv_focal
        y_norm = (j - cy) * inv_focal
        
        # Fast normalization
        norm_factor = 1.0 / math.sqrt(x_norm * x_norm + y_norm * y_norm + 1.0)
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
    processor = Equirectangular360(video_path)
    
    print(f"Video loaded: {processor.width}x{processor.height}")
    print("Ultra-fast coordinate generation enabled with JIT compilation")
    print("Minimal memory cache (no disk storage) for maximum speed")
    print()
    
    # Interactive viewer with real-time performance
    print("Starting interactive viewer...")
    print("Press 'B' during viewing to toggle benchmark mode")
    processor.interactive_viewer()

if __name__ == "__main__":
    main()