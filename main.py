import cv2
import numpy as np
import math
from numba import jit
import threading
from concurrent.futures import ThreadPoolExecutor
import time

class Equirectangular360:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Optimize video capture settings for speed
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent lag
        
        # Cache for coordinate mappings
        self._map_cache = {}
        
        # Frame cache for faster sequential access
        self._frame_cache = {}
        self._cache_size_limit = 50  # Limit cache to prevent memory issues
        
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
        # Check if we have cached mapping for these parameters
        if cache_key in self._map_cache:
            pixel_x, pixel_y = self._map_cache[cache_key]
            cache_hit = True
        else:
            # Generate new mapping and cache it using normalized angles
            pixel_x, pixel_y = self._generate_coordinate_mapping(
                norm_yaw, norm_pitch, norm_roll, fov, output_width, output_height, frame.shape, benchmark
            )
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
        """Generate coordinate mapping for remapping (using original logic)"""
        start_time = time.perf_counter() if benchmark else None
        
        # Convert angles to radians
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        fov_rad = math.radians(fov)
        
        # Calculate focal length
        focal_length = output_width / (2 * math.tan(fov_rad / 2))
        
        grid_start = time.perf_counter() if benchmark else None
        # Create coordinate grids
        x_grid, y_grid = np.meshgrid(np.arange(output_width), np.arange(output_height))
        
        # Convert to normalized coordinates (-1 to 1)
        x_norm = (x_grid - output_width / 2) / focal_length
        y_norm = (y_grid - output_height / 2) / focal_length
        
        # Create 3D direction vectors
        z = np.ones_like(x_norm)
        
        # Stack coordinates
        coords = np.stack([x_norm, y_norm, z], axis=-1)
        
        # Normalize direction vectors
        norm = np.linalg.norm(coords, axis=-1, keepdims=True)
        coords = coords / norm
        grid_time = time.perf_counter() - grid_start if benchmark else None
        
        rotation_start = time.perf_counter() if benchmark else None
        # Apply rotations (in order: roll, pitch, yaw) - using original method
        coords = self.rotate_coords(coords, roll_rad, pitch_rad, yaw_rad, benchmark)
        rotation_time = time.perf_counter() - rotation_start if benchmark else None
        
        spherical_start = time.perf_counter() if benchmark else None
        # Convert to spherical coordinates
        x, y, z = coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]
        
        # Calculate spherical coordinates (original working method)
        theta = np.arctan2(x, z)  # Azimuth angle
        phi = np.arcsin(np.clip(y, -1, 1))  # Elevation angle (clamp to avoid domain errors)
        
        # Convert to equirectangular coordinates
        u = (theta + np.pi) / (2 * np.pi)  # 0 to 1
        v = (phi + np.pi/2) / np.pi        # 0 to 1
        
        # Convert to pixel coordinates
        pixel_x = u * (frame_shape[1] - 1)
        pixel_y = v * (frame_shape[0] - 1)
        
        # Handle wrapping for x coordinates
        pixel_x = np.clip(pixel_x, 0, frame_shape[1] - 1)
        pixel_y = np.clip(pixel_y, 0, frame_shape[0] - 1)
        
        # Convert to float32 for OpenCV
        pixel_x = pixel_x.astype(np.float32)
        pixel_y = pixel_y.astype(np.float32)
        spherical_time = time.perf_counter() - spherical_start if benchmark else None
        
        total_time = time.perf_counter() - start_time if benchmark else None
        
        if benchmark:
            print(f"  Coordinate Generation - Total: {total_time*1000:.2f}ms | Grid: {grid_time*1000:.2f}ms | Rotation: {rotation_time*1000:.2f}ms | Spherical: {spherical_time*1000:.2f}ms")
        
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
        print("P - Preload next 50 frames")
        print("C - Clear frame cache")
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
            cache_info = f"Frame Cache: {len(self._frame_cache)}/{self._cache_size_limit} | Coord Cache: {len(self._map_cache)}"
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
                end_frame = min(frame_idx + self._cache_size_limit, self.frame_count - 1)
                self.preload_frames(frame_idx, end_frame)
            elif key == ord('c'):
                # Clear frame cache
                self._frame_cache.clear()
                print("Frame cache cleared")
            elif key == 83:  # Right arrow
                frame_idx = min(frame_idx + 1, self.frame_count - 1)
            elif key == 81:  # Left arrow
                frame_idx = max(frame_idx - 1, 0)
            elif key == ord(' '):  # Space - auto play
                frame_idx = (frame_idx + 1) % self.frame_count
        
        cv2.destroyAllWindows()
    
    def clear_cache(self):
        """Clear both coordinate and frame caches"""
        self._map_cache.clear()
        self._frame_cache.clear()
    
    def get_cache_info(self):
        """Get information about the caches"""
        frame_cache_mb = sum(frame.nbytes for frame in self._frame_cache.values()) / (1024 * 1024) if self._frame_cache else 0
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
            if len(self._frame_cache) < self._cache_size_limit:
                self._frame_cache[frame_number] = frame.copy()
            elif len(self._frame_cache) >= self._cache_size_limit:
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
        
        frames_to_load = min(end_frame - start_frame + 1, self._cache_size_limit)
        
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

# Example usage
def main():
    # Replace with your video path
    video_path = "C:/insta360/x5/exports/VID_20250704_123015_00_001(1).mp4"
 
    # Create processor
    processor = Equirectangular360(video_path)
    
    print(f"Video loaded: {processor.width}x{processor.height}")
    print("Optimizations enabled: coordinate caching + multithreading")
    print()
    
    # Run benchmark first
    print("Running performance benchmark...")
    processor.benchmark_performance()
    
    # Option 1: Interactive viewer (now with benchmarking support)
    # Press 'B' during interactive viewing to toggle benchmark mode
    processor.interactive_viewer()
    
    # Option 2: Process single frame (with precomputation for speed)
    # print("Precomputing projection...")
    # processor.precompute_projection(yaw=0, pitch=0, roll=0, fov=90)
    # 
    # frame = processor.process_frame(
    #     frame_number=0,
    #     yaw=0,      # Look straight ahead
    #     pitch=0,    # Level horizon
    #     roll=0,     # No tilt
    #     fov=90      # 90 degree field of view
    # )
    # 
    # if frame is not None:
    #     cv2.imshow('Projected Frame', frame)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    
    # Option 3: Create projected video (now with multithreading)
    # processor.create_video_projection(
    #     output_path="projected_video.mp4",
    #     yaw=45,     # Look 45 degrees to the right
    #     pitch=0,    # Level horizon
    #     roll=0,     # No tilt
    #     fov=90,     # 90 degree field of view
    #     num_threads=4  # Use 4 threads for processing
    # )
    # 
    # print(processor.get_cache_info())

if __name__ == "__main__":
    main()