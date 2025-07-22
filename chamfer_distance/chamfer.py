"""
Chamfer Distance Comparison Script for YCB Dataset
Compares ground truth laser scans with NeRF-generated point clouds

Date: July 2025
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib.pyplot as plt


class PointCloudComparator:
    """Class for comparing ground truth and generated point clouds using Chamfer distance."""
    
    def __init__(self, log_level: str = "INFO"):
        """Initialize the comparator with logging setup."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_point_cloud(self, filepath: str, file_format: Optional[str] = None) -> o3d.geometry.PointCloud:
        """
        Load a point cloud from various file formats.
        
        Args:
            filepath: Path to the point cloud file
            file_format: Format of the file (auto-detected if None)
            
        Returns:
            Open3D PointCloud object
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Point cloud file not found: {filepath}")
                
            # Auto-detect format based on extension
            ext = Path(filepath).suffix.lower()
            
            if ext in ['.ply', '.pcd', '.xyz', '.xyzn', '.xyzrgb']:
                pcd = o3d.io.read_point_cloud(filepath)
            elif ext in ['.obj']:
                mesh = o3d.io.read_triangle_mesh(filepath)
                pcd = mesh.sample_points_uniformly(number_of_points=10000)
            elif ext == '.npy':
                # Load numpy array (assuming Nx3 format)
                points = np.load(filepath)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
                
            if len(pcd.points) == 0:
                raise ValueError(f"Point cloud is empty: {filepath}")
                
            self.logger.info(f"Loaded point cloud with {len(pcd.points)} points from {filepath}")
            return pcd
            
        except Exception as e:
            self.logger.error(f"Error loading point cloud {filepath}: {str(e)}")
            raise
    
    def preprocess_point_cloud(self, pcd: o3d.geometry.PointCloud, 
                             voxel_size: Optional[float] = None,
                             remove_outliers: bool = True,
                             nb_neighbors: int = 20,
                             std_ratio: float = 2.0,
                             max_points: Optional[int] = None) -> o3d.geometry.PointCloud:
        """
        Preprocess point cloud with optional downsampling and outlier removal.
        
        Args:
            pcd: Input point cloud
            voxel_size: Voxel size for downsampling (None to skip)
            remove_outliers: Whether to remove statistical outliers
            nb_neighbors: Number of neighbors for outlier detection
            std_ratio: Standard deviation ratio for outlier detection
            max_points: Maximum number of points to keep (random sampling if exceeded)
            
        Returns:
            Preprocessed point cloud
        """
        processed_pcd = pcd
        original_points = len(processed_pcd.points)
        
        # Downsample if voxel size is specified
        if voxel_size is not None:
            processed_pcd = processed_pcd.voxel_down_sample(voxel_size)
            self.logger.info(f"Downsampled from {original_points} to {len(processed_pcd.points)} points")
        
        # Remove outliers
        if remove_outliers:
            processed_pcd, _ = processed_pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors, std_ratio=std_ratio
            )
            self.logger.info(f"After outlier removal: {len(processed_pcd.points)} points")
        
        # Additional random downsampling if still too many points
        if max_points is not None and len(processed_pcd.points) > max_points:
            indices = np.random.choice(len(processed_pcd.points), max_points, replace=False)
            processed_pcd = processed_pcd.select_by_index(indices)
            self.logger.info(f"Random downsampling to {len(processed_pcd.points)} points")
            
        return processed_pcd
    
    def align_point_clouds(self, source: o3d.geometry.PointCloud, 
                          target: o3d.geometry.PointCloud,
                          use_icp: bool = True,
                          threshold: float = 0.02) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        Align two point clouds using ICP registration.
        
        Args:
            source: Source point cloud to be aligned
            target: Target point cloud (reference)
            use_icp: Whether to use ICP alignment
            threshold: ICP convergence threshold
            
        Returns:
            Tuple of (aligned_source, transformation_matrix)
        """
        if not use_icp:
            return source, np.eye(4)
            
        try:
            # Estimate normals for better ICP performance
            source.estimate_normals()
            target.estimate_normals()
            
            # Initial alignment using feature matching
            trans_init = np.eye(4)
            
            # ICP registration
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source, target, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
            )
            
            # Apply transformation
            aligned_source = source.transform(reg_p2p.transformation)
            
            self.logger.info(f"ICP fitness: {reg_p2p.fitness:.4f}, RMSE: {reg_p2p.inlier_rmse:.4f}")
            
            return aligned_source, reg_p2p.transformation
            
        except Exception as e:
            self.logger.warning(f"ICP alignment failed: {str(e)}. Using identity transformation.")
            return source, np.eye(4)
    
    def compute_chamfer_distance(self, pcd1: o3d.geometry.PointCloud, 
                               pcd2: o3d.geometry.PointCloud,
                               chunk_size: int = 1000) -> Dict[str, Any]:
        """
        Compute bidirectional Chamfer distance between two point clouds using chunked processing.
        
        Args:
            pcd1: First point cloud
            pcd2: Second point cloud
            chunk_size: Size of chunks for memory-efficient processing
            
        Returns:
            Dictionary containing chamfer distance metrics
        """
        points1 = np.asarray(pcd1.points)
        points2 = np.asarray(pcd2.points)
        
        self.logger.info(f"Computing Chamfer distance between {len(points1)} and {len(points2)} points")
        
        # Memory-efficient computation using chunked processing
        def compute_min_distances_chunked(source_points, target_points, chunk_size):
            """Compute minimum distances using chunked processing to save memory."""
            min_distances = []
            n_chunks = (len(source_points) + chunk_size - 1) // chunk_size
            
            for i in tqdm(range(n_chunks), desc="Processing chunks"):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(source_points))
                chunk = source_points[start_idx:end_idx]
                
                # Compute distances for this chunk
                distances = cdist(chunk, target_points, metric='euclidean')
                chunk_min_distances = np.min(distances, axis=1)
                min_distances.extend(chunk_min_distances)
                
            return np.array(min_distances)
        
        # Estimate memory usage and adjust chunk size if needed
        estimated_memory_gb = (len(points1) * len(points2) * 8) / (1024**3)  # 8 bytes per float64
        if estimated_memory_gb > 4:  # If estimated memory > 4GB
            # Adjust chunk size to use roughly 1GB of memory per chunk
            max_chunk_size = int(1024**3 / (len(points2) * 8))  # 1GB / (target_points * 8 bytes)
            chunk_size = min(chunk_size, max(max_chunk_size, 100))  # At least 100 points per chunk
            self.logger.warning(f"Large point clouds detected. Using chunk size of {chunk_size} to manage memory.")
        
        # Compute distances from pcd1 to pcd2
        self.logger.info("Computing distances from first point cloud to second...")
        min_distances_1_to_2 = compute_min_distances_chunked(points1, points2, chunk_size)
        
        # Compute distances from pcd2 to pcd1  
        self.logger.info("Computing distances from second point cloud to first...")
        min_distances_2_to_1 = compute_min_distances_chunked(points2, points1, chunk_size)
        
        # Chamfer distance is the mean of minimum distances in both directions
        chamfer_1_to_2 = np.mean(min_distances_1_to_2)
        chamfer_2_to_1 = np.mean(min_distances_2_to_1)
        chamfer_distance = (chamfer_1_to_2 + chamfer_2_to_1) / 2
        
        # Additional metrics
        max_distance_1_to_2 = np.max(min_distances_1_to_2)
        max_distance_2_to_1 = np.max(min_distances_2_to_1)
        
        results = {
            'chamfer_distance': chamfer_distance,
            'chamfer_1_to_2': chamfer_1_to_2,
            'chamfer_2_to_1': chamfer_2_to_1,
            'max_distance_1_to_2': max_distance_1_to_2,
            'max_distance_2_to_1': max_distance_2_to_1,
            'rmse_1_to_2': np.sqrt(np.mean(min_distances_1_to_2**2)),
            'rmse_2_to_1': np.sqrt(np.mean(min_distances_2_to_1**2))
        }
        
        return results
    
    def visualize_comparison(self, gt_pcd: o3d.geometry.PointCloud, 
                           gen_pcd: o3d.geometry.PointCloud,
                           save_path: Optional[str] = None):
        """
        Visualize the comparison between ground truth and generated point clouds.
        
        Args:
            gt_pcd: Ground truth point cloud
            gen_pcd: Generated point cloud
            save_path: Path to save the visualization
        """
        # Color the point clouds differently
        gt_pcd.paint_uniform_color([1, 0, 0])  # Red for ground truth
        gen_pcd.paint_uniform_color([0, 0, 1])  # Blue for generated
        
        # Log basic statistics instead of complex visualization
        self.logger.info(f"Ground truth points: {len(gt_pcd.points)}")
        self.logger.info(f"Generated points: {len(gen_pcd.points)}")
        
        if save_path:
            self.logger.info(f"Visualization save path specified: {save_path}")
            # Note: In a headless environment, actual visualization might not work
            # This is a placeholder for where screenshot saving would happen
    
    def compare_single_pair(self, gt_path: str, gen_path: str,
                          align: bool = True,
                          voxel_size: Optional[float] = None,
                          visualize: bool = False,
                          output_dir: Optional[str] = None,
                          max_points_per_cloud: Optional[int] = None,
                          auto_downsample: bool = True) -> Dict[str, Any]:
        """
        Compare a single pair of ground truth and generated point clouds.
        
        Args:
            gt_path: Path to ground truth point cloud
            gen_path: Path to generated point cloud
            align: Whether to align point clouds
            voxel_size: Voxel size for preprocessing
            visualize: Whether to show visualization
            output_dir: Directory to save results
            max_points_per_cloud: Maximum points per cloud (for memory management)
            auto_downsample: Automatically downsample large point clouds
            
        Returns:
            Dictionary containing comparison metrics
        """
        self.logger.info(f"Comparing {gt_path} vs {gen_path}")
        
        # Load point clouds
        gt_pcd = self.load_point_cloud(gt_path)
        gen_pcd = self.load_point_cloud(gen_path)
        
        # Auto-determine memory-safe limits
        if auto_downsample and max_points_per_cloud is None:
            total_points = len(gt_pcd.points) + len(gen_pcd.points)
            # If combined points would create a matrix > 1GB, limit each cloud
            if total_points > 100000:  # Conservative limit
                max_points_per_cloud = 50000
                self.logger.warning(f"Large point clouds detected ({total_points} total points). "
                                  f"Auto-limiting to {max_points_per_cloud} points per cloud.")
        
        # Preprocess with memory limits
        gt_pcd = self.preprocess_point_cloud(gt_pcd, voxel_size=voxel_size, 
                                           max_points=max_points_per_cloud)
        gen_pcd = self.preprocess_point_cloud(gen_pcd, voxel_size=voxel_size,
                                            max_points=max_points_per_cloud)
        
        # Align if requested
        if align:
            gen_pcd, transformation = self.align_point_clouds(gen_pcd, gt_pcd)
        
        # Compute chamfer distance
        metrics = self.compute_chamfer_distance(gt_pcd, gen_pcd)
        
        # Add metadata
        metrics['gt_points'] = len(gt_pcd.points)
        metrics['gen_points'] = len(gen_pcd.points)
        metrics['gt_path'] = gt_path
        metrics['gen_path'] = gen_path
        
        # Create visualization  
        if visualize:
            vis_path = None
            if output_dir:
                vis_path = os.path.join(output_dir, f"comparison_{Path(gt_path).stem}.png")
            self.visualize_comparison(gt_pcd, gen_pcd, vis_path)
        
        self.logger.info(f"Chamfer distance: {metrics['chamfer_distance']:.6f}")
        
        return metrics
    
    def batch_compare(self, gt_dir: str, gen_dir: str,
                     output_file: str = "chamfer_results.json",
                     **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Batch compare multiple pairs of point clouds.
        
        Args:
            gt_dir: Directory containing ground truth point clouds
            gen_dir: Directory containing generated point clouds
            output_file: Path to save results JSON
            **kwargs: Additional arguments for compare_single_pair
            
        Returns:
            Dictionary containing all comparison results
        """
        gt_files = list(Path(gt_dir).glob("*.ply")) + list(Path(gt_dir).glob("*.pcd"))
        results = {}
        
        self.logger.info(f"Found {len(gt_files)} ground truth files")
        
        for gt_file in tqdm(gt_files, desc="Comparing point clouds"):
            # Find corresponding generated file
            stem = gt_file.stem
            gen_candidates = [
                Path(gen_dir) / f"{stem}.ply",
                Path(gen_dir) / f"{stem}.pcd",
                Path(gen_dir) / f"{stem}_generated.ply",
                Path(gen_dir) / f"{stem}_nerf.ply"
            ]
            
            gen_file = None
            for candidate in gen_candidates:
                if candidate.exists():
                    gen_file = str(candidate)
                    break
            
            if gen_file is None:
                self.logger.warning(f"No generated point cloud found for {stem}")
                continue
            
            try:
                metrics = self.compare_single_pair(
                    str(gt_file), gen_file, **kwargs
                )
                results[stem] = metrics
                
            except Exception as e:
                self.logger.error(f"Error comparing {stem}: {str(e)}")
                continue
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_file}")
        
        # Print summary statistics
        self.print_summary_stats(results)
        
        return results
    
    def print_summary_stats(self, results: Dict[str, Dict[str, Any]]):
        """Print summary statistics of the comparison results."""
        if not results:
            self.logger.warning("No results to summarize")
            return
        
        chamfer_distances = [r['chamfer_distance'] for r in results.values()]
        
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Number of comparisons: {len(results)}")
        print(f"Mean Chamfer distance: {np.mean(chamfer_distances):.6f}")
        print(f"Median Chamfer distance: {np.median(chamfer_distances):.6f}")
        print(f"Std Chamfer distance: {np.std(chamfer_distances):.6f}")
        print(f"Min Chamfer distance: {np.min(chamfer_distances):.6f}")
        print(f"Max Chamfer distance: {np.max(chamfer_distances):.6f}")
        print("="*50)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Compare ground truth and NeRF-generated point clouds using Chamfer distance"
    )
    
    # Input/output arguments
    parser.add_argument("--gt_path", type=str, required=True,
                       help="Path to ground truth point cloud file or directory")
    parser.add_argument("--gen_path", type=str, required=True,
                       help="Path to generated point cloud file or directory")
    parser.add_argument("--output", type=str, default="chamfer_results.json",
                       help="Output file for results")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Output directory for visualizations")
    
    # Processing arguments
    parser.add_argument("--voxel_size", type=float, default=None,
                       help="Voxel size for downsampling (default: no downsampling)")
    parser.add_argument("--max_points", type=int, default=None,
                       help="Maximum points per point cloud (for memory management)")
    parser.add_argument("--no_auto_downsample", action="store_true",
                       help="Disable automatic downsampling of large point clouds")
    parser.add_argument("--chunk_size", type=int, default=1000,
                       help="Chunk size for memory-efficient Chamfer distance computation")
    parser.add_argument("--no_align", action="store_true",
                       help="Skip ICP alignment")
    parser.add_argument("--no_outlier_removal", action="store_true",
                       help="Skip outlier removal")
    
    # Visualization arguments
    parser.add_argument("--visualize", action="store_true",
                       help="Show visualization of point cloud comparisons")
    parser.add_argument("--batch", action="store_true",
                       help="Batch process directories instead of single files")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize comparator
    comparator = PointCloudComparator(log_level=args.log_level)
    
    try:
        if args.batch:
            # Batch processing
            results = comparator.batch_compare(
                gt_dir=args.gt_path,
                gen_dir=args.gen_path,
                output_file=args.output,
                align=not args.no_align,
                voxel_size=args.voxel_size,
                visualize=args.visualize,
                output_dir=args.output_dir,
                max_points_per_cloud=args.max_points,
                auto_downsample=not args.no_auto_downsample
            )
        else:
            # Single file comparison
            metrics = comparator.compare_single_pair(
                gt_path=args.gt_path,
                gen_path=args.gen_path,
                align=not args.no_align,
                voxel_size=args.voxel_size,
                visualize=args.visualize,
                output_dir=args.output_dir,
                max_points_per_cloud=args.max_points,
                auto_downsample=not args.no_auto_downsample
            )
            
            # Save single result
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            print(f"Chamfer distance: {metrics['chamfer_distance']:.6f}")
    
    except Exception as e:
        comparator.logger.error(f"Error during comparison: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()