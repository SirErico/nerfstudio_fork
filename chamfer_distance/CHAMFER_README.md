# Chamfer Distance Comparison Tool

This Python script compares ground truth laser scans from the YCB dataset with NeRF-generated point clouds using Chamfer distance metrics. It includes memory-efficient processing for large point clouds and comprehensive preprocessing options.

## Features

- **Memory-Efficient Processing**: Handles very large point clouds (hundreds of thousands of points) using chunked computation
- **Multiple File Format Support**: PLY, PCD, XYZ, OBJ, and NumPy arrays
- **ICP Alignment**: Automatic point cloud registration for better comparison
- **Preprocessing Options**: Voxel downsampling, outlier removal, and automatic size limiting
- **Batch Processing**: Process entire directories of point cloud pairs
- **Comprehensive Metrics**: Multiple distance metrics including bidirectional Chamfer distance, RMSE, and max distances

## Installation

Ensure you have the required dependencies:

```bash
pip install open3d numpy scipy tqdm matplotlib
```

## Usage

### Single File Comparison

Compare two point cloud files:

```bash
python3 chamfer.py --gt_path ground_truth.ply --gen_path generated.ply
```

### Batch Processing

Compare all point clouds in two directories:

```bash
python3 chamfer.py --gt_path ./ground_truth_dir/ --gen_path ./generated_dir/ --batch
```

### Memory Management for Large Point Clouds

For very large point clouds (like the ones causing the 1.77 TiB memory error):

```bash
# Automatically limit each point cloud to 50k points
python3 chamfer.py --gt_path large_gt.ply --gen_path large_gen.ply --max_points 50000

# Use voxel downsampling
python3 chamfer.py --gt_path large_gt.ply --gen_path large_gen.ply --voxel_size 0.01

# Disable automatic downsampling (not recommended for very large clouds)
python3 chamfer.py --gt_path gt.ply --gen_path gen.ply --no_auto_downsample
```

## Command Line Options

### Required Arguments
- `--gt_path`: Path to ground truth point cloud file or directory
- `--gen_path`: Path to generated point cloud file or directory

### Processing Options
- `--voxel_size FLOAT`: Voxel size for downsampling (default: no downsampling)
- `--max_points INT`: Maximum points per point cloud (for memory management)
- `--no_auto_downsample`: Disable automatic downsampling of large point clouds
- `--chunk_size INT`: Chunk size for memory-efficient computation (default: 1000)
- `--no_align`: Skip ICP alignment
- `--no_outlier_removal`: Skip outlier removal

### Output Options
- `--output FILE`: Output JSON file for results (default: chamfer_results.json)
- `--output_dir DIR`: Output directory for visualizations (default: output)
- `--visualize`: Show visualization of point cloud comparisons
- `--batch`: Batch process directories instead of single files

### Other Options
- `--log_level LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Output Format

The script outputs a JSON file with detailed metrics:

```json
{
  "object_name": {
    "chamfer_distance": 0.045123,
    "chamfer_1_to_2": 0.042156,
    "chamfer_2_to_1": 0.048091,
    "max_distance_1_to_2": 0.234567,
    "max_distance_2_to_1": 0.198765,
    "rmse_1_to_2": 0.051234,
    "rmse_2_to_1": 0.056789,
    "gt_points": 45230,
    "gen_points": 38901,
    "gt_path": "/path/to/gt.ply",
    "gen_path": "/path/to/gen.ply"
  }
}
```

## Memory Management

The script automatically handles memory-intensive computations:

1. **Automatic Detection**: Detects large point clouds and applies memory-safe limits
2. **Chunked Processing**: Breaks distance computations into manageable chunks
3. **Smart Downsampling**: Applies voxel downsampling or random sampling as needed
4. **Progress Tracking**: Shows progress bars for long-running computations

### Memory Usage Guidelines

- **Small clouds** (<10k points each): No special handling needed
- **Medium clouds** (10k-100k points): Automatic downsampling applied
- **Large clouds** (>100k points): Aggressive downsampling recommended
- **Very large clouds** (>500k points): Manual `--max_points` setting recommended

## File Format Support

### Supported Input Formats
- **PLY** (`.ply`): Standard point cloud format
- **PCD** (`.pcd`): Point Cloud Data format
- **XYZ** (`.xyz`, `.xyzn`, `.xyzrgb`): ASCII point cloud formats
- **OBJ** (`.obj`): 3D mesh (automatically sampled to points)
- **NumPy** (`.npy`): NumPy arrays (Nx3 format expected)

### File Naming Conventions for Batch Processing

For batch processing, the script looks for matching files using these patterns:
- `object_name.ply` → `object_name.ply`
- `object_name.ply` → `object_name_generated.ply`
- `object_name.ply` → `object_name_nerf.ply`

## Examples

### Example 1: Basic YCB Object Comparison
```bash
python3 chamfer.py \
    --gt_path ./ycb_scans/003_cracker_box.ply \
    --gen_path ./nerf_outputs/003_cracker_box_generated.ply
```

### Example 2: Batch Processing with Memory Management
```bash
python3 chamfer.py \
    --gt_path ./ycb_ground_truth/ \
    --gen_path ./nerf_generated/ \
    --batch \
    --max_points 25000 \
    --voxel_size 0.005 \
    --output ycb_comparison_results.json
```

### Example 3: High-Memory System (No Limits)
```bash
python3 chamfer.py \
    --gt_path large_scan.ply \
    --gen_path large_nerf.ply \
    --no_auto_downsample \
    --chunk_size 5000
```

## Troubleshooting

### Memory Errors
- Use `--max_points 10000` to limit point cloud sizes
- Increase `--voxel_size` for more aggressive downsampling  
- Reduce `--chunk_size` if still getting memory errors

### Alignment Issues
- Use `--no_align` if point clouds are already aligned
- Check that point clouds are in the same coordinate system
- Verify point cloud orientations are similar

### File Format Issues
- Ensure file extensions match the actual format
- For NumPy files, ensure they contain Nx3 arrays
- Check that OBJ files contain valid meshes

## Performance Tips

1. **Pre-process large point clouds** with external tools before comparison
2. **Use voxel downsampling** instead of random sampling for better preservation of geometry
3. **Process in batches** rather than individual files for better efficiency
4. **Monitor memory usage** with system tools when processing very large datasets

## Testing

Run the included test script to verify functionality:

```bash
python3 test_chamfer.py
```

This creates synthetic point clouds and demonstrates the memory-efficient processing.
