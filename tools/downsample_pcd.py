import open3d as o3d

def denoise_point_cloud(pcd_file_path, output_file_path, voxel_size=0.05):

    pcd = o3d.io.read_point_cloud(pcd_file_path)

    if not pcd.has_points():
        raise ValueError("The provided PCD file is empty or invalid.")
    
    output = pcd.voxel_down_sample(voxel_size=voxel_size)

    o3d.io.write_point_cloud(output_file_path, output)
    
if __name__ == "__main__":
    
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Denoise a point cloud using voxel grid filtering.')
    parser.add_argument('input_pcd_file', type=str, help='Path to the input PCD file.')
    parser.add_argument('output_pcd_file', type=str, help='Path to save the denoised PCD file.')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='Size of the voxel grid for downsampling (default: 0.05).')
    args = parser.parse_args()

    
    denoise_point_cloud(args.input_pcd_file, args.output_pcd_file, args.voxel_size)

