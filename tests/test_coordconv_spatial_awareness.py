import os
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.feature import canny
import cv2
from tqdm import tqdm

# Import from our modules
from models.diffusion.core import DiffusionConfig, DiffusionModelLoader
from models.coord_conv.unet import convert_to_coordconv
from models.diffusion.coord import sample_from_coord_model
from training.trainer import DiffusionTrainer


class SpatialAwarenessTester:
    """Class to test spatial awareness improvements with CoordConv."""
    
    def __init__(
        self,
        standard_samples_dir: str,
        coordconv_samples_dir: str,
        output_dir: str = "outputs/spatial_awareness_test",
        with_r: bool = True,
        normalize_coords: bool = True
    ):
        """Initialize the spatial awareness tester.
        
        Args:
            standard_samples_dir: Directory with standard UNet generated samples
            coordconv_samples_dir: Directory with CoordConv UNet generated samples
            output_dir: Directory to save test results
            with_r: Whether radius channel was used in CoordConv (for reporting)
            normalize_coords: Whether coordinates were normalized (for reporting)
        """
        self.standard_samples_dir = standard_samples_dir
        self.coordconv_samples_dir = coordconv_samples_dir
        self.output_dir = output_dir
        self.with_r = with_r
        self.normalize_coords = normalize_coords
        
        os.makedirs(output_dir, exist_ok=True)
        
    def detect_edges(self, image_path):
        """Detect edges in an image using Canny edge detection.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Edge image and original image
        """
        # Load image
        img = np.array(Image.open(image_path).convert('RGB'))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = canny(gray, sigma=1.0)
        
        return edges, img
        
    def compute_edge_metrics(self):
        """Compute edge metrics to evaluate structure and spatial consistency.
        
        Returns:
            Dictionary of metrics
        """
        # Get samples
        standard_samples = sorted(Path(self.standard_samples_dir).glob("*.png"))
        coordconv_samples = sorted(Path(self.coordconv_samples_dir).glob("*.png"))
        
        num_samples = min(len(standard_samples), len(coordconv_samples))
        
        if num_samples == 0:
            raise ValueError("No samples found for comparison")
        
        # Initialize metrics
        metrics = {
            "edge_density_standard": [],
            "edge_density_coordconv": [],
            "edge_distance_standard": [],
            "edge_distance_coordconv": [],
            "structural_similarity": []
        }
        
        # Process each pair of images
        for i in range(num_samples):
            # Detect edges
            std_edges, std_img = self.detect_edges(standard_samples[i])
            coord_edges, coord_img = self.detect_edges(coordconv_samples[i])
            
            # Edge density (percentage of edge pixels)
            std_density = np.mean(std_edges)
            coord_density = np.mean(coord_edges)
            metrics["edge_density_standard"].append(std_density)
            metrics["edge_density_coordconv"].append(coord_density)
            
            # Compute average distance of edge pixels from the center
            h, w = std_edges.shape
            y_grid, x_grid = np.mgrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            distance_map = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
            
            # Mask with edges
            if np.any(std_edges):
                std_dist = np.mean(distance_map[std_edges])
                metrics["edge_distance_standard"].append(std_dist)
            else:
                metrics["edge_distance_standard"].append(np.nan)
                
            if np.any(coord_edges):
                coord_dist = np.mean(distance_map[coord_edges])
                metrics["edge_distance_coordconv"].append(coord_dist)
            else:
                metrics["edge_distance_coordconv"].append(np.nan)
            
            # Structural similarity
            sim = ssim(std_img, coord_img, multichannel=True)
            metrics["structural_similarity"].append(sim)
        
        # Calculate averages
        for key in metrics:
            metrics[f"{key}_mean"] = np.nanmean(metrics[key])
            metrics[f"{key}_std"] = np.nanstd(metrics[key])
        
        return metrics
        
    def analyze_quadrant_distribution(self):
        """Analyze object distribution across image quadrants.
        
        Returns:
            Dictionary of quadrant distribution metrics
        """
        # Get samples
        standard_samples = sorted(Path(self.standard_samples_dir).glob("*.png"))
        coordconv_samples = sorted(Path(self.coordconv_samples_dir).glob("*.png"))
        
        num_samples = min(len(standard_samples), len(coordconv_samples))
        
        if num_samples == 0:
            raise ValueError("No samples found for comparison")
        
        # Initialize quadrant metrics
        std_quadrant_density = np.zeros((num_samples, 2, 2))
        coord_quadrant_density = np.zeros((num_samples, 2, 2))
        
        # Process each pair of images
        for i in range(num_samples):
            # Detect edges
            std_edges, _ = self.detect_edges(standard_samples[i])
            coord_edges, _ = self.detect_edges(coordconv_samples[i])
            
            # Split into quadrants
            h, w = std_edges.shape
            h_mid, w_mid = h // 2, w // 2
            
            # Standard image quadrant edge density
            std_quadrant_density[i, 0, 0] = np.mean(std_edges[:h_mid, :w_mid])  # Top-left
            std_quadrant_density[i, 0, 1] = np.mean(std_edges[:h_mid, w_mid:])  # Top-right
            std_quadrant_density[i, 1, 0] = np.mean(std_edges[h_mid:, :w_mid])  # Bottom-left
            std_quadrant_density[i, 1, 1] = np.mean(std_edges[h_mid:, w_mid:])  # Bottom-right
            
            # CoordConv image quadrant edge density
            coord_quadrant_density[i, 0, 0] = np.mean(coord_edges[:h_mid, :w_mid])  # Top-left
            coord_quadrant_density[i, 0, 1] = np.mean(coord_edges[:h_mid, w_mid:])  # Top-right
            coord_quadrant_density[i, 1, 0] = np.mean(coord_edges[h_mid:, :w_mid])  # Bottom-left
            coord_quadrant_density[i, 1, 1] = np.mean(coord_edges[h_mid:, w_mid:])  # Bottom-right
        
        # Calculate quadrant distribution metrics
        quad_metrics = {
            "std_quadrant_means": np.mean(std_quadrant_density, axis=0),
            "coord_quadrant_means": np.mean(coord_quadrant_density, axis=0),
            "std_quadrant_std": np.std(std_quadrant_density, axis=0),
            "coord_quadrant_std": np.std(coord_quadrant_density, axis=0),
            
            # Variance across quadrants (higher means more uneven distribution)
            "std_quadrant_variance": np.mean([np.var(std_quadrant_density[i].flatten()) for i in range(num_samples)]),
            "coord_quadrant_variance": np.mean([np.var(coord_quadrant_density[i].flatten()) for i in range(num_samples)]),
        }
        
        return quad_metrics
        
    def generate_visualizations(self, metrics, quad_metrics):
        """Generate visualizations of the metrics.
        
        Args:
            metrics: Dictionary of metrics
            quad_metrics: Dictionary of quadrant metrics
        """
        # Create figure for edge metrics
        plt.figure(figsize=(12, 8))
        
        # Edge density comparison
        plt.subplot(2, 2, 1)
        plt.bar(['Standard', 'CoordConv'], 
                [metrics['edge_density_standard_mean'], metrics['edge_density_coordconv_mean']],
                yerr=[metrics['edge_density_standard_std'], metrics['edge_density_coordconv_std']])
        plt.title('Edge Density (higher is more detailed)')
        plt.ylabel('Mean Edge Density')
        
        # Edge distance comparison
        plt.subplot(2, 2, 2)
        plt.bar(['Standard', 'CoordConv'], 
                [metrics['edge_distance_standard_mean'], metrics['edge_distance_coordconv_mean']],
                yerr=[metrics['edge_distance_standard_std'], metrics['edge_distance_coordconv_std']])
        plt.title('Edge Distance from Center (lower is more centered)')
        plt.ylabel('Mean Distance')
        
        # Structural similarity
        plt.subplot(2, 2, 3)
        plt.bar(['Standard vs CoordConv'], [metrics['structural_similarity_mean']],
                yerr=[metrics['structural_similarity_std']])
        plt.title('Structural Similarity')
        plt.ylabel('SSIM Score')
        
        # Quadrant variance
        plt.subplot(2, 2, 4)
        plt.bar(['Standard', 'CoordConv'], 
                [quad_metrics['std_quadrant_variance'], quad_metrics['coord_quadrant_variance']])
        plt.title('Quadrant Variance (lower is more uniform)')
        plt.ylabel('Variance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics_comparison.png'))
        plt.close()
        
        # Create quadrant visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Standard model quadrant densities
        im0 = axes[0].imshow(quad_metrics['std_quadrant_means'], cmap='viridis', vmin=0, vmax=0.5)
        axes[0].set_title('Standard UNet Quadrant Edge Density')
        for i in range(2):
            for j in range(2):
                axes[0].text(j, i, f"{quad_metrics['std_quadrant_means'][i, j]:.3f}",
                           ha="center", va="center", color="white" if quad_metrics['std_quadrant_means'][i, j] > 0.25 else "black")
        
        # CoordConv model quadrant densities
        im1 = axes[1].imshow(quad_metrics['coord_quadrant_means'], cmap='viridis', vmin=0, vmax=0.5)
        axes[1].set_title('CoordConv UNet Quadrant Edge Density')
        for i in range(2):
            for j in range(2):
                axes[1].text(j, i, f"{quad_metrics['coord_quadrant_means'][i, j]:.3f}",
                           ha="center", va="center", color="white" if quad_metrics['coord_quadrant_means'][i, j] > 0.25 else "black")
        
        # Add a colorbar
        fig.colorbar(im1, ax=axes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'quadrant_analysis.png'))
        plt.close()
        
    def create_edge_overlay_visualization(self):
        """Create edge overlay visualization to compare structure detection."""
        # Get samples
        standard_samples = sorted(Path(self.standard_samples_dir).glob("*.png"))
        coordconv_samples = sorted(Path(self.coordconv_samples_dir).glob("*.png"))
        
        num_samples = min(len(standard_samples), len(coordconv_samples), 5)  # Limit to 5 samples
        
        if num_samples == 0:
            return
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Detect edges
            std_edges, std_img = self.detect_edges(standard_samples[i])
            coord_edges, coord_img = self.detect_edges(coordconv_samples[i])
            
            # Original images
            axes[i, 0].imshow(std_img)
            axes[i, 0].set_title(f"Standard UNet Sample {i+1}")
            axes[i, 0].axis('off')
            
            # Standard UNet edge overlay
            std_overlay = std_img.copy()
            std_overlay[std_edges, 0] = 255  # Highlight edges in red
            std_overlay[std_edges, 1] = 0
            std_overlay[std_edges, 2] = 0
            axes[i, 1].imshow(std_overlay)
            axes[i, 1].set_title(f"Standard UNet Edge Detection")
            axes[i, 1].axis('off')
            
            # CoordConv UNet edge overlay
            coord_overlay = coord_img.copy()
            coord_overlay[coord_edges, 0] = 255  # Highlight edges in red
            coord_overlay[coord_edges, 1] = 0
            coord_overlay[coord_edges, 2] = 0
            axes[i, 2].imshow(coord_overlay)
            axes[i, 2].set_title(f"CoordConv UNet Edge Detection")
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'edge_detection_comparison.png'))
        plt.close()
    
    def run_test(self):
        """Run spatial awareness tests and generate report."""
        print(f"Starting spatial awareness analysis...")
        print(f"Standard samples directory: {self.standard_samples_dir}")
        print(f"CoordConv samples directory: {self.coordconv_samples_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Compute edge metrics
        print("Computing edge metrics...")
        metrics = self.compute_edge_metrics()
        
        # Analyze quadrant distribution
        print("Analyzing quadrant distribution...")
        quad_metrics = self.analyze_quadrant_distribution()
        
        # Generate visualizations
        print("Generating visualizations...")
        self.generate_visualizations(metrics, quad_metrics)
        self.create_edge_overlay_visualization()
        
        # Generate report
        report_path = os.path.join(self.output_dir, "spatial_analysis_report.txt")
        with open(report_path, "w") as f:
            f.write("=== Spatial Awareness Analysis: Standard UNet vs CoordConv UNet ===\n\n")
            f.write(f"CoordConv Parameters: with_r={self.with_r}, normalize_coords={self.normalize_coords}\n\n")
            
            f.write("=== Edge Density ===\n")
            f.write(f"Standard UNet: {metrics['edge_density_standard_mean']:.4f} ± {metrics['edge_density_standard_std']:.4f}\n")
            f.write(f"CoordConv UNet: {metrics['edge_density_coordconv_mean']:.4f} ± {metrics['edge_density_coordconv_std']:.4f}\n")
            f.write(f"Difference: {metrics['edge_density_coordconv_mean'] - metrics['edge_density_standard_mean']:.4f}\n\n")
            
            f.write("=== Edge Distance from Center ===\n")
            f.write(f"Standard UNet: {metrics['edge_distance_standard_mean']:.4f} ± {metrics['edge_distance_standard_std']:.4f}\n")
            f.write(f"CoordConv UNet: {metrics['edge_distance_coordconv_mean']:.4f} ± {metrics['edge_distance_coordconv_std']:.4f}\n")
            f.write(f"Difference: {metrics['edge_distance_coordconv_mean'] - metrics['edge_distance_standard_mean']:.4f}\n\n")
            
            f.write("=== Structural Similarity ===\n")
            f.write(f"SSIM: {metrics['structural_similarity_mean']:.4f} ± {metrics['structural_similarity_std']:.4f}\n\n")
            
            f.write("=== Quadrant Analysis ===\n")
            f.write("Standard UNet quadrant edge density:\n")
            f.write(f"  Top-left: {quad_metrics['std_quadrant_means'][0, 0]:.4f}\n")
            f.write(f"  Top-right: {quad_metrics['std_quadrant_means'][0, 1]:.4f}\n")
            f.write(f"  Bottom-left: {quad_metrics['std_quadrant_means'][1, 0]:.4f}\n")
            f.write(f"  Bottom-right: {quad_metrics['std_quadrant_means'][1, 1]:.4f}\n")
            f.write(f"  Variance: {quad_metrics['std_quadrant_variance']:.4f}\n\n")
            
            f.write("CoordConv UNet quadrant edge density:\n")
            f.write(f"  Top-left: {quad_metrics['coord_quadrant_means'][0, 0]:.4f}\n")
            f.write(f"  Top-right: {quad_metrics['coord_quadrant_means'][0, 1]:.4f}\n")
            f.write(f"  Bottom-left: {quad_metrics['coord_quadrant_means'][1, 0]:.4f}\n")
            f.write(f"  Bottom-right: {quad_metrics['coord_quadrant_means'][1, 1]:.4f}\n")
            f.write(f"  Variance: {quad_metrics['coord_quadrant_variance']:.4f}\n\n")
            
            f.write("=== Conclusion ===\n")
            # Edge density comparison
            if metrics['edge_density_coordconv_mean'] > metrics['edge_density_standard_mean']:
                f.write("CoordConv produces images with more edge details, suggesting improved structural definition.\n")
            else:
                f.write("Standard UNet produces images with more edge details.\n")
            
            # Edge distance comparison
            if metrics['edge_distance_coordconv_mean'] < metrics['edge_distance_standard_mean']:
                f.write("CoordConv tends to place structures more centrally in the image.\n")
            else:
                f.write("Standard UNet tends to place structures more centrally in the image.\n")
            
            # Quadrant variance comparison
            if quad_metrics['coord_quadrant_variance'] < quad_metrics['std_quadrant_variance']:
                f.write("CoordConv distributes content more evenly across the image quadrants.\n")
            else:
                f.write("Standard UNet distributes content more evenly across the image quadrants.\n")
            
            # Overall assessment
            if (metrics['edge_density_coordconv_mean'] > metrics['edge_density_standard_mean'] and
                quad_metrics['coord_quadrant_variance'] < quad_metrics['std_quadrant_variance']):
                f.write("\nOverall, CoordConv appears to improve spatial awareness in generated images.\n")
            elif (metrics['edge_density_coordconv_mean'] < metrics['edge_density_standard_mean'] and
                  quad_metrics['coord_quadrant_variance'] > quad_metrics['std_quadrant_variance']):
                f.write("\nOverall, Standard UNet appears to have better spatial properties in generated images.\n")
            else:
                f.write("\nResults are mixed, with each model showing different spatial strengths.\n")
        
        print(f"Analysis complete! Report saved to {report_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test spatial awareness in CoordConv diffusion model")
    parser.add_argument(
        "--standard_samples_dir", type=str, required=True,
        help="Directory with standard UNet generated samples"
    )
    parser.add_argument(
        "--coordconv_samples_dir", type=str, required=True,
        help="Directory with CoordConv UNet generated samples"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/spatial_awareness_test",
        help="Directory to save test results"
    )
    parser.add_argument(
        "--with_r", action="store_true",
        help="Whether radius channel was used in CoordConv (for reporting)"
    )
    parser.add_argument(
        "--normalize_coords", action="store_true", default=True,
        help="Whether coordinates were normalized (for reporting)"
    )
    return parser.parse_args()


def main():
    """Run the spatial awareness test."""
    args = parse_args()
    
    tester = SpatialAwarenessTester(
        standard_samples_dir=args.standard_samples_dir,
        coordconv_samples_dir=args.coordconv_samples_dir,
        output_dir=args.output_dir,
        with_r=args.with_r,
        normalize_coords=args.normalize_coords
    )
    
    tester.run_test()


if __name__ == "__main__":
    main() 