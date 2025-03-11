#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run tests for CoordConv in diffusion models")
    
    # Main test command
    subparsers = parser.add_subparsers(dest="command", help="Test command to run")
    
    # All tests command
    all_parser = subparsers.add_parser("all", help="Run all CoordConv tests")
    all_parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to the dataset for training"
    )
    all_parser.add_argument(
        "--output_dir", type=str, default="outputs/coordconv_tests",
        help="Base directory to save test results"
    )
    all_parser.add_argument(
        "--model_id", type=str, default="google/ddpm-celebahq-256",
        help="HuggingFace model ID to use as base model"
    )
    all_parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for training"
    )
    all_parser.add_argument(
        "--num_epochs", type=int, default=5,
        help="Number of epochs to train"
    )
    all_parser.add_argument(
        "--with_r", action="store_true",
        help="Whether to include radius channel in CoordConv"
    )
    all_parser.add_argument(
        "--skip_training", action="store_true",
        help="Skip training and just run analysis on existing samples"
    )
    
    # Performance test command
    perf_parser = subparsers.add_parser("performance", help="Run performance comparison")
    perf_parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to the dataset for training"
    )
    perf_parser.add_argument(
        "--output_dir", type=str, default="outputs/coordconv_performance_test",
        help="Directory to save testing results"
    )
    perf_parser.add_argument(
        "--model_id", type=str, default="google/ddpm-celebahq-256",
        help="HuggingFace model ID to use as base model"
    )
    perf_parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for training"
    )
    perf_parser.add_argument(
        "--num_epochs", type=int, default=10,
        help="Number of epochs to train"
    )
    perf_parser.add_argument(
        "--with_r", action="store_true",
        help="Whether to include radius channel in CoordConv"
    )
    perf_parser.add_argument(
        "--skip_training", action="store_true",
        help="Skip training and just run analysis on existing samples"
    )
    
    # Spatial awareness test command
    spatial_parser = subparsers.add_parser("spatial", help="Run spatial awareness analysis")
    spatial_parser.add_argument(
        "--standard_samples_dir", type=str, required=True,
        help="Directory with standard UNet generated samples"
    )
    spatial_parser.add_argument(
        "--coordconv_samples_dir", type=str, required=True,
        help="Directory with CoordConv UNet generated samples"
    )
    spatial_parser.add_argument(
        "--output_dir", type=str, default="outputs/spatial_awareness_test",
        help="Directory to save test results"
    )
    spatial_parser.add_argument(
        "--with_r", action="store_true",
        help="Whether radius channel was used in CoordConv (for reporting)"
    )
    
    return parser.parse_args()

def run_performance_test(args):
    """Run performance comparison test."""
    cmd = [
        "python", "-m", "tests.test_coordconv_performance",
        "--dataset_path", args.dataset_path,
        "--output_dir", args.output_dir,
        "--model_id", args.model_id,
        "--batch_size", str(args.batch_size),
        "--num_epochs", str(args.num_epochs)
    ]
    
    if args.with_r:
        cmd.append("--with_r")
    
    if args.skip_training:
        cmd.append("--skip_training")
    
    # Run the command
    print(f"Running performance test with command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    return {
        "standard_samples_dir": os.path.join(args.output_dir, "standard_samples"),
        "coordconv_samples_dir": os.path.join(args.output_dir, "coordconv_samples")
    }

def run_spatial_awareness_test(args):
    """Run spatial awareness test."""
    cmd = [
        "python", "-m", "tests.test_coordconv_spatial_awareness",
        "--standard_samples_dir", args.standard_samples_dir,
        "--coordconv_samples_dir", args.coordconv_samples_dir,
        "--output_dir", args.output_dir
    ]
    
    if args.with_r:
        cmd.append("--with_r")
    
    # Run the command
    print(f"Running spatial awareness test with command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_all_tests(args):
    """Run all tests in sequence."""
    # Create output directories
    performance_output_dir = os.path.join(args.output_dir, "performance")
    spatial_output_dir = os.path.join(args.output_dir, "spatial")
    
    # Create argument objects for each test
    perf_args = argparse.Namespace(
        dataset_path=args.dataset_path,
        output_dir=performance_output_dir,
        model_id=args.model_id,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        with_r=args.with_r,
        skip_training=args.skip_training
    )
    
    # Run performance test first
    sample_dirs = run_performance_test(perf_args)
    
    # Create arguments for spatial awareness test
    spatial_args = argparse.Namespace(
        standard_samples_dir=sample_dirs["standard_samples_dir"],
        coordconv_samples_dir=sample_dirs["coordconv_samples_dir"],
        output_dir=spatial_output_dir,
        with_r=args.with_r
    )
    
    # Run spatial awareness test
    run_spatial_awareness_test(spatial_args)
    
    # Print summary
    print("\n==== CoordConv Test Suite Complete ====")
    print(f"Performance test results: {performance_output_dir}")
    print(f"Spatial awareness test results: {spatial_output_dir}")
    print("\nTo view the comparison images and reports, check the output directories.")

def main():
    """Run the selected test."""
    args = parse_args()
    
    # Make sure we're in the project root directory
    if not (Path.cwd() / "models" / "coord_conv").exists():
        print("Error: This script should be run from the project root directory with PYTHONPATH=.")
        print("Example: PYTHONPATH=. python -m tests.run_coordconv_tests all --dataset_path data/my_dataset")
        sys.exit(1)
    
    if args.command == "all":
        run_all_tests(args)
    elif args.command == "performance":
        run_performance_test(args)
    elif args.command == "spatial":
        run_spatial_awareness_test(args)
    else:
        print("Please specify a test command: all, performance, or spatial")
        sys.exit(1)


if __name__ == "__main__":
    main() 