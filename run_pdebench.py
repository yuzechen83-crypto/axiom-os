"""
Run PDEBench Experiments for Axiom-OS
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

This script provides a convenient entry point for:
1. Running cross-Reynolds generalization tests
2. Benchmarking against FNO/U-Net
3. Testing with synthetic data (no download required)

Usage:
    # Quick test with synthetic data
    python run_pdebench.py --mode test
    
    # Cross-Reynolds experiment (requires PDEBench data)
    python run_pdebench.py --mode cross_re --data_dir ./data/pdebench
    
    # Full benchmark
    python run_pdebench.py --mode benchmark --data_dir ./data/pdebench
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run_quick_test():
    """Run quick test with synthetic data (no download required)."""
    print("="*70)
    print("Axiom-OS PDEBench Integration - Quick Test")
    print("="*70)
    print("\nThis test uses synthetic data to verify the integration works.")
    print("For real experiments, download PDEBench data first.\n")
    
    from axiom_os.tests.test_pdebench_integration import run_tests
    run_tests()


def run_cross_reynolds_experiment(data_dir: str, epochs: int = 50):
    """Run cross-Reynolds generalization experiment."""
    print("="*70)
    print("Cross-Reynolds Generalization Experiment")
    print("="*70)
    print(f"\nData directory: {data_dir}")
    print("Train: Re=100 -> Test: Re=1000")
    print("This tests if Discovery Engine finds Reynolds-invariant physics.\n")
    
    try:
        from axiom_os.experiments.cross_reynolds_generalization import (
            CrossReynoldsConfig, CrossReynoldsExperiment
        )
        
        config = CrossReynoldsConfig(
            data_dir=data_dir,
            dataset_name="2D_NavierStokes",
            resolution=64,
            train_re=100,
            test_re=1000,
            epochs=epochs,
            batch_size=16,
            run_discovery=True,
            n_rollout_steps=5,
            output_dir="./outputs/cross_reynolds_experiment",
        )
        
        experiment = CrossReynoldsExperiment(config)
        results = experiment.run_full_experiment()
        
        # Print summary
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)
        
        print("\nFinal Test MSE (Re=1000):")
        for name in ["FNO", "U-Net", "RCLN"]:
            key = f"{name}_final_test_mse"
            if key in results:
                print(f"  {name:15s}: {results[key]:.6f}")
        
        if "discovered_formula" in results and results["discovered_formula"]:
            print(f"\nDiscovered Formula:")
            print(f"  {results['discovered_formula']}")
            
            print(f"\n🎯 Key Question: Does this formula capture Reynolds-invariant physics?")
            print(f"   If RCLN generalizes better than pure FNO/U-Net: YES!")
        
        return results
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease download PDEBench data first:")
        print("  1. Visit: https://github.com/pdebench/PDEBench")
        print("  2. Download 2D_NavierStokes_Re100_64x64.h5")
        print("  3. Download 2D_NavierStokes_Re1000_64x64.h5")
        print(f"  4. Place in: {data_dir}/")
        return None


def run_benchmark(data_dir: str, epochs: int = 50, re: int = 100):
    """Run full benchmark against FNO/U-Net."""
    print("="*70)
    print("PDEBench Benchmark - Axiom-OS vs FNO vs U-Net")
    print("="*70)
    print(f"\nData directory: {data_dir}")
    print(f"Reynolds number: {re}")
    print(f"Epochs: {epochs}\n")
    
    try:
        from axiom_os.experiments.pdebench_benchmark import (
            BenchmarkConfig, PDEBenchBenchmark
        )
        
        config = BenchmarkConfig(
            data_dir=data_dir,
            dataset_name="2D_NavierStokes",
            resolution=64,
            reynolds_number=re,
            epochs=epochs,
            batch_size=16,
            run_discovery=True,
            n_rollout_steps=5,
            output_dir="./outputs/pdebench_benchmark",
        )
        
        benchmark = PDEBenchBenchmark(config)
        results = benchmark.run_benchmark()
        
        return results
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease download PDEBench data first.")
        return None


def check_data_available(data_dir: str) -> bool:
    """Check if PDEBench data is available."""
    data_path = Path(data_dir)
    
    # Look for typical PDEBench files
    patterns = [
        "*NavierStokes*.h5",
        "*Burgers*.h5",
        "*Advection*.h5",
    ]
    
    for pattern in patterns:
        if list(data_path.glob(pattern)):
            return True
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Run PDEBench experiments for Axiom-OS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (no data download needed)
  python run_pdebench.py --mode test
  
  # Cross-Reynolds experiment
  python run_pdebench.py --mode cross_re --data_dir ./data/pdebench
  
  # Full benchmark
  python run_pdebench.py --mode benchmark --data_dir ./data/pdebench --epochs 100
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "cross_re", "benchmark"],
        default="test",
        help="Experiment mode (default: test)"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/pdebench",
        help="Directory containing PDEBench HDF5 files"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--re",
        type=int,
        default=100,
        help="Reynolds number for benchmark mode"
    )
    
    args = parser.parse_args()
    
    # Execute selected mode
    if args.mode == "test":
        run_quick_test()
    
    elif args.mode == "cross_re":
        if not check_data_available(args.data_dir):
            print("❌ PDEBench data not found!")
            print(f"\nExpected data in: {args.data_dir}")
            print("\nTo download:")
            print("  1. Visit: https://github.com/pdebench/PDEBench")
            print("  2. Download 2D_NavierStokes data")
            print("  3. Or run: python -m axiom_os.datasets.download_pdebench")
            print("\nTo run test without data:")
            print("  python run_pdebench.py --mode test")
            sys.exit(1)
        
        run_cross_reynolds_experiment(args.data_dir, args.epochs)
    
    elif args.mode == "benchmark":
        if not check_data_available(args.data_dir):
            print("❌ PDEBench data not found!")
            print(f"\nExpected data in: {args.data_dir}")
            sys.exit(1)
        
        run_benchmark(args.data_dir, args.epochs, args.re)


if __name__ == "__main__":
    main()
