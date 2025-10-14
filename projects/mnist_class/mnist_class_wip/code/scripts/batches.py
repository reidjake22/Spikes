import os
import glob
import argparse
from collections import defaultdict

def check_missing_batches(results_dir, checkpoint, noise, split):
    """
    Check which batches are missing and group them into contiguous runs
    """
    wsd_version = f"epoch_0_item_{checkpoint}"
    base_path = os.path.join(results_dir, wsd_version, f'noise_{noise}', split)
    
    # Expected range based on split
    if split == 'train':
        expected_range = range(0, 1200)
    elif split == 'test':
        expected_range = range(0, 200)
    else:
        raise ValueError("Split must be 'train' or 'test'")
    
    # Find existing batch files
    existing_batches = set()
    if os.path.exists(base_path):
        batch_dirs = glob.glob(os.path.join(base_path, "neuron_input_batch_*"))
        for batch_dir in batch_dirs:
            # Extract batch number from directory name
            batch_name = os.path.basename(batch_dir)
            if batch_name.startswith("neuron_input_batch_"):
                try:
                    batch_num = int(batch_name.split("_")[-1])
                    # Check if metrics.parquet exists
                    metrics_file = os.path.join(batch_dir, "metrics.parquet")
                    if os.path.exists(metrics_file):
                        existing_batches.add(batch_num)
                except ValueError:
                    continue
    
    # Find missing batches
    missing_batches = sorted(set(expected_range) - existing_batches)
    
    # Group missing batches into contiguous runs
    missing_runs = []
    if missing_batches:
        start = missing_batches[0]
        end = missing_batches[0]
        
        for i in range(1, len(missing_batches)):
            if missing_batches[i] == end + 1:
                end = missing_batches[i]
            else:
                missing_runs.append((start, end))
                start = missing_batches[i]
                end = missing_batches[i]
        
        # Don't forget the last run
        missing_runs.append((start, end))
    
    return existing_batches, missing_batches, missing_runs

def main():
    parser = argparse.ArgumentParser(description='Check missing batches')
    parser.add_argument('--checkpoint', type=int, required=True, choices=[0, 60000])
    parser.add_argument('--noise', type=int, required=True, choices=[0, 5])
    parser.add_argument('--split', type=str, required=True, choices=['test', 'train'])
    args = parser.parse_args()
    
    results_dir = "/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/results/trials"
    
    print(f"\n🔍 Checking checkpoint {args.checkpoint}, noise {args.noise}, split {args.split}")
    print("=" * 60)
    
    existing, missing, runs = check_missing_batches(results_dir, args.checkpoint, args.noise, args.split)
    
    total_expected = 1200 if args.split == 'train' else 200
    completed = len(existing)
    missing_count = len(missing)
    
    print(f"📊 Summary:")
    print(f"   Total expected: {total_expected}")
    print(f"   Completed: {completed} ({completed/total_expected*100:.1f}%)")
    print(f"   Missing: {missing_count} ({missing_count/total_expected*100:.1f}%)")
    
    if not missing:
        print("\n✅ All batches completed!")
        return
    
    print(f"\n📋 Missing batch runs:")
    for start, end in runs:
        if start == end:
            print(f"   Batch {start}")
        else:
            print(f"   Batches {start}-{end} ({end-start+1} batches)")
    
    print(f"\n🚀 Commands to run missing batches:")
    for start, end in runs:
        if start == end:
            cmd = f"python /home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/code/scripts/cpu_gen.py --split {args.split} --checkpoint {args.checkpoint} --noise {args.noise} --start_batch {start} --end_batch {start+1}"
        else:
            cmd = f"python /home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/code/scripts/cpu_gen.py --split {args.split} --checkpoint {args.checkpoint} --noise {args.noise} --start_batch {start} --end_batch {end+1}"
        print(f"   {cmd}")
    
    # Show first few missing batches for quick testing
    if missing:
        print(f"\n🧪 Quick test command (first missing batch):")
        first_missing = missing[0]
        cmd = f"python cpu_gen.py --split {args.split} --checkpoint {args.checkpoint} --noise {args.noise} --start_batch {first_missing} --end_batch {first_missing+1} --processes 1"
        print(f"   {cmd}")

def check_all_conditions():
    """Check all combinations and show overview"""
    results_dir = "/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/results/trials"
    
    conditions = [
        (0, 0, 'train'), (0, 0, 'test'),
        (0, 5, 'train'), (0, 5, 'test'),
        (60000, 0, 'train'), (60000, 0, 'test'),
        (60000, 5, 'train'), (60000, 5, 'test'),
    ]
    
    print("\n🌍 Overview of all conditions:")
    print("=" * 80)
    print(f"{'Checkpoint':<12} {'Noise':<8} {'Split':<8} {'Done':<8} {'Missing':<10} {'Progress':<10}")
    print("-" * 80)
    
    for checkpoint, noise, split in conditions:
        existing, missing, runs = check_missing_batches(results_dir, checkpoint, noise, split)
        total = 1200 if split == 'train' else 200
        progress = len(existing) / total * 100
        
        print(f"{checkpoint:<12} {noise:<8} {split:<8} {len(existing):<8} {len(missing):<10} {progress:.1f}%")
    
    print("\n📝 Detailed missing runs:")
    print("=" * 80)
    
    for checkpoint, noise, split in conditions:
        existing, missing, runs = check_missing_batches(results_dir, checkpoint, noise, split)
        if missing:
            print(f"\n🔸 Checkpoint {checkpoint}, Noise {noise}, Split {split}:")
            for start, end in runs:
                if start == end:
                    print(f"   Missing: {start}")
                else:
                    print(f"   Missing: {start}-{end} ({end-start+1} batches)")

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        # No arguments, show overview of all conditions
        check_all_conditions()
    else:
        # Arguments provided, check specific condition
        main()