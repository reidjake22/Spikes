#!/usr/bin/env python3

"""
Complete SNN Analysis Script
Runs classifier generalization, liquid generalization, and specific information analyses.
Saves all results and visualizations to organized output directories.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for headless operation
plt.switch_backend('Agg')

def setup_output_directories(base_output_dir="analysis_results"):
    """Create organized output directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Place output in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, f"{base_output_dir}_{timestamp}")
    
    subdirs = [
        "classifier_generalization",
        "liquid_generalization", 
        "specific_information",
        "plots",
        "data"
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    print(f"✅ Output directories created in: {output_dir}")
    return output_dir

def load_dataset(checkpoint, noise_level, split='train', feature_type='count') -> tuple:
    """
    Load dataset from specified checkpoint and noise level.
    """
    # Updated path - point directly to the correct location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Fix path to point to: /home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/results/trials
    base_path = os.path.join(script_dir, "..", "..", "mnist_class_wip", "results", "trials")
    base_path = os.path.abspath(base_path)
    
    data_path = f"{base_path}/epoch_0_item_{checkpoint}/noise_{noise_level}/{split}"
    
    if not os.path.exists(data_path):
        print(f"Expected path: {data_path}")
        print(f"Trying absolute path...")
        # Try the absolute path directly
        data_path = f"/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/results/trials/epoch_0_item_{checkpoint}/noise_{noise_level}/{split}"
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Path does not exist: {data_path}")
        else:
            print(f"Using absolute path: {data_path}")
    
    # Get all batch directories
    batch_dirs = glob(os.path.join(data_path, "neuron_input_batch_*"))
    batch_dirs = sorted(batch_dirs, key=lambda x: int(x.split('_')[-1]))
    
    print(f"  Loading {len(batch_dirs)} batches for checkpoint {checkpoint}, noise {noise_level}")
    
    # Load all parquet files and concatenate
    dfs = []
    for batch_dir in batch_dirs:
        parquet_file = os.path.join(batch_dir, "metrics.parquet")
        if os.path.exists(parquet_file):
            df = pd.read_parquet(parquet_file)
            batch_num = int(batch_dir.split('_')[-1])
            df['stimulus_id'] = df['stimulus_id'] + (batch_num * 50)
            dfs.append(df)
    
    if not dfs:
        raise FileNotFoundError(f"No parquet files found in {data_path}")
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Create pivot tables based on feature type
    if feature_type == 'count':
        X = full_df.pivot(index='stimulus_id', columns='neuron_id', values='count')
        X.columns = [f'neuron_{i}_count' for i in X.columns]
    elif feature_type == 'latency':
        X = full_df.pivot(index='stimulus_id', columns='neuron_id', values='latency')
        X.columns = [f'neuron_{i}_latency' for i in X.columns]
    elif feature_type == 'both':
        count_df = full_df.pivot(index='stimulus_id', columns='neuron_id', values='count')
        latency_df = full_df.pivot(index='stimulus_id', columns='neuron_id', values='latency')
        count_df.columns = [f'neuron_{i}_count' for i in count_df.columns]
        latency_df.columns = [f'neuron_{i}_latency' for i in latency_df.columns]
        X = pd.concat([count_df, latency_df], axis=1)
    else:
        raise ValueError("feature_type must be 'count', 'latency', or 'both'")
    
    y = full_df.groupby('stimulus_id')['label'].first().sort_index()
    X = X.sort_index()
    y = y.sort_index()
    
    return X, y

def run_classifier_generalization(checkpoints, noise_levels, output_dir):
    """
    Run classifier generalization analysis: train on clean, test on noisy.
    Saves results after each checkpoint.
    """
    print("\n" + "="*50)
    print("CLASSIFIER GENERALIZATION ANALYSIS")
    print("="*50)
    
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    results_df = pd.DataFrame(index=checkpoints, columns=noise_levels, dtype=float)
    detailed_results = []
    
    for checkpoint in checkpoints:    
        print(f"\n=== Processing checkpoint {checkpoint} ===")
        
        try:
            # Train classifier on clean data (noise=0)
            X, y = load_dataset(checkpoint, 0, split='train')
            
            print("  Training classifier on clean data...")
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
            scores = cross_val_score(clf, X, y, cv=5)
            
            # Fit on full clean dataset
            clf.fit(X, y)
            
            # Test on clean data first
            clean_score = clf.score(X, y)
            results_df.loc[checkpoint, 0] = clean_score
            
            detailed_results.append({
                'checkpoint': checkpoint,
                'train_noise': 0,
                'test_noise': 0,
                'cv_mean': scores.mean(),
                'cv_std': scores.std(),
                'test_accuracy': clean_score
            })
            
            print(f"  Clean data accuracy: {clean_score:.4f}")
            
            # Test on each noise level
            for noise_level in noise_levels[1:]:
                print(f"  Testing on noise level {noise_level}...")
                try:
                    X_noise, y_noise = load_dataset(checkpoint, noise_level, split='train')
                    score = clf.score(X_noise, y_noise)
                    results_df.loc[checkpoint, noise_level] = score
                    
                    detailed_results.append({
                        'checkpoint': checkpoint,
                        'train_noise': 0,
                        'test_noise': noise_level,
                        'cv_mean': scores.mean(),
                        'cv_std': scores.std(),
                        'test_accuracy': score
                    })
                    
                    print(f"    Noise {noise_level} accuracy: {score:.4f}")
                except Exception as e:
                    print(f"    Error loading noise {noise_level}: {e}")
                    results_df.loc[checkpoint, noise_level] = np.nan
            
            # 💾 SAVE AFTER EACH CHECKPOINT
            print(f"  💾 Saving results after checkpoint {checkpoint}...")
            results_df.to_csv(os.path.join(output_dir, "classifier_generalization", "generalization_matrix.csv"))
            pd.DataFrame(detailed_results).to_csv(os.path.join(output_dir, "classifier_generalization", "detailed_results.csv"), index=False)
            
            # Save checkpoint-specific results
            checkpoint_results = results_df.loc[[checkpoint]].dropna(axis=1)
            checkpoint_results.to_csv(os.path.join(output_dir, "classifier_generalization", f"checkpoint_{checkpoint}_results.csv"))
            print(f"    Saved checkpoint {checkpoint} specific results")
            
        except Exception as e:
            print(f"Error processing checkpoint {checkpoint}: {e}")
            for noise_level in noise_levels:
                results_df.loc[checkpoint, noise_level] = np.nan
            
            # Save even on error (partial results)
            results_df.to_csv(os.path.join(output_dir, "classifier_generalization", "generalization_matrix.csv"))
            pd.DataFrame(detailed_results).to_csv(os.path.join(output_dir, "classifier_generalization", "detailed_results.csv"), index=False)
    
    # Create final plot
    plt.figure(figsize=(10, 6))
    results_df_sorted = results_df.sort_index()
    results_df_sorted = results_df_sorted[sorted(results_df_sorted.columns, key=int)]
    
    for checkpoint in results_df_sorted.index:
        plt.plot(
            results_df_sorted.columns.astype(int),
            results_df_sorted.loc[checkpoint],
            marker='o',
            label=f'Checkpoint {checkpoint}',
            linewidth=2,
            markersize=6
        )
    
    plt.xlabel("Noise Level (σ)")
    plt.ylabel("Accuracy")
    plt.title("Classifier Generalization Across Noise Levels\n(Trained on Clean Data, Tested on Noisy Data)")
    plt.legend(title="SNN Checkpoint")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "classifier_generalization.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Classifier generalization analysis complete!")
    return results_df

def run_liquid_generalization(checkpoints, noise_levels, output_dir):
    """
    Run liquid generalization analysis: train and test on same noise level.
    Saves results after each checkpoint.
    """
    print("\n" + "="*50)
    print("LIQUID GENERALIZATION ANALYSIS")
    print("="*50)
    
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    
    rows = []
    
    for checkpoint in checkpoints:
        print(f"\nProcessing checkpoint {checkpoint}...")
        checkpoint_rows = []
        
        for noise_level in noise_levels:
            print(f"  Noise level {noise_level}...")
            
            try:
                X, y = load_dataset(checkpoint, noise_level, split='train')
                
                clf = LogisticRegression(max_iter=1000, random_state=42)
                scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
                
                clf.fit(X, y)
                train_score = clf.score(X, y)
                
                result = {
                    'checkpoint': checkpoint,
                    'noise_level': noise_level,
                    'cv_mean': scores.mean(),
                    'cv_std': scores.std(),
                    'train_score': train_score
                }
                
                rows.append(result)
                checkpoint_rows.append(result)
                print(f"    CV Score: {scores.mean():.4f} ± {scores.std():.4f}")
                
            except Exception as e:
                print(f"    Error: {e}")
                error_result = {
                    'checkpoint': checkpoint,
                    'noise_level': noise_level,
                    'cv_mean': np.nan,
                    'cv_std': np.nan,
                    'train_score': np.nan
                }
                rows.append(error_result)
                checkpoint_rows.append(error_result)
        
        # 💾 SAVE AFTER EACH CHECKPOINT
        print(f"  💾 Saving results after checkpoint {checkpoint}...")
        
        # Save cumulative results
        results_df = pd.DataFrame(rows)
        results_df.to_csv(os.path.join(output_dir, "liquid_generalization", "liquid_generalization_results.csv"), index=False)
        
        # Save checkpoint-specific results
        checkpoint_df = pd.DataFrame(checkpoint_rows)
        checkpoint_df.to_csv(os.path.join(output_dir, "liquid_generalization", f"checkpoint_{checkpoint}_results.csv"), index=False)
        print(f"    Saved checkpoint {checkpoint} specific results")
    
    results_df = pd.DataFrame(rows)
    
    # Create final plot
    results_df['noise_level'] = results_df['noise_level'].astype(int)
    results_df = results_df.sort_values(by=['checkpoint', 'noise_level'])
    
    plt.figure(figsize=(10, 6))
    
    for checkpoint in results_df['checkpoint'].unique():
        df_subset = results_df[results_df['checkpoint'] == checkpoint]
        plt.plot(
            df_subset['noise_level'],
            df_subset['cv_mean'],
            marker='o',
            label=f'Checkpoint {checkpoint}',
            linewidth=2,
            markersize=6
        )
        plt.fill_between(
            df_subset['noise_level'],
            df_subset['cv_mean'] - df_subset['cv_std'],
            df_subset['cv_mean'] + df_subset['cv_std'],
            alpha=0.2
        )
    
    plt.xlabel("Noise Level (σ)")
    plt.ylabel("Mean CV Accuracy")
    plt.title("Liquid Generalization vs Noise\n(Train and Test on Same Noise Level)")
    plt.legend(title="SNN Checkpoint")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "liquid_generalization.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Liquid generalization analysis complete!")
    return results_df

def compute_specific_information_wide(X: pd.DataFrame, y: pd.Series, n_bins: int = 3) -> pd.DataFrame:
    """
    Compute specific information for each neuron and each class label.
    """
    neurons = X.columns
    classes = np.unique(y)
    N = len(y)
    
    spec_info = pd.DataFrame(index=neurons, columns=classes, dtype=float)
    
    for neuron in neurons:
        counts = X[neuron].values
        bins = np.linspace(counts.min(), counts.max(), n_bins + 1)
        bin_idx = np.digitize(counts, bins[1:-1])
        
        p_r = pd.Series(bin_idx).value_counts().sort_index() / N
        
        df_nr = pd.DataFrame({'class': y.values, 'bin': bin_idx})
        p_r_c = df_nr.groupby(['class', 'bin']).size().div(df_nr.groupby('class').size()).unstack(fill_value=0)
        
        for c in classes:
            info = 0.0
            for b in range(n_bins):
                pr_c = p_r_c.loc[c, b] if b in p_r_c.columns else 0.0
                pr = p_r.get(b, 0.0)
                if pr_c > 0 and pr > 0:
                    info += pr_c * np.log2(pr_c / pr)
            spec_info.loc[neuron, c] = info
    
    return spec_info

def plot_spec_info(spec_info: pd.Series, checkpoint: int, noise_level: int, output_dir: str):
    """
    Plot specific information with log scale rank on x-axis.
    """
    sorted_info = spec_info.sort_values(ascending=False)
    ranks = np.log10(np.arange(1, len(sorted_info) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, sorted_info, marker='o', markersize=3, alpha=0.7)
    plt.title(f"Max Specific Information\n(Checkpoint: {checkpoint}, Noise Level: {noise_level})")
    plt.xlabel("Log10(Rank)")
    plt.ylabel("Informativity (bits)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"specific_info_checkpoint_{checkpoint}_noise_{noise_level}.png"
    plt.savefig(os.path.join(output_dir, "plots", filename), dpi=300, bbox_inches='tight')
    plt.close()

def run_specific_information_analysis(checkpoints, noise_levels, output_dir):
    """
    Run specific information analysis for all conditions.
    Saves results after each checkpoint.
    """
    print("\n" + "="*50)
    print("SPECIFIC INFORMATION ANALYSIS")
    print("="*50)
    
    summary_results = []
    
    for checkpoint in checkpoints:
        print(f"\nProcessing checkpoint {checkpoint}...")
        checkpoint_summary = []
        
        for noise_level in noise_levels:
            print(f"  Computing specific information for noise level {noise_level}...")
            
            try:
                X, y = load_dataset(checkpoint, noise_level, split='train')
                
                # Compute specific information
                spec_info = compute_specific_information_wide(X, y, n_bins=3)
                max_spec_info_per_neuron = spec_info.max(axis=1)
                
                # Plot the results
                plot_spec_info(max_spec_info_per_neuron, checkpoint, noise_level, output_dir)
                
                # Save detailed data for this specific condition
                spec_info.to_csv(os.path.join(output_dir, "specific_information", 
                                            f"detailed_spec_info_checkpoint_{checkpoint}_noise_{noise_level}.csv"))
                max_spec_info_per_neuron.to_csv(os.path.join(output_dir, "specific_information",
                                                            f"max_spec_info_checkpoint_{checkpoint}_noise_{noise_level}.csv"))
                
                # Collect summary statistics
                result = {
                    'checkpoint': checkpoint,
                    'noise_level': noise_level,
                    'mean_max_spec_info': max_spec_info_per_neuron.mean(),
                    'std_max_spec_info': max_spec_info_per_neuron.std(),
                    'median_max_spec_info': max_spec_info_per_neuron.median(),
                    'max_spec_info': max_spec_info_per_neuron.max(),
                    'n_neurons': len(max_spec_info_per_neuron),
                    'n_informative_neurons': (max_spec_info_per_neuron > 0.01).sum()
                }
                
                summary_results.append(result)
                checkpoint_summary.append(result)
                
                print(f"    Mean max spec info: {max_spec_info_per_neuron.mean():.4f}")
                print(f"    Informative neurons (>0.01 bits): {(max_spec_info_per_neuron > 0.01).sum()}")
                
            except Exception as e:
                print(f"    Error: {e}")
                error_result = {
                    'checkpoint': checkpoint,
                    'noise_level': noise_level,
                    'mean_max_spec_info': np.nan,
                    'std_max_spec_info': np.nan,
                    'median_max_spec_info': np.nan,
                    'max_spec_info': np.nan,
                    'n_neurons': np.nan,
                    'n_informative_neurons': np.nan
                }
                summary_results.append(error_result)
                checkpoint_summary.append(error_result)
        
        # 💾 SAVE AFTER EACH CHECKPOINT
        print(f"  💾 Saving summary results after checkpoint {checkpoint}...")
        
        # Save cumulative summary results
        summary_df = pd.DataFrame(summary_results)
        summary_df.to_csv(os.path.join(output_dir, "specific_information", "summary_results.csv"), index=False)
        
        # Save checkpoint-specific summary
        checkpoint_summary_df = pd.DataFrame(checkpoint_summary)
        checkpoint_summary_df.to_csv(os.path.join(output_dir, "specific_information", f"checkpoint_{checkpoint}_summary.csv"), index=False)
        print(f"    Saved checkpoint {checkpoint} specific summary")
    
    # Create final summary plot
    summary_df = pd.DataFrame(summary_results)
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Mean specific information vs noise
    plt.subplot(2, 2, 1)
    for checkpoint in checkpoints:
        df_subset = summary_df[summary_df['checkpoint'] == checkpoint]
        plt.plot(df_subset['noise_level'], df_subset['mean_max_spec_info'], 
                marker='o', label=f'Checkpoint {checkpoint}')
    plt.xlabel("Noise Level")
    plt.ylabel("Mean Max Spec Info (bits)")
    plt.title("Mean Specific Information vs Noise")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Number of informative neurons vs noise
    plt.subplot(2, 2, 2)
    for checkpoint in checkpoints:
        df_subset = summary_df[summary_df['checkpoint'] == checkpoint]
        plt.plot(df_subset['noise_level'], df_subset['n_informative_neurons'], 
                marker='o', label=f'Checkpoint {checkpoint}')
    plt.xlabel("Noise Level")
    plt.ylabel("Number of Informative Neurons")
    plt.title("Informative Neurons vs Noise")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Max specific information vs noise
    plt.subplot(2, 2, 3)
    for checkpoint in checkpoints:
        df_subset = summary_df[summary_df['checkpoint'] == checkpoint]
        plt.plot(df_subset['noise_level'], df_subset['max_spec_info'], 
                marker='o', label=f'Checkpoint {checkpoint}')
    plt.xlabel("Noise Level")
    plt.ylabel("Max Spec Info (bits)")
    plt.title("Peak Specific Information vs Noise")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Standard deviation vs noise
    plt.subplot(2, 2, 4)
    for checkpoint in checkpoints:
        df_subset = summary_df[summary_df['checkpoint'] == checkpoint]
        plt.plot(df_subset['noise_level'], df_subset['std_max_spec_info'], 
                marker='o', label=f'Checkpoint {checkpoint}')
    plt.xlabel("Noise Level")
    plt.ylabel("Std Max Spec Info (bits)")
    plt.title("Variability in Specific Information")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "specific_information_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Specific information analysis complete!")
    return summary_df

def main():
    """
    Main function to run complete analysis pipeline.
    """
    print("🚀 Starting Complete SNN Analysis Pipeline")
    print("=" * 60)
    
    # Configuration
    checkpoints = [0, 20000, 40000, 60000]
    noise_levels = [0, 5, 15, 30, 50]
    
    # Setup output directories
    output_dir = setup_output_directories()
    
    # Save configuration
    config = {
        'checkpoints': checkpoints,
        'noise_levels': noise_levels,
        'timestamp': datetime.now().isoformat(),
        'script_version': '1.1_per_checkpoint_saving'
    }
    pd.Series(config).to_csv(os.path.join(output_dir, "analysis_config.csv"))
    
    try:
        # Run all analyses
        classifier_results = run_classifier_generalization(checkpoints, noise_levels, output_dir)
        liquid_results = run_liquid_generalization(checkpoints, noise_levels, output_dir)
        spec_info_results = run_specific_information_analysis(checkpoints, noise_levels, output_dir)
        
        print("\n" + "="*60)
        print("🎉 ANALYSIS COMPLETE!")
        print("="*60)
        print(f"📁 All results saved to: {output_dir}")
        print("\n📊 Summary:")
        print(f"   • Classifier generalization: {len(checkpoints)} checkpoints × {len(noise_levels)} noise levels")
        print(f"   • Liquid generalization: {len(liquid_results)} total conditions")
        print(f"   • Specific information: {len(spec_info_results)} total conditions")
        print(f"   • Total plots generated: {len(checkpoints) * len(noise_levels) + 3} files")
        print(f"   • Individual checkpoint files: {len(checkpoints) * 3} files")
            
        # Create final summary report
        with open(os.path.join(output_dir, "ANALYSIS_SUMMARY.txt"), 'w') as f:
            f.write("SNN Analysis Summary Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output directory: {output_dir}\n\n")
            f.write("Main result files:\n")
            f.write("- classifier_generalization/generalization_matrix.csv\n")
            f.write("- liquid_generalization/liquid_generalization_results.csv\n")
            f.write("- specific_information/summary_results.csv\n")
            f.write("\nPer-checkpoint files:\n")
            f.write("- classifier_generalization/checkpoint_*_results.csv\n")
            f.write("- liquid_generalization/checkpoint_*_results.csv\n")
            f.write("- specific_information/checkpoint_*_summary.csv\n")
            f.write("\nPlots:\n")
            f.write("- plots/classifier_generalization.png\n")
            f.write("- plots/liquid_generalization.png\n")
            f.write("- plots/specific_information_summary.png\n")
            f.write(f"- plots/specific_info_checkpoint_*_noise_*.png ({len(checkpoints) * len(noise_levels)} files)\n")
        
        print(f"📋 Summary report: {os.path.join(output_dir, 'ANALYSIS_SUMMARY.txt')}")
        
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")

        # Save error log
        with open(os.path.join(output_dir, "error_log.txt"), 'w') as f:
            f.write(f"Analysis failed at: {datetime.now().isoformat()}\n")
            f.write(f"Error: {str(e)}\n\n")


def main2():
    """
    Focused analysis of increasing checkpoint values at noise=0.
    Compares classification accuracy and specific information across checkpoints.
    Tests on test split with noise=0 only.
    """
    print("🚀 Starting Checkpoint Evolution Analysis")
    print("=" * 60)
    
    # Configuration - multiple checkpoints, but only noise=0
    checkpoints = [0, 500, 1000, 1500, 2500, 5000, 7500, 10000, 20000]
    noise_level = 0  # Only using noise=0
    
    # Setup output directories
    output_dir = setup_output_directories(base_output_dir="checkpoint_analysis")
    
    # Save configuration
    config = {
        'checkpoints': checkpoints,
        'noise_level': noise_level,
        'split': 'test',
        'timestamp': datetime.now().isoformat(),
        'script_version': '1.2_checkpoint_evolution'
    }
    pd.Series(config).to_csv(os.path.join(output_dir, "analysis_config.csv"))
    
    # Results containers
    accuracy_results = []
    spec_info_summary = []
    
    print("\n" + "="*50)
    print("CHECKPOINT EVOLUTION ANALYSIS")
    print("="*50)
    
    try:
        for checkpoint in checkpoints:
            print(f"\n=== Processing checkpoint {checkpoint} ===")
            
            try:
                # Load test data for this checkpoint at noise=0
                X_test, y_test = load_dataset(checkpoint, noise_level, split='test')
                
                # Train classifier on test data
                print(f"  Training classifier on test data (noise=0)...")
                
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import cross_val_score
                
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
                scores = cross_val_score(clf, X_test, y_test, cv=5)
                
                # Save accuracy results
                accuracy_result = {
                    'checkpoint': checkpoint,
                    'cv_mean': scores.mean(),
                    'cv_std': scores.std(),
                }
                accuracy_results.append(accuracy_result)
                
                print(f"  Test accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
                
                # Compute specific information
                print(f"  Computing specific information...")
                spec_info = compute_specific_information_wide(X_test, y_test, n_bins=3)
                max_spec_info_per_neuron = spec_info.max(axis=1)
                
                # Plot the results
                plot_spec_info(max_spec_info_per_neuron, checkpoint, noise_level, output_dir)
                
                # Save detailed specific information data
                spec_info.to_csv(os.path.join(output_dir, "specific_information", 
                                              f"detailed_spec_info_checkpoint_{checkpoint}.csv"))
                max_spec_info_per_neuron.to_csv(os.path.join(output_dir, "specific_information",
                                                             f"max_spec_info_checkpoint_{checkpoint}.csv"))
                
                # Collect summary statistics
                result = {
                    'checkpoint': checkpoint,
                    'mean_max_spec_info': max_spec_info_per_neuron.mean(),
                    'std_max_spec_info': max_spec_info_per_neuron.std(),
                    'median_max_spec_info': max_spec_info_per_neuron.median(),
                    'max_spec_info': max_spec_info_per_neuron.max(),
                    'n_neurons': len(max_spec_info_per_neuron),
                    'n_informative_neurons': (max_spec_info_per_neuron > 0.01).sum()
                }
                spec_info_summary.append(result)
                
                print(f"    Mean max spec info: {max_spec_info_per_neuron.mean():.4f}")
                print(f"    Informative neurons (>0.01 bits): {(max_spec_info_per_neuron > 0.01).sum()}")
                
                # 💾 SAVE AFTER EACH CHECKPOINT
                print(f"  💾 Saving results after checkpoint {checkpoint}...")
                
                # Save accuracy results
                accuracy_df = pd.DataFrame(accuracy_results)
                accuracy_df.to_csv(os.path.join(output_dir, "classifier_generalization", "accuracy_results.csv"), index=False)
                
                # Save specific information summary
                spec_info_df = pd.DataFrame(spec_info_summary)
                spec_info_df.to_csv(os.path.join(output_dir, "specific_information", "spec_info_summary.csv"), index=False)
                
            except Exception as e:
                print(f"Error processing checkpoint {checkpoint}: {e}")
        
        # Create final plots
        
        # 1. Accuracy vs checkpoint plot
        plt.figure(figsize=(10, 6))
        accuracy_df = pd.DataFrame(accuracy_results)
        
        plt.errorbar(
            accuracy_df['checkpoint'],
            accuracy_df['cv_mean'],
            yerr=accuracy_df['cv_std'],
            marker='o',
            markersize=8,
            capsize=6,
            linewidth=2,
            elinewidth=1.5
        )
        
        plt.xlabel("Checkpoint")
        plt.ylabel("Cross-Validation Accuracy")
        plt.title("Classification Accuracy vs Training Checkpoint\n(Test Split, Noise=0)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plots", "accuracy_vs_checkpoint.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Specific information vs checkpoint plot
        plt.figure(figsize=(12, 9))
        spec_info_df = pd.DataFrame(spec_info_summary)
        
        plt.subplot(2, 2, 1)
        plt.plot(spec_info_df['checkpoint'], spec_info_df['mean_max_spec_info'], 'o-', linewidth=2)
        plt.xlabel("Checkpoint")
        plt.ylabel("Mean Max Spec Info (bits)")
        plt.title("Mean Specific Information vs Checkpoint")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(spec_info_df['checkpoint'], spec_info_df['n_informative_neurons'], 'o-', linewidth=2)
        plt.xlabel("Checkpoint")
        plt.ylabel("Number of Neurons")
        plt.title("Number of Informative Neurons vs Checkpoint")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(spec_info_df['checkpoint'], spec_info_df['max_spec_info'], 'o-', linewidth=2)
        plt.xlabel("Checkpoint")
        plt.ylabel("Max Spec Info (bits)")
        plt.title("Peak Specific Information vs Checkpoint")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        # Add correlation between accuracy and mean specific information
        merged_df = pd.merge(accuracy_df, spec_info_df, on='checkpoint')
        plt.scatter(merged_df['cv_mean'], merged_df['mean_max_spec_info'], s=80)
        
        # Add checkpoint labels to each point
        for i, ckpt in enumerate(merged_df['checkpoint']):
            plt.annotate(f"{int(ckpt)}", 
                        (merged_df['cv_mean'].iloc[i], merged_df['mean_max_spec_info'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points')
            
        plt.xlabel("Classification Accuracy")
        plt.ylabel("Mean Max Spec Info (bits)")
        plt.title("Information vs Accuracy")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plots", "spec_info_vs_checkpoint.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Combined plot showing both metrics
        plt.figure(figsize=(12, 6))
        
        # Setup primary y-axis for accuracy
        ax1 = plt.gca()
        ax1.set_xlabel("Checkpoint")
        ax1.set_ylabel("Classification Accuracy", color='blue')
        ax1.plot(accuracy_df['checkpoint'], accuracy_df['cv_mean'], 'o-', color='blue', linewidth=2, label="Accuracy")
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim([0, 1])
        
        # Setup secondary y-axis for specific information
        ax2 = ax1.twinx()
        ax2.set_ylabel("Mean Max Specific Information (bits)", color='red')
        ax2.plot(spec_info_df['checkpoint'], spec_info_df['mean_max_spec_info'], 'o-', color='red', linewidth=2, label="Specific Info")
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.title("Classification Accuracy and Specific Information vs Checkpoint\n(Test Split, Noise=0)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plots", "combined_metrics.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n" + "="*60)
        print("🎉 CHECKPOINT EVOLUTION ANALYSIS COMPLETE!")
        print("="*60)
        print(f"📁 All results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        
        # Save error log
        with open(os.path.join(output_dir, "error_log.txt"), 'w') as f:
            f.write(f"Analysis failed at: {datetime.now().isoformat()}\n")
            f.write(f"Error: {str(e)}\n\n")

if __name__ == "__main__":
    # Uncomment one of these to run the desired analysis
    # main()  # Run full analysis across checkpoints and noise levels
    main2()  # Run focused analysis of checkpoint evolution at noise=0