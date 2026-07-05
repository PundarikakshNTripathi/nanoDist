import os
import sys
sys.path.insert(0, "src")
import matplotlib.pyplot as plt
import numpy as np
import re
from distributed_trainer.core.memory_manager import compare_memory_with_and_without_optimizations

def generate_memory_plot():
    # Dummy network
    x = np.random.randn(128, 1024).astype(np.float32)
    params = {
        'W1': np.random.randn(1024, 4096).astype(np.float32),
        'b1': np.zeros(4096).astype(np.float32),
        'W2': np.random.randn(4096, 1024).astype(np.float32),
        'b2': np.zeros(1024).astype(np.float32)
    }
    
    results = compare_memory_with_and_without_optimizations(x, params, num_workers=4)
    
    base = results['breakdown_baseline']
    opt = results['breakdown_optimized']
    
    labels = ['Params', 'Optimizer', 'Activations']
    base_vals = [base['params'] / 1e6, base['optimizer'] / 1e6, base['activations'] / 1e6] # MB
    opt_vals = [opt['params'] / 1e6, opt['optimizer'] / 1e6, opt['activations'] / 1e6]
    
    x_axis = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x_axis - width/2, base_vals, width, label='Baseline (FP32, No ZeRO)')
    ax.bar(x_axis + width/2, opt_vals, width, label='Optimized (FP16, ZeRO, Checkpointing)')
    
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Per-GPU Memory Footprint Comparison')
    ax.set_xticks(x_axis)
    ax.set_xticklabels(labels)
    ax.legend()
    
    os.makedirs('benchmarks/images', exist_ok=True)
    plt.savefig('benchmarks/images/memory_comparison.png')
    print("Memory benchmark plot saved to benchmarks/images/memory_comparison.png")
    
    update_readme_metrics(base_vals, opt_vals, results['savings_ratio'])

def update_readme_metrics(base_vals, opt_vals, savings_ratio):
    """Dynamically updates the README.md with the latest benchmark metrics."""
    readme_path = "README.md"
    if not os.path.exists(readme_path):
        return
        
    with open(readme_path, "r") as f:
        content = f.read()
        
    markdown_table = f"""<!-- BENCHMARK_METRICS_START -->
| Component | Baseline (MB) | Optimized (MB) |
|-----------|---------------|----------------|
| Parameters | {base_vals[0]:.2f} | {opt_vals[0]:.2f} |
| Optimizer State | {base_vals[1]:.2f} | {opt_vals[1]:.2f} |
| Activations | {base_vals[2]:.2f} | {opt_vals[2]:.2f} |
| **Total per GPU** | **{sum(base_vals):.2f}** | **{sum(opt_vals):.2f}** |

**Total Memory Reduction Ratio:** {savings_ratio * 100:.2f}%
<!-- BENCHMARK_METRICS_END -->"""

    # Use regex to replace the block
    new_content = re.sub(
        r"<!-- BENCHMARK_METRICS_START -->.*?<!-- BENCHMARK_METRICS_END -->",
        markdown_table,
        content,
        flags=re.DOTALL
    )
    
    with open(readme_path, "w") as f:
        f.write(new_content)
    print("README.md dynamically updated with latest benchmark metrics.")

if __name__ == "__main__":
    generate_memory_plot()
