import os
import matplotlib.pyplot as plt
import numpy as np
from distributed_trainer.core.memory_manager import compare_memory_with_and_without_optimizations

def generate_memory_plot():
    # Dummy network
    x = np.random.randn(128, 1024).astype(np.float32)
    params = {
        'W1': np.random.randn(1024, 4096).astype(np.float32),
        'W2': np.random.randn(4096, 1024).astype(np.float32)
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

if __name__ == "__main__":
    generate_memory_plot()
