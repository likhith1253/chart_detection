import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def generate_bar_chart(save_path, index):
    plt.figure(figsize=(10, 6))
    categories = [f'Cat {i}' for i in range(5)]
    values = [random.randint(10, 100) for _ in range(5)]
    plt.bar(categories, values, color=plt.cm.Paired(np.linspace(0, 1, 5)))
    plt.title(f'Bar Chart {index}')
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.savefig(save_path)
    plt.close()

def generate_line_chart(save_path, index):
    plt.figure(figsize=(10, 6))
    x = np.linspace(0, 10, 10)
    y = np.random.randint(0, 100, 10)
    plt.plot(x, y, marker='o', linestyle='-', color=np.random.rand(3,))
    plt.title(f'Line Chart {index}')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.savefig(save_path)
    plt.close()

def generate_scatter_plot(save_path, index):
    plt.figure(figsize=(10, 6))
    x = np.random.rand(50)
    y = np.random.rand(50)
    colors = np.random.rand(50)
    plt.scatter(x, y, c=colors, alpha=0.5, cmap='viridis')
    plt.title(f'Scatter Plot {index}')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.savefig(save_path)
    plt.close()

def generate_pie_chart(save_path, index):
    plt.figure(figsize=(8, 8))
    labels = [f'Part {i}' for i in range(4)]
    sizes = [random.randint(10, 50) for _ in range(4)]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Set3(np.linspace(0, 1, 4)))
    plt.title(f'Pie Chart {index}')
    plt.savefig(save_path)
    plt.close()

def generate_histogram(save_path, index):
    plt.figure(figsize=(10, 6))
    data = np.random.randn(1000)
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Histogram {index}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()

def main():
    output_dir = Path("data/raw_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        (generate_bar_chart, 300, 'bar'),
        (generate_line_chart, 200, 'line'),
        (generate_scatter_plot, 200, 'scatter'),
        (generate_pie_chart, 150, 'pie'),
        (generate_histogram, 150, 'hist')
    ]

    total_generated = 0
    for func, count, prefix in configs:
        print(f"Generating {count} {prefix} charts...")
        for i in tqdm(range(count)):
            filename = f"synthetic_{prefix}_{i:05d}.png"
            func(output_dir / filename, total_generated)
            total_generated += 1

    print(f"Successfully generated {total_generated} charts.")

if __name__ == "__main__":
    main()
