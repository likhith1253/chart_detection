"""
Script to generate test images for the chart research project.
Generates sample charts using matplotlib for evaluation.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path to access config.py
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    import config
    output_dir = config.RAW_IMAGE_DIR
except ImportError:
    print("Warning: Could not import config.py. Using default ./data/raw_images")
    output_dir = project_root / "data" / "raw_images"
    output_dir.mkdir(parents=True, exist_ok=True)

def generate_bar_chart():
    """Generates a bar chart and saves it."""
    categories = ['Category A', 'Category B', 'Category C', 'Category D']
    values = [25, 40, 30, 55]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, values, color='skyblue')

    plt.title('Sales per Category (Bar Chart)', fontsize=16)
    plt.xlabel('Categories', fontsize=12)
    plt.ylabel('Sales ($)', fontsize=12)

    # Add data labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, yval, ha='center', va='bottom')

    path = output_dir / "bar_chart_01.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Generated: {path}")

def generate_line_chart():
    """Generates a line chart and saves it."""
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    growth = [10, 15, 12, 20, 25, 22]

    plt.figure(figsize=(8, 6))
    plt.plot(months, growth, marker='o', linestyle='-', color='coral')

    plt.title('Monthly User Growth (Line Chart)', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('New Users', fontsize=12)

    # Add data labels
    for x, y in zip(months, growth):
        plt.text(x, y + 1, str(y), ha='center', va='bottom')

    path = output_dir / "line_chart_01.png"
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Generated: {path}")

def generate_pie_chart():
    """Generates a pie chart and saves it."""
    labels = ['Product A', 'Product B', 'Product C', 'Product D']
    sizes = [35, 25, 20, 20]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    explode = (0.1, 0, 0, 0)  # Extract the first slice

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
            autopct='%1.1f%%', shadow=True, startangle=140)

    plt.title('Market Share Distribution (Pie Chart)', fontsize=16)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    path = output_dir / "pie_chart_01.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Generated: {path}")

def generate_scatter_plot():
    """Generates a scatter plot and saves it."""
    np.random.seed(42)  # For reproducibility
    age = np.random.randint(18, 65, 50)
    income = age * 1.5 + np.random.randn(50) * 15

    plt.figure(figsize=(8, 6))
    plt.scatter(age, income, c='green', alpha=0.6, edgecolors='none')

    plt.title('Age vs Income (Scatter Plot)', fontsize=16)
    plt.xlabel('Age (Years)', fontsize=12)
    plt.ylabel('Income (Thousands)', fontsize=12)

    path = output_dir / "scatter_plot_01.png"
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Generated: {path}")

def generate_histogram():
    """Generates a histogram and saves it."""
    np.random.seed(42)
    test_scores = np.random.normal(75, 10, 200)

    plt.figure(figsize=(8, 6))
    n, bins, patches = plt.hist(test_scores, bins=15, color='purple', alpha=0.7, edgecolor='black')

    plt.title('Distribution of Test Scores (Histogram)', fontsize=16)
    plt.xlabel('Score Range', fontsize=12)
    plt.ylabel('Number of Students', fontsize=12)

    path = output_dir / "histogram_01.png"
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Generated: {path}")

def  main():
    """Main execution block to generate all charts."""
    print("Generating synthetic chart images...")
    generate_bar_chart()
    generate_line_chart()
    generate_pie_chart()
    generate_scatter_plot()
    generate_histogram()
    print("All charts generated successfully.")

if __name__ == "__main__":
    main()
