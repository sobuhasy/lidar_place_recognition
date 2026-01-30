import numpy as np
import matplotlib.pyplot as plt
from src.dataset.loader import load_scan, get_scan_files

# PATH TO YOUR EXTRACTED DATA (Adjust if necessary)
VEL_DIR = "data/raw/2012-01-08_vel" 

def visualize_scan():
    files = get_scan_files(VEL_DIR)
    if not files:
        print("No .bin files found! Did you extract the archive?")
        return

    # Load the first scan
    print(f"Loading: {files[0]}")
    points = load_scan(str(files[0]))
    print(f"Point Cloud Shape: {points.shape}") # Should be approx (25000, 4)

    # Simple 2D Top-Down View (X vs Y)
    plt.figure(figsize=(10, 10))
    # Plot every 10th point to speed it up
    plt.scatter(points[::10, 0], points[::10, 1], s=0.5, c=points[::10, 3]) 
    plt.title(f"NCLT Scan: Top Down View\n{files[0].name}")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    visualize_scan()