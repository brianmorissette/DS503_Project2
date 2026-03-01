import csv
import random
from typing import List, Tuple


NUM_POINTS = 6000

W_MIN, W_MAX = 0.0, 10_000.0
X_MIN, X_MAX = 20_000.0, 1_000_000.0
Y_MIN, Y_MAX = 0.0, 500_000.0
Z_MIN, Z_MAX = 0.0, 50_000.0

OUTPUT_FILENAME = "random_dataset.csv"


def generate_point() -> Tuple[float, float, float, float]:
    """Generate a single 4D point (w, x, y, z) within the specified ranges."""
    w = random.uniform(W_MIN, W_MAX)
    x = random.uniform(X_MIN, X_MAX)
    y = random.uniform(Y_MIN, Y_MAX)
    z = random.uniform(Z_MIN, Z_MAX)
    return w, x, y, z


def generate_dataset(num_points: int) -> List[Tuple[float, float, float, float]]:
    """Generate a list of 4D points."""
    return [generate_point() for _ in range(num_points)]


def write_dataset_to_csv(filename: str, points: List[Tuple[float, float, float, float]]) -> None:
    """Write the dataset to a CSV file with header w,x,y,z."""
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["w", "x", "y", "z"])
        writer.writerows(points)


def summarize_dimensions(points: List[Tuple[float, float, float, float]]):
    """Compute min and max for each dimension."""
    ws, xs, ys, zs = zip(*points)
    return {
        "w": (min(ws), max(ws)),
        "x": (min(xs), max(xs)),
        "y": (min(ys), max(ys)),
        "z": (min(zs), max(zs)),
    }


def main() -> None:
    points = generate_dataset(NUM_POINTS)
    write_dataset_to_csv(OUTPUT_FILENAME, points)

    print(f"Generated {len(points)} points and wrote them to {OUTPUT_FILENAME}.")

    summary = summarize_dimensions(points)
    for dim in ("w", "x", "y", "z"):
        dim_min, dim_max = summary[dim]
        print(f"{dim}: min={dim_min:.4f}, max={dim_max:.4f}")

    # Optionally print a small sample for sanity check
    sample_size = min(5, len(points))
    print(f"\nFirst {sample_size} points:")
    for point in points[:sample_size]:
        print(point)


if __name__ == "__main__":
    main()

