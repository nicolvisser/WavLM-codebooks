from pathlib import Path
from simple_parsing import ArgumentParser

import faiss
import numpy as np
import torch
from sklearn.cluster import kmeans_plusplus
from tqdm import tqdm


def main():
    # Replace click options with simple_parsing
    parser = ArgumentParser()
    
    # Define the arguments using simple_parsing
    parser.add_argument("--input-dir", "-i", type=Path, required=True, help="Input directory.")
    parser.add_argument("--match-pattern", "-m", type=str, default="**/*.pt", help="Match pattern for input files.")
    parser.add_argument("--output-dir", "-o", type=Path, required=True, help="Output directory.")
    parser.add_argument("--n-clusters", "-k", type=int, default=500, help="Number of clusters.")
    parser.add_argument("--keep-percentage", "-p", type=float, default=0.1, help="Percentage of features to keep.")
    parser.add_argument("--n-iter", "-n", type=int, default=300, help="Number of iterations.")
    
    args = parser.parse_args()  # Parse the arguments

    # Use args to access the parameters
    input_dir = args.input_dir
    match_pattern = args.match_pattern
    output_dir = args.output_dir
    n_clusters = args.n_clusters
    keep_percentage = args.keep_percentage
    n_iter = args.n_iter

    # define output paths
    output_dir.mkdir(parents=True, exist_ok=True)
    init_centroids_path = output_dir / "init_centroids.npy"
    state_dict_path = output_dir / "codebook.pt"
    objective_path = output_dir / "objective.csv"

    print("Loading features into memory...")
    feature_paths = list(input_dir.glob(match_pattern))
    features = []
    for feature_path in tqdm(feature_paths):
        feats = np.load(feature_path)
        assert feats.ndim == 2
        np.random.shuffle(feats)
        feats = feats[: int(feats.shape[0] * keep_percentage)]
        features.append(feats)
    features = np.concatenate(features, axis=0)  # may consume a lot of memory

    print("Features shape:", features.shape)

    print("Initializing centroids with k-means++...")
    init_centers, init_indices = kmeans_plusplus(
        X=features,
        n_clusters=n_clusters,
    )

    np.save(init_centroids_path, init_centers)
    print(f"Saved initial centroids to {init_centroids_path}")

    print("Training k-means...")
    kmeans = faiss.Kmeans(
        d=features.shape[1],
        k=n_clusters,
        niter=n_iter,
        verbose=True,
        spherical=False,
        max_points_per_centroid=1_000_000,  # high enough to use all
    )
    kmeans.train(features, init_centroids=init_centers)

    np.savetxt(
        objective_path,
        np.array(kmeans.obj).reshape(-1, 1),
        delimiter=",",
    )
    print(f"Saved objective function to {objective_path}")
    print("Inspect the objective function to check for convergence.")

    state_dict = {
        "codebook": torch.from_numpy(kmeans.centroids),
    }
    torch.save(state_dict, state_dict_path)
    print(f"Saved centroids to {state_dict_path}")


if __name__ == "__main__":
    main()
