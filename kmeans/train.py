from pathlib import Path

import click
import faiss
import numpy as np
import torch
from sklearn.cluster import kmeans_plusplus
from tqdm import tqdm


@click.command()
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    prompt=True,
)
@click.option(
    "--match-pattern",
    "-m",
    type=str,
    prompt=True,
    default="**/*.pt",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    prompt=True,
)
@click.option(
    "--n-clusters",
    "-k",
    type=int,
    prompt=True,
    default=500,
)
@click.option(
    "--keep-percentage",
    "-p",
    type=float,
    prompt=True,
    default=0.1,
)
@click.option(
    "--n-iter",
    "-n",
    type=int,
    prompt=True,
    default=300,
)
def main(
    input_dir: Path,
    match_pattern: str,
    output_dir: Path,
    n_clusters: int,
    keep_percentage: float,
    n_iter: int,
):
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
