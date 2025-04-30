import torch
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def run_one_episode(model, state_np, action_np, max_steps, device, latent_dim):
    """
    Runs the model on a single episode and returns:
      - mus: list of (T_i, latent_dim) arrays, one per segment
      - boundaries: list of segment-end positions
    """
    s = torch.from_numpy(state_np[None]).to(device)    # [1, T, state_dim]
    a = torch.from_numpy(action_np[None]).to(device)   # [1, T, action_dim]
    lengths = torch.full((1,), max_steps, dtype=torch.long, device=device)

    with torch.no_grad():
        _, _, _, all_b, all_z = model((s, a), lengths)

    # extract means (mus)
    mus = [logits_z[:, :latent_dim].squeeze(0).cpu().numpy()
           for logits_z in all_z['logits']]

    # extract boundary positions
    boundary_positions = [b.squeeze(0).argmax().item()
                          for b in all_b['samples']]

    return mus, boundary_positions


def extract_segment_latents(model, states, actions, max_steps, device, latent_dim):
    """
    Loops over all episodes to extract segment mus and boundary positions.
    Returns:
      - all_latents: list of lists of mus arrays
      - all_boundaries: list of boundary lists
    """
    all_latents, all_boundaries = [], []
    for s_np, a_np in zip(states, actions):
        mus, boundaries = run_one_episode(model, s_np, a_np,
                                         max_steps, device, latent_dim)
        all_latents.append(mus)
        all_boundaries.append(boundaries)
    return all_latents, all_boundaries


def cluster_latents(all_latents, n_clusters, random_seed):
    """
    Flattens and clusters latents with a sklearn pipeline (StandardScaler + KMeans).
    Returns:
      - labels_flat: array of shape (n_points,) with cluster assignments
      - centroids: array of shape (n_clusters, latent_dim)
    """
    flat = np.vstack(all_latents)
    pipeline = make_pipeline(
        StandardScaler(),
        KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init='auto',
            max_iter=300,
            random_state=random_seed,
        )
    )
    labels_flat = pipeline.fit_predict(flat)
    centroids = pipeline.named_steps['kmeans'].cluster_centers_
    return labels_flat, centroids


def labels_per_timestep(boundaries, seg_labels, T):
    """
    Creates a per-timestep label array of length T,
    filtering and sorting boundaries to avoid negative segments.
    """
    # pair boundaries with labels and filter out-of-range
    pairs = [(int(b), int(lbl))
             for b, lbl in zip(boundaries, seg_labels)
             if 0 < b < T]
    if not pairs:
        # no valid boundaries: repeat first label
        fill = seg_labels[0] if seg_labels else 0
        return np.full(T, fill, dtype=int)

    # sort by boundary position
    pairs.sort(key=lambda x: x[0])
    b_sorted, lbls_sorted = zip(*pairs)

    # create segment lengths
    ends = np.concatenate(([0], b_sorted, [T]))
    lengths = np.diff(ends)

    # extend labels with last label for tail
    labels_ext = np.array(lbls_sorted + (lbls_sorted[-1],), dtype=int)
    return np.repeat(labels_ext, lengths)
