import torch
from typing import Dict, Tuple, Optional


class TacticalKNNIndexer:
    """
    Executes a GPU-accelerated Euclidean distance calculation with torch.cdist() against a pre-compiled
    database of M >= 10,000 historical match feature vectors and aggregates transition ledgers to construct
    the k-NN pseudo-prior matrix Q_KNN.
    """

    def __init__(
        self,
        historical_database: Optional[Dict[int, torch.Tensor]] = None,
        n_future_database: Optional[Dict[int, torch.Tensor]] = None,
        T_future_database: Optional[Dict[int, torch.Tensor]] = None,
        k_neighbours: int = 50,
    ):
        """
        historical_database: a dictionary mapping minute timestamps (which are the keys, e.g.
        10, 20, 30...) to a 2D PyTorch tensor of shape (M=num_matches, 5) containing normalised
        historical vectors.
        n_future_database: FILL ME IN
        T_future_database: FILL ME IN
        """

        self.use_pinned = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_pinned else "cpu")
        self.k = k_neighbours

        # if no historical dbs passed, initialise an empty registry
        self.db = historical_database or {}  # shape per minute: (M, 5)
        self.n_future_db = n_future_database or {}  # shape per minute: (M, 12, 12)
        self.T_future_db = T_future_database or {}  # shape per minute: (M, 12)

    def register_historical_slice(
        self,
        minute_timestamp: int,
        vectors_matrix: torch.Tensor,
        n_future_matrix: Optional[torch.Tensor] = None,
        T_future_matrix: Optional[torch.Tensor] = None,
    ):
        """
        Loads the feature vectors and corresponding future ledgers into memory for a specific match minute.
        Auto pushes the database to the active hardware device (GPU VRAM or CPU RAM).
        """
        self.db[minute_timestamp] = vectors_matrix.to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        if n_future_matrix is not None and T_future_matrix is not None:
            self.n_future_db[minute_timestamp] = n_future_matrix.to(
                self.device, dtype=torch.float32, non_blocking=True
            )
            self.T_future_db[minute_timestamp] = T_future_matrix.to(
                self.device, dtype=torch.float32, non_blocking=True
            )

    def find_nearest_neighbours(
        self, live_vector: torch.Tensor, clock_seconds: float
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Ingests the normalised (5,) live vector from the TacticalVectoriser class and the match clock
        timestamp t. Returns:
            top_k_distances: Shape (k,), containing the distances to the k most similar historical fixtures.
            top_k_indices: Shape (k,) containing the row indices of the k most similar historical fixtures.
            minute_bucket: the current minute of the match.
        """

        # map current match clock to the nearest integer minute bucket
        minute_bucket = int(round(clock_seconds / 60.0))

        # fallback if that minute isn't in the database
        if minute_bucket not in self.db:
            if not self.db:
                raise ValueError(
                    "[-] Critical Error: Historical K-NN database is completely empty!"
                )
            # get the minute that's got the lowest distance to the one we want
            minute_bucket = min(self.db.keys(), key=lambda k: abs(k - minute_bucket))

        historical_matrix = self.db[
            minute_bucket
        ]  # shape: (M, 5) [so M rows and 5 columns]

        # make sure k isn't greater than the number of historical matches
        num_historical_matches = historical_matrix.shape[0]
        active_k = min(self.k, num_historical_matches)

        # reshape live vector from (5,) to (1, 5) to satisfy torch.cdist matrix dimensions
        # .unsqueeze(0) adds a dimension at index 0, so (5,) becomes (1, 5)
        live_query = live_vector.to(device=self.device, non_blocking=True).unsqueeze(0)

        # native PyTorch Euclidean distance calculation
        # output shape of distances: (1, M), so it's a row vector
        distances = torch.cdist(live_query, historical_matrix, p=2.0)

        # extract the k lowest distances (largest=False grabs minimum distances)
        # .squeeze removes the extra dimension at the start (the 'batch dimension'), leaving
        # 1D tensors of shape (k,)
        top_k_distances, top_k_indices = torch.topk(
            distances, k=active_k, largest=False
        )

        return top_k_distances.squeeze(0), top_k_indices.squeeze(0), minute_bucket

    def build_pseudo_prior(
        self, top_k_indices: torch.Tensor, minute_bucket: int, epsilon: float = 1e-6
    ) -> torch.Tensor:
        """
        Aggregates future transition counts and holding times of the K neighbour matches and calculates
        lambdas to generate the dense pseudo-prior transition intensity matrix Q_KNN.
        """
        if (
            minute_bucket not in self.n_future_db
            or minute_bucket not in self.T_future_db
        ):
            raise ValueError(
                f"[-] Future ledgers for minute {minute_bucket} are not registered in memory!"
            )
        n_slice = self.n_future_db[minute_bucket]  # shape: (M, 12, 12)
        T_slice = self.T_future_db[minute_bucket]  # shape: (M, 12)

        # extract the K matrices of the most similar matches, no matter what order they're in
        n_knn = n_slice[top_k_indices]  # shape: (K, 12, 12)
        # sum up all the transition counts in one vectorised burst!
        n_knn = n_knn.sum(dim=0)  # shape: (12, 12)

        T_knn = T_slice[top_k_indices]  # shape: (K, 12)
        T_knn = T_knn.sum(dim=0)  # shape: (12,)

        # do unsqueeze(1) to add an extra dimension to T_knn, so it goes from
        # shape (12,) to shape (12, 1)
        lambda_knn = n_knn / (T_knn.unsqueeze(1) + epsilon)

        return lambda_knn

    def get_pseudo_prior(
        self, live_vector: torch.Tensor, clock_seconds: float, epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Master pipeline for stage 3:
        1. Computes Euclidean distance to find K nearest neighbours.
        2. Aggregates future ledgers to construct Q_KNN.
        Returns: lambda_knn, top_k_distances, top_k_indices.
        """
        distances, indices, minute_bucket = self.find_nearest_neighbours(
            live_vector, clock_seconds
        )
        lambda_knn = self.build_pseudo_prior(indices, minute_bucket, epsilon)
        return lambda_knn, distances, indices
