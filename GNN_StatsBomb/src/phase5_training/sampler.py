"""
Player-aware batch sampler for contrastive learning.

Guarantees that every mini-batch contains multiple possessions per player,
so the InfoNCE contrastive loss always has positive pairs.  Without this,
random shuffling produces batches where most players appear only once,
starving the contrastive loss of signal.

Designed as a drop-in ``batch_sampler`` for ``torch.utils.data.DataLoader``.
When active, ``DataLoader`` must **not** set ``batch_size``, ``shuffle``, or
``sampler`` (they are mutually exclusive with ``batch_sampler``).
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterator, List

from torch.utils.data import Sampler


class PlayerAwareBatchSampler(Sampler[List[int]]):
    """Yield batches of K players x M possessions each.

    Parameters
    ----------
    player_index : dict[int, list[int]]
        Mapping from player_id to the dataset indices where that player is
        the actor.  Built by ``PossessionGraphDataset.build_player_index()``.
    players_per_batch : int
        Number of distinct players per batch (K).
    possessions_per_player : int
        Number of possessions sampled per player per batch (M).
    total_possessions : int
        Total number of possessions in the training set.  Used to keep epoch
        length comparable to vanilla random-shuffle training.
    seed : int
        Random seed for reproducibility.  Incremented each epoch.
    """

    def __init__(
        self,
        player_index: Dict[int, List[int]],
        players_per_batch: int = 16,
        possessions_per_player: int = 6,
        total_possessions: int = 0,
        seed: int = 42,
    ):
        self.player_index = player_index
        self.K = players_per_batch
        self.M = possessions_per_player
        self.batch_size = self.K * self.M
        self.seed = seed
        self._epoch = 0

        self.player_ids = list(player_index.keys())
        self._total_possessions = total_possessions or sum(
            len(v) for v in player_index.values()
        )

    # ------------------------------------------------------------------
    # Iterator — called once per epoch by DataLoader
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        per_player_queues: Dict[int, List[int]] = {}
        for pid, indices in self.player_index.items():
            q = list(indices)
            rng.shuffle(q)
            per_player_queues[pid] = q

        n_batches = len(self)
        player_pool: List[int] = []

        for _ in range(n_batches):
            if len(player_pool) < self.K:
                fresh = list(self.player_ids)
                rng.shuffle(fresh)
                player_pool.extend(fresh)

            chosen = player_pool[:self.K]
            player_pool = player_pool[self.K:]

            batch_indices: List[int] = []
            for pid in chosen:
                queue = per_player_queues[pid]
                if len(queue) < self.M:
                    queue = list(self.player_index[pid])
                    rng.shuffle(queue)
                    per_player_queues[pid] = queue

                selected = queue[:self.M]
                per_player_queues[pid] = queue[self.M:]

                if len(selected) < self.M:
                    extra = rng.choices(self.player_index[pid], k=self.M - len(selected))
                    selected.extend(extra)

                batch_indices.extend(selected)

            yield batch_indices

    # ------------------------------------------------------------------
    # Length — number of batches per epoch
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return math.ceil(self._total_possessions / self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        """Allow external epoch tracking (e.g. for distributed training)."""
        self._epoch = epoch
