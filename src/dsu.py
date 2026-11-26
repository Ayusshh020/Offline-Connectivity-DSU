"""Disjoint Set Union (DSU) / Union-Find Data Structure.

Implements Disjoint Set Union with:
1. Path Compression: Flattens the tree during find operations.
2. Union by Rank or Size: Keeps tree height minimal by attaching smaller trees
   under the root of larger trees.

This achieves an amortized time complexity of O(alpha(N)) per operation,
where alpha is the Inverse Ackermann function (effectively constant, <= 4
for all practical graph sizes).
"""

from __future__ import annotations

from typing import List


class DisjointSetUnion:
    """Disjoint Set Union (DSU) data structure supporting fast union and find."""

    def __init__(self, n: int, use_size: bool = False) -> None:
        """Initialize DSU with `n` elements (0 to n - 1).

        Args:
            n: Number of elements/vertices.
            use_size: If True, uses union-by-size; if False (default),
                      uses union-by-rank.
        """
        if n < 0:
            raise ValueError(f"Number of elements cannot be negative, got {n}")

        self.n: int = n
        self.use_size: bool = use_size
        # Each element initially points to itself as its own root
        self.parent: List[int] = list(range(n))

        if self.use_size:
            # Component sizes initialized to 1
            self.size: List[int] = [1] * n
            self.rank: List[int] = []
        else:
            # Tree depths/ranks initialized to 0
            self.rank: List[int] = [0] * n
            self.size: List[int] = []

        self._num_components: int = n

    def find(self, i: int) -> int:
        """Find the representative (root) of the set containing element `i`.

        Uses iterative path compression to avoid Python recursion limits on
        deep trees (such as long chain graphs).

        Args:
            i: Vertex index (0 <= i < n).

        Returns:
            Root representative of vertex `i`.
        """
        if i < 0 or i >= self.n:
            raise IndexError(f"Element index {i} out of bounds for size {self.n}")

        # Step 1: Find root
        root = i
        while root != self.parent[root]:
            root = self.parent[root]

        # Step 2: Path compression - point every node on path directly to root
        curr = i
        while curr != root:
            parent_node = self.parent[curr]
            self.parent[curr] = root
            curr = parent_node

        return root

    def union(self, u: int, v: int) -> bool:
        """Unite the sets containing vertices `u` and `v`.

        Args:
            u: First vertex.
            v: Second vertex.

        Returns:
            True if `u` and `v` were in different components and successfully
            merged; False if they were already in the same component.
        """
        root_u = self.find(u)
        root_v = self.find(v)

        # Already in the same component
        if root_u == root_v:
            return False

        if self.use_size:
            # Union by size: attach smaller component under root of larger
            if self.size[root_u] < self.size[root_v]:
                self.parent[root_u] = root_v
                self.size[root_v] += self.size[root_u]
            else:
                self.parent[root_v] = root_u
                self.size[root_u] += self.size[root_v]
        else:
            # Union by rank: attach tree with smaller rank under larger rank
            if self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            elif self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            else:
                # Same rank: arbitrarily attach root_v under root_u and increase rank
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

        self._num_components -= 1
        return True

    def connected(self, u: int, v: int) -> bool:
        """Check if vertices `u` and `v` belong to the same connected component.

        Args:
            u: First vertex.
            v: Second vertex.

        Returns:
            True if connected, False otherwise.
        """
        if u < 0 or u >= self.n or v < 0 or v >= self.n:
            raise IndexError(f"Vertex index out of bounds: u={u}, v={v}, n={self.n}")
        return self.find(u) == self.find(v)

    @property
    def num_components(self) -> int:
        """Current number of connected components."""
        return self._num_components
