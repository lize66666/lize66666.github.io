"""Rebuild the embedded 8-particle approximation. Requires Node, NumPy and SciPy."""
import base64
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial.distance import cdist


PAGE = Path(__file__).resolve().parents[1] / "index.html"
EPSILON = 24.0  # Squared scene-coordinate units, C_ij = ||x_i - y_j||².
PARTS = 8       # A 2 × 2 × 2 subdivision of each cube.


def load_geometry(html):
    geometry = html[html.index("const chinese ="):html.index("// Balanced eight-particle")]
    result = subprocess.run(
        ["node", "-"], input=geometry + "\nconsole.log(JSON.stringify([origin,target,pairing]));",
        encoding="utf-8", capture_output=True, check=True,
    )
    return [np.asarray(array) for array in json.loads(result.stdout)]


def sinkhorn(source, target):
    # Removing translations changes only row/column potentials for squared cost.
    cost = cdist(source - source.mean(axis=0), target - target.mean(axis=0), "sqeuclidean")
    kernel = np.exp(-cost / EPSILON)
    right = np.ones(len(target))
    for iteration in range(10000):
        left = 1 / (kernel @ right)
        right = 1 / (kernel.T @ left)
        if iteration % 10 == 0:
            plan = left[:, None] * kernel * right
            if np.max(np.abs(plan.sum(axis=1) - 1)) < 1e-12:
                np.testing.assert_allclose(plan.sum(axis=0), 1, rtol=0, atol=1e-12)
                return plan  # Conditional mass, one unit per source and target voxel.
    raise RuntimeError("Sinkhorn did not converge")


def balanced_particles(plan, source, target):
    # Split both row and column CDFs into eight equal-mass intervals in spatial order.
    # Their overlapping intervals form a doubly stochastic graph on the particle slots.
    # A perfect matching chooses one edge per interval on *both* sides.
    so = np.lexsort(source.T[::-1])
    to = np.lexsort(target.T[::-1])
    mass = PARTS * plan[so][:, to]
    row_starts = mass.cumsum(axis=1) - mass
    col_starts = mass.cumsum(axis=0) - mass
    rows, cols, weights = [], [], []
    for i, j in zip(*np.nonzero(mass > 1e-12)):
        length = mass[i, j]
        r, c = row_starts[i, j], col_starts[i, j]
        cuts = [0., length]
        for start in (r, c):
            cuts.extend(k - start for k in range(int(start)+1, int(start+length)+1)
                        if 0 < k-start < length)
        cuts.sort()
        for lo, hi in zip(cuts, cuts[1:]):
            if hi-lo < 1e-12:
                continue
            mid = (lo + hi) / 2
            rows.append(PARTS * so[i] + min(PARTS-1, int(r+mid)))
            cols.append(PARTS * to[j] + min(PARTS-1, int(c+mid)))
            weights.append(hi - lo)
    slots = PARTS * len(source)
    graph = coo_matrix((weights, (rows, cols)), shape=(slots, slots)).tocsr()
    # Favor representative overlaps; a deterministic seed breaks spatial symmetry.
    rng = np.random.default_rng(121)
    graph.data = 15 - np.log(graph.data) - rng.gumbel(size=graph.nnz)
    i, j = min_weight_full_bipartite_matching(graph)
    np.testing.assert_array_equal(i, np.arange(slots))
    np.testing.assert_array_equal(np.sort(j), i)
    counts = np.zeros_like(plan)
    np.add.at(counts, (i//PARTS, j//PARTS), 1/PARTS)
    for delta in ((counts-plan)[:, to], (counts-plan).T[:, so]):
        assert np.abs(delta.cumsum(axis=1)).max() <= 1/PARTS + 1e-10
    assert (np.count_nonzero(counts, axis=1) > 1).all()
    return j.astype("<u2")


if __name__ == "__main__":
    html = PAGE.read_text(encoding="utf-8")
    source, target, pairing = load_geometry(html)
    cost = cdist(source, target, "sqeuclidean")
    a, b = linear_sum_assignment(cost)
    np.testing.assert_array_equal(np.sort(pairing), a)
    np.testing.assert_allclose(cost[a, pairing].sum(), cost[a, b].sum(), rtol=1e-12)
    slots = balanced_particles(sinkhorn(source, target), source, target)
    encoded = base64.b64encode(slots.tobytes()).decode("ascii")
    html, changes = re.subn(r"const splitPlan = '[A-Za-z0-9+/=]*';",
                            f"const splitPlan = '{encoded}';", html)
    assert changes == 1
    PAGE.write_bytes(html.encode("utf-8"))
    print(f"Verified {len(source)} ordinary matches and {len(slots)} balanced particle paths.")
