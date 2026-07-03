#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

N = 2048
RADIUS = 1.0 / 16.0
COLLIDE_DIST = 2 * RADIUS          # equal-radius pair
SPRING, DAMPING, SHEAR, ATTRACTION = 0.5, 0.02, 0.1, 0.0
COLLIDER_POS = np.array([-1.2, -0.8, 0.8])
COLLIDER_RADIUS = 0.2


def pair_forces(pos, vel, pos2, vel2, collide_dist):
    """DEM sphere-sphere force of particle(s) `pos` against `pos2` (float64)."""
    rel = pos2 - pos
    dist = np.sqrt((rel ** 2).sum(axis=-1))
    hit = (dist < collide_dist) & (dist > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = rel / dist[..., None]
    relvel = vel2 - vel
    tanvel = relvel - (relvel * norm).sum(axis=-1)[..., None] * norm
    f = (-SPRING * (collide_dist - dist))[..., None] * norm
    f += DAMPING * relvel
    f += SHEAR * tanvel
    f += ATTRACTION * rel
    return np.where(hit[..., None], f, 0.0)


def reference(meta):
    pos = np.stack([2.0 * V.gen_hash01(N, 1).astype(np.float64) - 1.0,
                    2.0 * V.gen_hash01(N, 2).astype(np.float64) - 1.0,
                    2.0 * V.gen_hash01(N, 3).astype(np.float64) - 1.0], axis=1)
    vel = np.stack([0.02 * (2.0 * V.gen_hash01(N, 4).astype(np.float64) - 1.0),
                    0.02 * (2.0 * V.gen_hash01(N, 5).astype(np.float64) - 1.0),
                    0.02 * (2.0 * V.gen_hash01(N, 6).astype(np.float64) - 1.0)], axis=1)

    # With cellSize == collideDist, the 27-cell traversal reaches exactly every
    # pair closer than collideDist, so an O(n^2) cutoff sum is equivalent.
    f = pair_forces(pos[:, None, :], vel[:, None, :],
                    pos[None, :, :], vel[None, :, :], COLLIDE_DIST)
    force = f.sum(axis=1)

    # cursor/collider sphere
    force += pair_forces(pos, vel, COLLIDER_POS[None, :],
                         np.zeros((1, 3)), RADIUS + COLLIDER_RADIUS).reshape(N, 3)

    newvel = vel + force
    return newvel.reshape(-1).astype(V.F32)


if __name__ == "__main__":
    V.run(reference)
