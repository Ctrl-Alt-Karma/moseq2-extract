"""Deterministic-RANSAC qualification. Synthetic fixtures only."""
import hashlib

import numpy as np

from moseq2_extract.extract.roi import plane_ransac
from moseq2_extract.util import EXTRACT_OUTPUT_POLICIES


def synthetic_depth(seed=1234):
    """A tilted synthetic floor with noise and an object. No candidate data."""
    rs = np.random.RandomState(seed)
    h, w = 60, 80
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    floor = 700.0 + 0.35 * xx - 0.20 * yy
    depth = floor + rs.normal(0, 2.0, size=floor.shape)
    depth[20:35, 25:45] -= 60.0          # an object sitting on the floor
    return depth.astype("float64")


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a)
        h.update(a.dtype.str.encode())
        h.update(repr(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes(order="C"))
    return h.hexdigest()


DEPTH_RANGE = (600, 800)


def test_repeated_calls_are_bit_identical():
    depth = synthetic_depth()
    first = plane_ransac(depth, bg_roi_depth_range=DEPTH_RANGE, iters=50)
    for _ in range(4):
        again = plane_ransac(depth, bg_roi_depth_range=DEPTH_RANGE, iters=50)
        assert digest(*first) == digest(*again)


def test_global_rng_state_does_not_change_result():
    depth = synthetic_depth()
    np.random.seed(0)
    a = plane_ransac(depth, bg_roi_depth_range=DEPTH_RANGE, iters=50)
    np.random.seed(999999)
    b = plane_ransac(depth, bg_roi_depth_range=DEPTH_RANGE, iters=50)
    np.random.random(1000)
    c = plane_ransac(depth, bg_roi_depth_range=DEPTH_RANGE, iters=50)
    assert digest(*a) == digest(*b) == digest(*c)


def test_does_not_consume_global_rng_state():
    depth = synthetic_depth()
    np.random.seed(4242)
    before = np.random.get_state()
    plane_ransac(depth, bg_roi_depth_range=DEPTH_RANGE, iters=50)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_mask_variant_also_deterministic():
    depth = synthetic_depth()
    mask = np.zeros(depth.shape, dtype=bool)
    mask[5:55, 5:75] = True
    a = plane_ransac(depth, bg_roi_depth_range=DEPTH_RANGE, iters=50, mask=mask)
    b = plane_ransac(depth, bg_roi_depth_range=DEPTH_RANGE, iters=50, mask=mask)
    assert digest(*a) == digest(*b)


def test_plane_is_still_a_real_fit():
    """Guard that determinism did not come from breaking the fit."""
    depth = synthetic_depth()
    plane, dist = plane_ransac(depth, bg_roi_depth_range=DEPTH_RANGE, iters=50)
    assert plane.shape == (4,)
    assert np.all(np.isfinite(plane))
    assert dist.shape == (depth.size,)
    # the floor pixels should sit close to the fitted plane
    assert float(np.median(dist)) < 30.0


def test_output_policy_records_determinism():
    assert EXTRACT_OUTPUT_POLICIES["roi_plane_fit"] == \
        "lstsq-inlier-refit-deterministic-rng0"


def test_no_global_seeding_in_source():
    import inspect
    from moseq2_extract.extract import roi as roi_mod
    src = inspect.getsource(roi_mod)
    assert "np.random.seed(" not in src
    assert "np.random.choice(" not in src
    assert "np.random.RandomState(0)" in src


def _run():
    """Minimal runner: this environment has no pytest installed."""
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print("  %-52s PASS" % name)
        except Exception:
            failed += 1
            print("  %-52s FAIL" % name)
            traceback.print_exc()
    print("TOTAL %d  FAILED %d" % (len(fns), failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_run())
