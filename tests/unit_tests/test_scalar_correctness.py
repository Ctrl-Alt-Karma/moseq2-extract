"""
Regression tests for scalar-computation correctness.

These cover cases where the extracted numbers were wrong (rather than merely
noisy), so they are worth pinning explicitly.
"""
import numpy as np
import numpy.testing as npt
from unittest import TestCase

from moseq2_extract.extract.proc import feature_hampel_filter


def _glitched_track(n=200, value=150.0, glitch_index=100, glitch=300.0, seed=0):
    """A smooth 1-D track containing a single large tracking glitch."""
    rng = np.random.RandomState(seed)
    track = np.full(n, value) + rng.normal(0, 0.5, n)
    track[glitch_index] = glitch
    return track


class TestHampelFilter(TestCase):
    def test_removes_centroid_glitch(self):
        # The threshold used to be `median + sig * MAD`, which added the median
        # *position* (~150 px) to the tolerance, so no outlier ever exceeded it
        # and the filter silently did nothing.
        x = _glitched_track(seed=0)
        y = _glitched_track(value=120.0, glitch=280.0, seed=1)
        features = {
            'centroid': np.stack([x, y], axis=1),
            'orientation': np.zeros(len(x)),
        }

        out = feature_hampel_filter(features, centroid_hampel_span=5,
                                    centroid_hampel_sig=3)

        # the glitch is replaced by the local median, not left at 300/280
        assert out['centroid'][100, 0] < 200
        assert out['centroid'][100, 1] < 200

    def test_filters_both_centroid_axes(self):
        # The loop ran `range(1)`, so only the x coordinate was ever filtered
        # and y-axis tracking glitches survived into velocity and distance.
        clean = np.full(200, 150.0)
        y = _glitched_track(value=150.0, glitch=400.0, seed=2)
        features = {
            'centroid': np.stack([clean.copy(), y], axis=1),
            'orientation': np.zeros(200),
        }

        out = feature_hampel_filter(features, centroid_hampel_span=5,
                                    centroid_hampel_sig=3)

        assert out['centroid'][100, 1] < 200, "y coordinate was not filtered"

    def test_angle_filter_without_centroid_filter(self):
        # `padded_orientation` used to be built inside the centroid branch, so
        # enabling only the angle filter raised UnboundLocalError.
        orientation = _glitched_track(value=0.1, glitch=3.0, seed=3)
        features = {
            'centroid': np.stack([np.full(200, 150.0), np.full(200, 150.0)], axis=1),
            'orientation': orientation,
        }

        out = feature_hampel_filter(features, centroid_hampel_span=0,
                                    angle_hampel_span=5, angle_hampel_sig=3)

        assert out['orientation'][100] < 1.0

    def test_leaves_clean_track_untouched(self):
        # A track with no outliers must survive the filter essentially unchanged.
        rng = np.random.RandomState(4)
        track = np.full(200, 150.0) + rng.normal(0, 0.1, 200)
        features = {
            'centroid': np.stack([track.copy(), track.copy()], axis=1),
            'orientation': np.zeros(200),
        }

        out = feature_hampel_filter(features, centroid_hampel_span=5,
                                    centroid_hampel_sig=6)

        npt.assert_allclose(out['centroid'][:, 0], track, atol=1.0)


class TestAreaUnits(TestCase):
    def test_area_mm_uses_squared_pixel_factor(self):
        # area_px is a pixel count (px^2); converting to mm^2 needs the product
        # of the x and y linear mm/px factors, not their mean. Using the mean
        # left the result in px*mm and under-reported the area.
        area_px = np.array([1500.0])
        px_to_mm = np.array([[1.62, 1.66]])

        correct = area_px * px_to_mm[:, 0] * px_to_mm[:, 1]
        old_behaviour = area_px * px_to_mm.mean(axis=1)

        npt.assert_allclose(correct, [1500.0 * 1.62 * 1.66])
        # the old result was smaller by roughly one linear factor
        assert old_behaviour[0] < correct[0]
        npt.assert_allclose(correct[0] / old_behaviour[0],
                            (1.62 * 1.66) / np.mean([1.62, 1.66]), rtol=1e-6)
