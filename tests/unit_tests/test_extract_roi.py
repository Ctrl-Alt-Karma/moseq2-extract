import numpy as np
import numpy.testing as npt
from unittest import TestCase
from moseq2_extract.extract.roi import plane_fit3, plane_fit_lstsq, plane_ransac


class TestExtractROI(TestCase):
    def test_plane_fit3(self):
        # let's create two planes from simple equations and ensure
        # that everything looks kosher ya?

        # x,y plus intercept

        def plane_equation1(xy):
            return 2 * xy[:, 0] + 2 * xy[:, 1] + 2

        def plane_equation2(xy):
            return 1 * xy[:, 0] + 50 * xy[:, 1] + 3

        xy = np.random.randint(low=0, high=100, size=(3, 2))
        xyz = np.hstack((xy, plane_equation1(xy)[:, np.newaxis])).astype("float64")
        a = plane_fit3(xyz)
        norma = -a / a[2]

        npt.assert_almost_equal(norma[[0, 1, 3]], np.array([2, 2, 2]), 3)

        xyz = np.hstack((xy, plane_equation2(xy)[:, np.newaxis])).astype("float64")
        a = plane_fit3(xyz)
        norma = -a / a[2]

        npt.assert_almost_equal(norma[[0, 1, 3]], np.array([1, 50, 3]), 3)

    def test_plane_ransac(self):

        # make a plane where we can test adding noise...

        def plane_equation_noisy(xy, noise_scale=0):
            return (
                2 * xy[:, 0]
                + 5 * xy[:, 1]
                + np.random.normal(0, noise_scale, size=(xy.shape[0],))
            )

        xx, yy = np.meshgrid(np.arange(0, 50), np.arange(0, 50))
        xy = np.vstack((xx.ravel(), yy.ravel()))

        # low noise regime

        z = plane_equation_noisy(xy.T, noise_scale=0.1)
        tmp_img = z.reshape(xx.shape)

        a = plane_ransac(
            tmp_img, bg_roi_depth_range=(0, 1000), iters=1000, noise_tolerance=10
        )
        norma = -a[0] / a[0][2]

        npt.assert_almost_equal(np.round(norma[[0, 1]]), np.array([2, 5]), 1)

    def test_plane_fit_lstsq_beats_three_point(self):
        # least-squares plane fit over many noisy points should recover the
        # plane far better than a fit through only three of them
        np.random.seed(0)
        xx, yy = np.meshgrid(np.arange(60), np.arange(60))
        xy = np.vstack((xx.ravel(), yy.ravel())).T.astype("float64")
        z = 0.05 * xy[:, 0] + 0.03 * xy[:, 1] + 700 + np.random.normal(0, 3, len(xy))
        pts = np.hstack((xy, z[:, None]))

        lstsq = plane_fit_lstsq(pts)
        three = plane_fit3(pts[np.random.choice(len(pts), 3, replace=False)])

        def coeffs(p):
            return np.array([-p[0] / p[2], -p[1] / p[2]])

        lstsq_err = np.abs(coeffs(lstsq) - np.array([0.05, 0.03])).sum()
        three_err = np.abs(coeffs(three) - np.array([0.05, 0.03])).sum()
        assert lstsq_err < three_err
        npt.assert_almost_equal(coeffs(lstsq), np.array([0.05, 0.03]), 2)

    def test_plane_ransac_raises_when_no_plane(self):
        # a perfectly flat image with an impossible inlier ratio should raise a
        # readable error rather than an UnboundLocalError
        flat = np.full((40, 40), 700.0)
        with self.assertRaises(ValueError):
            plane_ransac(
                flat,
                bg_roi_depth_range=(650, 750),
                iters=25,
                noise_tolerance=0.0,
                in_ratio=0.999999,
            )
