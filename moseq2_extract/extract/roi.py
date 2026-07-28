"""
ROI detection pre-processing utilities for fitting a plane to an input depth image.
"""

import numpy as np
from tqdm.auto import tqdm


def plane_fit3(points):
    """
    Fit a plane to 3 points (min number of points for fitting a plane)

    Args:
    points (numpy.ndarray): each row is a group of points, columns correspond to x,y,z.

    Returns:
    plane (numpy.array): linear plane fit-->a*x+b*y+c*z+d
    """

    a = points[1] - points[0]
    b = points[2] - points[0]
    # cross prod to make sure the three points make an area, hence a plane.
    normal = np.array(
        [
            [a[1] * b[2] - a[2] * b[1]],
            [a[2] * b[0] - a[0] * b[2]],
            [a[0] * b[1] - a[1] * b[0]],
        ]
    ).astype("float")
    denom = np.sum(np.square(normal)).astype("float")
    if denom < np.spacing(1):
        plane = np.empty((4,))
        plane[:] = np.nan
    else:
        normal /= np.sqrt(denom)
        d = np.dot(-points[0], normal)
        plane = np.hstack((normal.flatten(), d))

    return plane


def plane_fit_lstsq(points):
    """
    Fit a plane to N (>=3) points by total least squares (SVD).

    Unlike ``plane_fit3``, which is defined by exactly three points, this fits
    the plane that minimizes the squared orthogonal distance to every point,
    so the floor reference is defined by all inliers rather than three noisy
    samples.

    Args:
    points (numpy.ndarray): (N, 3) array of x, y, z coordinates.

    Returns:
    plane (numpy.array): linear plane fit --> a*x + b*y + c*z + d, unit normal.
    """

    if points.shape[0] < 3:
        plane = np.empty((4,))
        plane[:] = np.nan
        return plane

    centroid = points.mean(axis=0)
    # smallest right-singular vector of the centered points is the plane normal
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    norm = np.linalg.norm(normal)
    if norm < np.spacing(1):
        plane = np.empty((4,))
        plane[:] = np.nan
        return plane

    normal = normal / norm
    d = -np.dot(normal, centroid)
    return np.hstack((normal, d))


def plane_ransac(
    depth_image,
    bg_roi_depth_range=(650, 750),
    iters=1000,
    noise_tolerance=30,
    in_ratio=0.1,
    progress_bar=False,
    mask=None,
    **kwargs,
):
    """
    Fit a plane using a naive RANSAC implementation

    Args:
    depth_image (numpy.ndarray): background image to fit plane to
    bg_roi_depth_range (tuple): min/max depth (mm) to consider pixels for plane
    iters (int): number of RANSAC iterations
    noise_tolerance (float): distance from plane to consider a point an inlier
    in_ratio (float): fraction of points required to consider a plane fit good
    progress_bar (bool): display progress bar
    mask (numpy.ndarray): boolean mask to find region to use
    kwargs (dict): dictionary containing extra keyword arguments from moseq2_extract.proc.get_roi()

    Returns:
    best_plane (numpy.array): plane fit to data
    dist (numpy.array): distance of the calculated coordinates and "best plane"
    """

    use_points = np.logical_and(
        depth_image > bg_roi_depth_range[0], depth_image < bg_roi_depth_range[1]
    )
    if np.sum(use_points) <= 10:
        raise ValueError(
            f'Too few datapoints exist within given "bg roi depth range" {bg_roi_depth_range} -- data point count: {np.sum(use_points)}.'
            "Please adjust this parameter to fit your recording sessions."
        )

    if mask is not None:
        use_points = np.logical_and(use_points, mask)
        # re-check AFTER masking: the guard above only saw the depth-range
        # filter, so a restrictive mask can still leave too few points. Sampling
        # 3 points without replacement from fewer than 3 raises a bare numpy
        # ValueError, so fail here with something actionable instead.
        if np.sum(use_points) <= 10:
            raise ValueError(
                f"Too few datapoints remain after applying the ROI mask to depth "
                f'range {bg_roi_depth_range} -- data point count: {np.sum(use_points)}. '
                "Check the mask and the depth range for this session."
            )

    xx, yy = np.meshgrid(
        np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0])
    )

    coords = np.vstack(
        (
            xx[use_points].ravel(),
            yy[use_points].ravel(),
            depth_image[use_points].ravel(),
        )
    )
    coords = coords.T

    best_dist = np.inf
    best_num = 0
    best_plane = None

    npoints = np.sum(use_points)

    for _ in tqdm(range(iters), disable=not progress_bar, desc="Finding plane"):

        sel = coords[np.random.choice(coords.shape[0], 3, replace=False)]
        tmp_plane = plane_fit3(sel)

        if np.all(np.isnan(tmp_plane)):
            continue

        dist = np.abs(np.dot(coords, tmp_plane[:3]) + tmp_plane[3])
        inliers = dist < noise_tolerance
        ninliers = np.sum(inliers)

        # score on inlier count; break ties on mean distance of the inliers
        # (not of all points, which would penalize planes that fit the floor
        # tightly while an object sits on it)
        if (ninliers / npoints) > in_ratio:
            mean_inlier_dist = dist[inliers].mean() if ninliers > 0 else np.inf
            if ninliers > best_num or (
                ninliers == best_num and mean_inlier_dist < best_dist
            ):
                best_dist = mean_inlier_dist
                best_num = ninliers
                best_plane = tmp_plane

    if best_plane is None:
        raise ValueError(
            "RANSAC could not find a plane: no candidate exceeded the required "
            f"inlier ratio ({in_ratio}) within depth range {bg_roi_depth_range} "
            f"at noise tolerance {noise_tolerance}. Try widening the depth range "
            "or increasing the noise tolerance."
        )

    # refit the plane to the full set of inliers by least squares, so the floor
    # reference reflects every pixel that agrees with it, not the 3 that seeded it
    inliers = np.abs(np.dot(coords, best_plane[:3]) + best_plane[3]) < noise_tolerance
    refit_plane = plane_fit_lstsq(coords[inliers])
    if not np.any(np.isnan(refit_plane)):
        best_plane = refit_plane

    # fit the plane to our x,y,z coordinates
    coords = np.vstack((xx.ravel(), yy.ravel(), depth_image.ravel())).T
    dist = np.abs(np.dot(coords, best_plane[:3]) + best_plane[3])

    return best_plane, dist
