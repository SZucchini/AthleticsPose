import itertools

import cv2
import numpy as np
import scipy
from pycalib.calib import triangulate

from athleticspose.multiview.triangulate import triangulate_with_conf
from athleticspose.statics.bones import h36m_bones


def joints2orientations(p3d_CxNxJx3, mask_vis_NxJ, bones_Jx2):
    C = p3d_CxNxJx3.shape[0]
    N = p3d_CxNxJx3.shape[1]
    B = bones_Jx2.shape[0]

    p3d_CxNxJx3 = np.copy(p3d_CxNxJx3)
    p3d_CxNxJx3[:, ~mask_vis_NxJ, :] = np.nan
    pairs = p3d_CxNxJx3[:, :, bones_Jx2, :]

    dirs = pairs[:, :, :, 1, :] - pairs[:, :, :, 0, :]
    dirs = dirs.reshape((C, N * B, 3))
    mask = np.min(~np.isnan(dirs), axis=(0, 2))
    dirs = dirs[:, mask, :]
    norm = np.linalg.norm(dirs, axis=2)
    dirs = dirs / norm[:, :, None]
    return dirs


def joints2projections(p2d_CxNxJx2, mask_vis_NxJ):
    C = p2d_CxNxJx2.shape[0]
    p2d_CxNxJx2[:, ~mask_vis_NxJ, :] = np.nan

    p2d = p2d_CxNxJx2.reshape((C, -1, 2))
    idx = np.isnan(p2d).any(axis=(0, 2))
    p2d = p2d[:, ~idx, :]
    return p2d


def visible_from_all_cam(mask_CxNxJ):
    mask = np.min(mask_CxNxJ, axis=0)
    return mask


def collinearity_w2c(R_w2c, n, idx_v, idx_t, num_v, num_t):
    nmat = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    t0 = num_v * 3
    A = scipy.sparse.lil_matrix((3, (num_v + num_t) * 3), dtype=np.float64)
    A[:, idx_v * 3 : idx_v * 3 + 3] = nmat @ R_w2c
    A[:, t0 + idx_t * 3 : t0 + idx_t * 3 + 3] = nmat
    return A


def coplanarity_w2c(Ra, Rb, na, nb, idx_t1, idx_t2, num_t):
    rows = na.shape[0]
    assert na.shape[1] == 3
    assert nb.shape[0] == rows
    assert nb.shape[1] == 3

    m = np.cross(na @ Ra, nb @ Rb)
    A = scipy.sparse.lil_matrix((rows, num_t * 3), dtype=np.float64)
    A[:, idx_t1 * 3 : idx_t1 * 3 + 3] = m @ Ra.T
    A[:, idx_t2 * 3 : idx_t2 * 3 + 3] = -m @ Rb.T
    return A


def z_test_w2c(R1, t1, R2, t2, n1, n2):
    def triangulate(R1, t1, R2, t2, n1, n2):
        Xh = cv2.triangulatePoints(
            np.hstack([R1, t1[:, None]]),
            np.hstack([R2, t2[:, None]]),
            n1[:, :2].T,
            n2[:, :2].T,
        )
        Xh /= Xh[3, :]
        return Xh[:3, :].T

    def z_count(R, t, Xw_Nx3):
        X = R @ Xw_Nx3.T + t.reshape((3, 1))
        return np.sum(X[2, :] > 0)

    Xp = triangulate(R1, t1, R2, t2, n1, n2)
    Xn = triangulate(R1, -t1, R2, -t2, n1, n2)
    zp = z_count(R1, t1, Xp) + z_count(R2, t2, Xp)
    zn = z_count(R1, t1, Xn) + z_count(R2, t2, Xn)
    return 1 if zp > zn else -1, zp, zn


def calib_linear(v_CxNx3, n_CxMx3):
    C = v_CxNx3.shape[0]
    M = n_CxMx3.shape[1]
    v_Nx3C = np.hstack(v_CxNx3)

    Y, D, Zt = np.linalg.svd(v_Nx3C)
    R_all = np.sqrt(C) * Zt[:3, :]
    Rx = np.linalg.inv(R_all[:3, :3])
    R_all = Rx @ R_all
    R_w2c_list = R_all.T.reshape((-1, 3, 3))

    A = []
    for idx_t, (R, n) in enumerate(zip(R_w2c_list, n_CxMx3, strict=False)):
        for idx_v in range(n.shape[0]):
            A.append(collinearity_w2c(R, n[idx_v, :], idx_v, idx_t, M, C))
    A = scipy.sparse.vstack(A)

    B = []
    for (a, Ra, na), (b, Rb, nb) in itertools.combinations(zip(range(C), R_w2c_list, n_CxMx3, strict=False), 2):
        B.append(coplanarity_w2c(Ra, Rb, na, nb, a, b, C))

    B = scipy.sparse.vstack(B)
    C = scipy.sparse.lil_matrix((A.shape[0] + B.shape[0], A.shape[1]), dtype=np.float64)
    C[: A.shape[0]] = A
    C[A.shape[0] :, -B.shape[1] :] = B

    w, v = scipy.linalg.eigh((C.T @ C).toarray(), subset_by_index=(0, 5), overwrite_a=True, overwrite_b=True)
    k = v[:, :4]

    _, s, vt = np.linalg.svd(k[-B.shape[1] : -B.shape[1] + 3, :])
    t = k @ vt[3, :].T
    X = t[: -B.shape[1]].reshape((-1, 3))
    t = t[-B.shape[1] :]
    s = np.linalg.norm(t[3:6])
    t = t / s
    X = X / s
    t_w2c_list = t.reshape((-1, 3))

    R1 = R_w2c_list[0]
    R2 = R_w2c_list[1]
    t1 = t_w2c_list[0]
    t2 = t_w2c_list[1]
    n1 = n_CxMx3[0]
    n2 = n_CxMx3[1]
    sign, Np, Nn = z_test_w2c(R1, t1, R2, t2, n1, n2)

    t_w2c_list = sign * t_w2c_list
    X = sign * X

    return R_w2c_list, t_w2c_list.reshape((-1, 3, 1)), X


def calib_linear_ransac(v_CxNx3, n_CxMx3, K, n_iter, th_2d, th_3d, seed):
    C = v_CxNx3.shape[0]
    N = v_CxNx3.shape[1]
    M = n_CxMx3.shape[1]
    assert n_CxMx3.shape[0] == C

    E_best = np.inf
    e_best = []
    e3d_best = []

    rng = np.random.default_rng(seed)
    for _ in range(n_iter):
        n = rng.choice(N, size=10, replace=False)
        m = rng.choice(M, size=10, replace=False)
        R, t, X = calib_linear(v_CxNx3[:, n, :], n_CxMx3[:, m, :])

        P = []
        for i in range(C):
            P.append(np.hstack((R[i], t[i].reshape((3, 1)))))
        P = np.array(P)

        e = []
        for i in range(M):
            x = n_CxMx3[:, i, :].reshape((C, 3))[:, :2]
            X = triangulate(x, P)
            for c in range(C):
                x2 = P[c] @ X.reshape((4, 1))
                x2 = x2 / x2[2]
                e.append(x2.flatten() - n_CxMx3[c, i, :].flatten())

        e3d = []
        for i in range(N):
            e3 = 0
            for c in range(1, C):
                v1 = v_CxNx3[0, i] @ R[0]
                v2 = v_CxNx3[c, i] @ R[c]
                e3 += np.dot(v1, v2)
            e3 = np.clip(e3 / (C - 1), -1, 1)
            e3d.append(e3)
        e3d = np.arccos(e3d)

        e_sum = np.linalg.norm(e) / M
        e3d_sum = np.linalg.norm(e3d) / N
        if e_sum + e3d_sum < E_best:
            E_best = e_sum + e3d_sum
            e_best = e
            e3d_best = e3d

    e_best = np.array(e_best).reshape((M, C, 3))
    mask_2d = np.linalg.norm(e_best[:, :, :2], axis=2)
    mask_2d = mask_2d.T

    for i in range(C):
        mask_2d[i, :] = K[i, 0, 0] * mask_2d[i, :]
    mask_2d = mask_2d <= th_2d
    mask_2d = np.min(mask_2d, axis=0)

    e3d_best = np.array(e3d_best).reshape((N))
    mask_3d = e3d_best <= th_3d

    return calib_linear(v_CxNx3[:, mask_3d, :], n_CxMx3[:, mask_2d, :])


def calibrate(
    kpts2d: np.ndarray,
    score2d: np.ndarray,
    kpts3d: np.ndarray,
    score3d: np.ndarray,
    intrinsics: np.ndarray,
    ransac: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate camera parameters using 2D keypoints and 3D keypoints with scores.

    Args:
        kpts2d: 2D keypoints with shape (C, T, J, 2) where C is number of cameras,
        score2d: Confidence scores for 2D keypoints with shape (C, T, J).
        kpts3d: 3D keypoints with shape (C, T, J, 3) where C is number of cameras,
        score3d: Confidence scores for 3D keypoints with shape (C, T, J).
        intrinsics: Camera intrinsic parameters with shape (C, 3, 3).
        ransac: Whether to use RANSAC for robust estimation. Default is False.

    Returns:
        tuple containing:
            - rot_est: Estimated rotation matrices with shape (C, 3, 3)
            - t_est: Estimated translation vectors with shape (C, 3, 1)
            - triangulated_kpts3d: Triangulated 3D keypoints with shape (T, J, 3)

    """
    mask_cxnxj = (score2d > 0) * (score3d == 1)
    mask_vis_nxj = visible_from_all_cam(mask_cxnxj)
    vc = joints2orientations(kpts3d, mask_vis_nxj, np.array(h36m_bones))
    y = joints2projections(kpts2d, mask_vis_nxj)

    n = np.ones((y.shape[0], y.shape[1], 3), dtype=np.float64)
    n[:, :, :2] = y
    ni = []
    for i in range(len(intrinsics)):
        ni.append(n[i] @ np.linalg.inv(intrinsics[i]).T)
    n = np.array(ni)

    if ransac:
        rot_est, t_est, _ = calib_linear_ransac(vc, n, intrinsics, 555, 10, 1, 42)
    else:
        rot_est, t_est, _ = calib_linear(vc, n)

    triangulated_kpts3d = triangulate_with_conf(kpts2d, score2d, intrinsics, rot_est, t_est, (score2d > 0))
    return rot_est, t_est, triangulated_kpts3d
