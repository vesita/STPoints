"""PnP extrinsic calibration via IPPE solver.

Given 4 coplanar 3D-2D correspondences (front face of a 3D box),
solves LiDAR→Camera extrinsic using cv2.SOLVEPNP_IPPE (closed-form,
no initial value needed).
"""

import cv2
import numpy as np


def solve_pnp_ippe(points_3d, points_2d, camera_matrix, dist_coeffs=None):
    """Solve PnP with IPPE for 4 coplanar 3D–2D point pairs.

    Args:
        points_3d: list[list[float]] — 4 points in LiDAR/world coords [[x,y,z], …]
        points_2d: list[list[float]] — 4 corresponding image pixels    [[u,v], …]
        camera_matrix: list[float] — 9 floats, 3×3 intrinsic (row-major)
        dist_coeffs: list[float] | None — distortion coeffs or None

    Returns:
        dict with keys:
            rvec, tvec, extrinsic (4×4 row-major homogeneous),
            reprojection_error (mean pixel error), success (bool)
    """
    pts_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    pts_2d = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)

    # 预检：检查点是否过于紧凑（所有点间距太小导致 IPPE 退化）
    center = pts_3d.mean(axis=0)
    centered = pts_3d - center
    max_dist = np.sqrt(np.max(np.sum(centered ** 2, axis=1)))
    if max_dist < 0.05:
        return {"success": False, "error": f"3D 点过于集中 (最大间距 {max_dist:.3f}m)，无法求解 PnP"}

    # 预检：检查 2D 点是否有效
    img_center = pts_2d.mean(axis=0)
    img_radius = np.sqrt(np.max(np.sum((pts_2d - img_center) ** 2, axis=1)))
    if img_radius < 5.0:
        return {"success": False, "error": f"2D 点过于集中 (半径 {img_radius:.1f}px)，无法求解 PnP"}

    if dist_coeffs is not None and len(dist_coeffs) > 0:
        dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    else:
        dist = np.zeros((4, 1), dtype=np.float64)

    try:
        retval, rvec, tvec = cv2.solvePnP(
            pts_3d, pts_2d, K, dist, flags=cv2.SOLVEPNP_IPPE
        )
    except cv2.error as e:
        print(f"[solve_pnp_ippe] OpenCV error: {e}")
        return {
            "success": False,
            "error": f"IPPE 求解失败: {str(e)}",
        }
    print(f"[solve_pnp_ippe] pts_3d=\n{pts_3d}")
    print(f"[solve_pnp_ippe] pts_2d=\n{pts_2d}")
    print(f"[solve_pnp_ippe] rvec={rvec.flatten()}, tvec={tvec.flatten()}")

    # 4×4 homogeneous extrinsic (row-major): P_cam = [R|t] · P_world
    R_mat, _ = cv2.Rodrigues(rvec)
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = R_mat
    extrinsic[:3, 3] = tvec.flatten()

    # Reprojection error — mean pixel distance
    proj_pts, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist)
    proj_flat = proj_pts.reshape(-1, 2)
    errors = np.sqrt(np.sum((proj_flat - pts_2d) ** 2, axis=1))
    mean_error = float(np.mean(errors))

    print(f"[solve_pnp_ippe] reprojected=\n{proj_flat}")
    print(f"[solve_pnp_ippe] per-point errors={errors.flatten()}")
    print(f"[solve_pnp_ippe] mean_error={mean_error:.2f}px")

    return {
        "rvec": rvec.flatten().tolist(),
        "tvec": tvec.flatten().tolist(),
        "extrinsic": extrinsic.flatten().tolist(),
        "reprojected_points": proj_flat.tolist(),
        "per_point_errors": errors.flatten().tolist(),
        "reprojection_error": mean_error,
        "success": bool(retval),
    }
