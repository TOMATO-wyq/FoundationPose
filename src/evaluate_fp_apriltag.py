#!/usr/bin/env python3
"""Offline, source-frame-matched FP vs AprilTag cube evaluation.

No fitted extrinsic: tag is centered on a 0.35 m cube face, edges parallel.
The cube reference axes follow tag axes; original OBJ face identity is unknown.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_pose(path):
    matrix = np.loadtxt(path)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid matrix: {path}")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"Invalid homogeneous row: {path}")
    r = matrix[:3, :3]
    if np.linalg.norm(r.T @ r - np.eye(3)) > .01 or abs(np.linalg.det(r) - 1) > .01:
        raise ValueError(f"Invalid rotation: {path}")
    # TXT 舍入会带来微小非正交误差，投影回 SO(3)，不调整位置或拟合安装变换。
    u, _, vt = np.linalg.svd(r)
    matrix[:3, :3] = u @ vt
    return matrix


def cube_symmetries():
    result = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            r = np.eye(3)[:, permutation] @ np.diag(signs)
            if np.linalg.det(r) > .5:
                result.append(r)
    return np.stack(result)


def angles(rotations):
    return np.rad2deg(np.arccos(np.clip((np.trace(rotations, axis1=-2, axis2=-1)-1)/2, -1, 1)))


def statistics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return dict(n=len(values), mean=float(np.mean(values)), median=float(np.median(values)),
                std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                variance=float(np.var(values, ddof=1)) if len(values) > 1 else None,
                rmse=float(np.sqrt(np.mean(values**2))),
                p95=float(np.percentile(values, 95)), maximum=float(np.max(values)),
                minimum=float(np.min(values)))


def write_rows(path, rows, fields=None):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def process(root, output, kind, transform, symmetries):
    sequence = root / f"d435i_{kind}_sequence"
    fp_dir = root / f"d435i_{kind}_fp/ob_in_cam"
    tag_dir = root / f"d435i_{kind}_tag_075"
    manifest = read_csv(sequence / "frames.csv")
    detections = read_csv(tag_dir / "detections.csv")
    tag_rows = {row["source_frame"]: row for row in detections}
    if len(tag_rows) != len(detections):
        raise ValueError("Duplicate tag source frame")
    if len({row["source_frame"] for row in manifest}) != len(manifest):
        raise ValueError("Duplicate exported source frame")
    calib = json.loads((sequence / "camera_calibration.json").read_text())
    if np.any(np.abs(calib["d"]) > 1e-12):
        raise ValueError("Nonzero distortion: unify FP/tag image processing before evaluation")
    all_frames = read_csv(sequence / "all_frames.csv")
    arrays = dict(time=[], fp=[], reference=[], rotations_fp=[], rotations_ref=[], error=[],
                  angle_raw=[], angle_sym=[], symmetry=[], margin=[], reprojection=[])
    rows = []
    first_stamp = int(manifest[0]["timestamp_ns"])
    for frame in manifest:
        number = frame["source_frame"]
        row = dict(source_frame=int(number), timestamp_ns=frame["timestamp_ns"],
                   time_s=(int(frame["timestamp_ns"])-first_stamp)/1e9,
                   rgb_file=frame["rgb_file"], status="", tag_margin="", tag_reprojection_px="")
        names = [f"{prefix}_{axis}_mm" for prefix in ("fp", "tag_center_ref", "delta") for axis in "xyz"]
        row.update({name: "" for name in names})
        row.update(translation_error_mm="", rotation_raw_deg="", rotation_cube_sym_deg="", symmetry_index="")
        tag = tag_rows.get(number)
        fp_path = fp_dir / Path(frame["rgb_file"]).with_suffix(".txt")
        tag_path = tag_dir / "tag_in_cam" / Path(frame["rgb_file"]).with_suffix(".txt")
        if tag is None:
            row["status"] = "tag_row_missing"
        elif tag["timestamp_ns"] != frame["timestamp_ns"] or tag["rgb_file"] != frame["rgb_file"]:
            raise ValueError(f"Tag/frame stamp mismatch at {kind}:{number}")
        elif not fp_path.exists():
            row["status"] = "fp_pose_missing"
        elif tag["status"] != "detected":
            row["status"] = "tag_not_detected"
        elif not tag_path.exists():
            row["status"] = "tag_pose_missing"
        else:
            fp = read_pose(fp_path)
            marker = read_pose(tag_path)
            if int(tag["id"]) != 0 or not np.allclose(marker[:3, 3], [float(tag[f"t{axis}_m"]) for axis in "xyz"], atol=1e-8):
                raise ValueError("Tag pose text disagrees with detection CSV")
            if marker[:3, 2] @ marker[:3, 3] >= 0:
                raise ValueError("Tag normal does not face camera; check axis convention")
            reference = marker @ transform
            delta = (fp[:3, 3] - reference[:3, 3]) * 1000
            relative = reference[:3, :3].T @ fp[:3, :3]
            candidates = angles(np.swapaxes(symmetries, 1, 2) @ relative)
            chosen = int(np.argmin(candidates))
            raw = float(angles(relative))
            row.update(status="valid", tag_margin=float(tag["decision_margin"]),
                       tag_reprojection_px=float(tag["reprojection_error_px"]),
                       translation_error_mm=float(np.linalg.norm(delta)), rotation_raw_deg=raw,
                       rotation_cube_sym_deg=float(candidates[chosen]), symmetry_index=chosen)
            for values, prefix in ((fp[:3, 3]*1000, "fp"), (reference[:3, 3]*1000, "tag_center_ref"), (delta, "delta")):
                for axis, value in zip("xyz", values):
                    row[f"{prefix}_{axis}_mm"] = float(value)
            for name, value in dict(time=row["time_s"], fp=fp[:3, 3]*1000,
                                    reference=reference[:3, 3]*1000,
                                    rotations_fp=fp[:3, :3], rotations_ref=reference[:3, :3],
                                    error=delta, angle_raw=raw, angle_sym=candidates[chosen],
                                    symmetry=chosen, margin=row["tag_margin"],
                                    reprojection=row["tag_reprojection_px"]).items():
                arrays[name].append(value)
        rows.append(row)
    if not arrays["time"]:
        raise ValueError(f"No valid matched data: {kind}")
    a = {key: np.asarray(value) for key, value in arrays.items()}
    norm = np.linalg.norm(a["error"], axis=1)
    metrics = {"translation_norm_mm": statistics(norm), "rotation_cube_sym_deg": statistics(a["angle_sym"]),
               "rotation_raw_canonical_deg": statistics(a["angle_raw"]),
               "tag_reprojection_px": statistics(a["reprojection"]), "tag_margin": statistics(a["margin"])}
    for index, axis in enumerate("xyz"):
        metrics[f"delta_{axis}_mm"] = statistics(a["error"][:, index])
    bias = a["error"].mean(axis=0)
    centered_rms = float(np.sqrt(np.mean(np.sum((a["error"]-bias)**2, axis=1))))
    summary = dict(dataset=kind, raw_rgb_frames=len(all_frames), exported_rgbd_frames=len(manifest),
                   fp_pose_files=len(list(fp_dir.glob("*.txt"))), tag_detected_rows=sum(r["status"]=="detected" for r in detections),
                   valid_pairs=len(a["time"]), valid_pair_rate_exported=len(a["time"])/len(manifest),
                   duration_seconds=(int(manifest[-1]["timestamp_ns"])-first_stamp)/1e9,
                   position_bias_xyz_mm=bias.tolist(), centered_difference_rms_mm=centered_rms,
                   metrics=metrics,
                   symmetry_assignment_changes=int(np.count_nonzero(np.diff(a["symmetry"]))),
                   note="Cube symmetry assumes geometry AND texture invariance; raw rotation uses arbitrary canonical tag-aligned cube axes.")
    if kind == "static":
        jitter = {}
        for label, positions, rotations in (("fp", a["fp"], a["rotations_fp"]), ("tag_reference", a["reference"], a["rotations_ref"])):
            mean_rotation = Rotation.from_matrix(rotations).mean().as_matrix()
            deviation = angles(mean_rotation.T @ rotations)
            jitter[label] = dict(position_mean_xyz_mm=positions.mean(0).tolist(),
                                  position_std_xyz_mm=positions.std(0, ddof=1).tolist(),
                                  position_variance_xyz_mm2=positions.var(0, ddof=1).tolist(),
                                  position_rms_about_mean_mm=float(np.sqrt(np.mean(np.sum((positions-positions.mean(0))**2, axis=1)))),
                                  orientation_rms_about_mean_deg=float(np.sqrt(np.mean(deviation**2))))
        summary["full_recording_spread_includes_motion"] = jitter
        initial = a["time"] <= 5.0
        stability = {}
        for label, positions, rotations in (("fp", a["fp"][initial], a["rotations_fp"][initial]),
                                           ("tag_reference", a["reference"][initial], a["rotations_ref"][initial])):
            mean_rotation = Rotation.from_matrix(rotations).mean().as_matrix()
            deviation = angles(mean_rotation.T @ rotations)
            stability[label] = dict(n=len(positions), position_std_xyz_mm=positions.std(0, ddof=1).tolist(),
                                    position_variance_xyz_mm2=positions.var(0, ddof=1).tolist(),
                                    position_range_xyz_mm=np.ptp(positions, axis=0).tolist(),
                                    position_rms_about_mean_mm=float(np.sqrt(np.mean(np.sum((positions-positions.mean(0))**2, axis=1)))),
                                    orientation_rms_about_mean_deg=float(np.sqrt(np.mean(deviation**2))))
        summary["initial_0_to_5_seconds_stability_exploratory"] = stability
        summary["static_recording_is_not_stationary"] = True
    target = output / kind
    target.mkdir()
    write_rows(target / "per_frame.csv", rows)
    write_rows(target / "statistics.csv", [dict(metric=name, **value) for name, value in metrics.items()])
    (target / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    timeline = np.array([row["time_s"] for row in rows])
    def series(field):
        return np.array([float(row[field]) if row[field] != "" else np.nan for row in rows])
    # 缺失结果保留 NaN 断点，不插值、也不把缺失当成零误差。
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for i, axis in enumerate("xyz"):
        axes[i].plot(timeline, series(f"fp_{axis}_mm"), label="FP", color="#2563a6", lw=1)
        axes[i].plot(timeline, series(f"tag_center_ref_{axis}_mm"), label="Tag-derived center", color="#b97722", lw=1, ls="--")
        axes[i].set_ylabel(f"{axis.upper()} (mm)")
        axes[i].legend(loc="best")
    axes[0].set_title(f"{kind.title()}: cube center in camera optical frame")
    axes[-1].set_xlabel("Source time since first RGB-D frame (s)")
    fig.tight_layout()
    save_plot(fig, target / "01_position_xyz")
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    for axis, color in zip("xyz", ["#2563a6", "#b97722", "#874f8e"]):
        axes[0].plot(timeline, series(f"delta_{axis}_mm"), label=f"{axis.upper()} difference", color=color, lw=1)
    axes[0].axhline(0, color="#555555", lw=.6)
    axes[0].set_ylabel("FP minus reference (mm)")
    axes[0].legend(loc="best")
    axes[1].plot(timeline, series("translation_error_mm"), color="#2563a6", lw=1)
    axes[1].axhline(metrics["translation_norm_mm"]["p95"], label="P95", color="#b97722", ls="--")
    axes[1].set_ylabel("Center distance (mm)")
    axes[1].legend()
    axes[2].plot(timeline, series("rotation_cube_sym_deg"), color="#2563a6", label="Cube symmetry adjusted", lw=1)
    axes[3].plot(timeline, series("rotation_raw_deg"), color="#b97722", label="Raw canonical: axis choice dependent, not absolute error", lw=.8)
    axes[3].set_ylabel("Raw rotation (deg)")
    axes[3].legend()
    axes[2].set_ylabel("Rotation difference (deg)")
    axes[2].legend()
    axes[0].set_title(f"{kind.title()}: FP vs tag-derived reference; no fitted extrinsic")
    axes[-1].set_xlabel("Source time (s)")
    fig.tight_layout()
    save_plot(fig, target / "02_errors")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].hist(norm, bins=40, color="#2563a6")
    axes[0, 0].set_xlabel("Center distance (mm)")
    axes[0, 1].hist(a["angle_sym"], bins=40, color="#2563a6")
    axes[0, 1].set_xlabel("Symmetry-adjusted rotation (deg)")
    axes[1, 0].plot(a["time"], a["reprojection"], color="#2563a6", lw=.8)
    axes[1, 0].set_xlabel("Source time (s)")
    axes[1, 0].set_ylabel("Tag corner reprojection RMSE (px)")
    axes[1, 1].scatter(a["reprojection"], norm, s=5, color="#2563a6", alpha=.4)
    axes[1, 1].set_xlabel("Tag reprojection RMSE (px)")
    axes[1, 1].set_ylabel("Center distance (mm)")
    for ax in axes[0]:
        ax.set_ylabel("Valid matched frames")
    fig.suptitle(f"{kind.title()}: distributions and tag quality (all valid pairs)")
    fig.tight_layout()
    save_plot(fig, target / "03_distributions_quality")
    missing = [row for row in rows if row["status"] != "valid"]
    write_rows(target / "missing_frames.csv", missing, list(rows[0]))
    largest = sorted((row for row in rows if row["status"] == "valid"), key=lambda row: row["translation_error_mm"], reverse=True)[:20]
    write_rows(target / "largest_20_differences.csv", largest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/tomato/6Dpose/outputs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new directory")
    vertices = np.array([[float(v) for v in line.split()[1:4]] for line in
                         (args.root.parent / "kfs_model/BlueTrueKFS13/BlueTrueKFS13.obj").open() if line.startswith("v ")])
    if not np.allclose(np.ptp(vertices, axis=0), .35, atol=1e-6) or not np.allclose((vertices.min(0)+vertices.max(0))/2, 0, atol=1e-6):
        raise ValueError("Model is not a centered 0.35 m cube")
    args.output.mkdir(parents=True)
    transform = np.eye(4)
    transform[2, 3] = -.175
    # 标签 +Z 向可见面外侧；中心位于其后方 175mm。轴向采用 tag 对齐的规范立方体。
    config = dict(cube_side_m=.35, tag_side_m=.075, tag_centered=True, edges_parallel=True,
                  T_tag_from_canonical_cube=transform.tolist(),
                  orientation="canonical cube axes follow tag axes; original OBJ face is unknown",
                  not_fitted=True, mounting_thickness_m=0,
                  caveats=["Tag is a reference measurement, not ground truth", "No mounting uncertainty calibrated", "24 cube rotations apply only if texture is also symmetric"])
    (args.output / "transform_config.json").write_text(json.dumps(config, indent=2))
    np.savetxt(args.output / "T_tag_from_canonical_cube.txt", transform, fmt="%.9g")
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2, "axes.spines.top": False, "axes.spines.right": False})
    symmetries = cube_symmetries()
    assert len(symmetries) == 24
    np.save(args.output / "cube_symmetries.npy", symmetries)
    summaries = [process(args.root, args.output, kind, transform, symmetries) for kind in ("static", "dynamic")]
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    combined = []
    for summary in summaries:
        for name, values in summary["metrics"].items():
            combined.append(dict(dataset=summary["dataset"], metric=name, **values))
    write_rows(args.output / "summary_statistics.csv", combined)
    report = ["# FP 与 AprilTag 离线对照报告", "", "## 变换与评估范围", "",
              "OBJ 检查：原点在中心，边长 0.35m。用户确认标签黑边 0.075m，贴面中心、边平行。",
              "相机下的箱体中心参考：`p_ref = p_tag - 0.175 * R_tag[:,2]`；",
              "完整变换：`T_camera_cube_ref = T_camera_tag @ T_tag_from_canonical_cube`。",
              "安装变换依据几何确定，没有根据 FP 拟合位置或旋转。纸/胶厚度假定为零。",
              "标签到具体 OBJ 面的朝向未独立标定，因此原始旋转差异依赖轴向约定，不能当作严格绝对精度。",
              "对称旋转差异在 24 个正方体几何旋转中取最小值；若纹理不满足全部对称，应限制对称集合。",
              "参考来自同一相机的 AprilTag PnP，不是无误差真值。数值是两估计器的差异，包含标签、安装和 FP 误差。",
              "", "## 总览", "", "|数据|有效匹配/导出帧|中心差均值 mm|RMSE mm|样本标准差 mm|P95 mm|最大 mm|对称角均值 °|角 P95 °|",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        t, r = s["metrics"]["translation_norm_mm"], s["metrics"]["rotation_cube_sym_deg"]
        report.append(f"|{s['dataset']}|{s['valid_pairs']}/{s['exported_rgbd_frames']}|{t['mean']:.3f}|{t['rmse']:.3f}|{t['std']:.3f}|{t['p95']:.3f}|{t['maximum']:.3f}|{r['mean']:.3f}|{r['p95']:.3f}|")
    report.extend(["", "样本标准差和方差使用 ddof=1；位置分量方差单位 mm²，角方差单位 deg²。",
                   "RMSE 为 sqrt(mean(error²))；位置距离为三维欧氏范数，不是单轴误差。",
                   "中心差均值不等于固定偏差向量；XYZ 偏差见下表。没有剔除异常值、平滑、插值或置信区间拟合。",
                   "相邻视频帧相关，样本数不代表同样数量的独立试验。缺失帧不算零误差，也不移位配对。",
                   "", "## XYZ 平均差（FP − 标签换算中心）", "", "|数据|X mm|Y mm|Z mm|去均值后差异 RMS mm|", "|---|---:|---:|---:|---:|"])
    for s in summaries:
        bias = s["position_bias_xyz_mm"]
        report.append(f"|{s['dataset']}|{bias[0]:.3f}|{bias[1]:.3f}|{bias[2]:.3f}|{s['centered_difference_rms_mm']:.3f}|")
    report.extend(["", "## 初始 0–5 秒稳定性参考（探索性窗口）", "", "|方法|X 标准差 mm|Y 标准差 mm|Z 标准差 mm|中心相对均值 RMS mm|旋转相对平均姿态 RMS °|", "|---|---:|---:|---:|---:|---:|"])
    for method, item in summaries[0]["initial_0_to_5_seconds_stability_exploratory"].items():
        std = item["position_std_xyz_mm"]
        report.append(f"|{method}|{std[0]:.3f}|{std[1]:.3f}|{std[2]:.3f}|{item['position_rms_about_mean_mm']:.3f}|{item['orientation_rms_about_mean_deg']:.3f}|")
    report.extend(["", "用户确认 static 录像包含主动翻面，本文按翻面测试解释：整段坐标标准差包含真实运动，不作为静止抖动。目录名 static 保留以对应源数据。",
                   "这里只展示初始 0–5 秒的探索性稳定性窗口，未独立确认完全静止，不是经过校准的噪声精度；整段离散度仍保存在 summary.json。",
                   "", "## 数据检查", "",
                   "逐帧核对原始 source_frame、纳秒时间戳和文件名；TXT 位姿有限性、旋转正交性、齐次末行及 tag CSV/TXT 平移一致性通过。",
                   "两组 CameraInfo 畸变系数均为零，当前两条输入链路没有不同畸变处理。未识别及缺失结果单独列出。",
                   "FP 没有单独记录源时间戳，依据同名 RGB 文件与其运行输入目录对齐；这不是墙钟时间的配对。",
                   "", "## 图表与逐帧数据", ""])
    for s in summaries:
        kind = s["dataset"]
        report.extend([f"### {kind}", "", f"原始 RGB {s['raw_rgb_frames']}，导出完整 RGB-D {s['exported_rgbd_frames']}，有效双结果 {s['valid_pairs']}。",
                       f"有效比例分母是导出 RGB-D 帧；不包括导出阶段缺深度/内参的 RGB。",
                       f"[逐帧 CSV]({kind}/per_frame.csv) · [统计]({kind}/statistics.csv) · [缺失帧]({kind}/missing_frames.csv) · [最大20差异]({kind}/largest_20_differences.csv)", "",
                       f"![中心轨迹]({kind}/01_position_xyz.png)", "", f"![差异曲线]({kind}/02_errors.png)", "", f"![分布与质量]({kind}/03_distributions_quality.png)", ""])
    report.extend(["## 使用限制与复现", "", "动态最大中心差异出现在原始帧 1829、源时间约 61.08 秒：221.87 mm，对称旋转差异约 52.21°，需回看该帧，不直接归责任一估计器。",
                   "这里只检验已保存位姿与测量的一致性，不根据误差大小自动判断 FP 跟丢。",
                   "标签外观会改变物体纹理；遮挡、斜视及单平面 PnP 也会产生参考误差。重投影误差低不保证绝对位置准确。",
                   "若需要严格 OBJ 轴向误差，需独立确定标签到具体模型面的旋转；本次不能通过 FP 结果反求再评估同一数据。",
                   "运行：`python3 ~/6Dpose/src/evaluate_fp_apriltag.py --output /新的输出目录`。",
                   "所有图同时保存 PNG 与 PDF；CSV 使用 UTF-8 BOM，可在表格软件打开。"])
    (args.output / "REPORT.zh-CN.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps([{k: s[k] for k in ("dataset", "valid_pairs", "position_bias_xyz_mm", "metrics")} for s in summaries], indent=2))


if __name__ == "__main__":
    main()
