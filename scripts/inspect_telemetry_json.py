import json
import glob
import numpy as np

for fpath in glob.glob("android_app/app/src/main/assets/*_telemetry.json"):
    print("=" * 60)
    print(f"File: {fpath}")
    with open(fpath) as f:
        data = json.load(f)
    print(f"Total frames: {len(data)}")
    blks = [d for d in data if d.get("is_blackout") == 1]
    print(f"Blackout frames: {len(blks)}")
    if blks:
        t_start = blks[0]["t"]
        t_end = blks[-1]["t"]
        print(f"Blackout duration: {t_end - t_start:.1f}s (t={t_start} to {t_end})")
        
        # Check naive error growth
        err_nv = [b["err_naive_m"] for b in blks]
        err_tt = [b["err_truetrack_m"] for b in blks]
        print(f"Naive error: start={err_nv[0]:.1f}m -> max={max(err_nv):.1f}m -> end={err_nv[-1]:.1f}m")
        print(f"TrueTrack error: start={err_tt[0]:.1f}m -> max={max(err_tt):.1f}m -> end={err_tt[-1]:.1f}m")

        # Check coordinate evolution: are naive/truetrack just gt + offset per frame, or cumulative?
        gt_lats = np.array([b["gt"][0] for b in blks])
        gt_lons = np.array([b["gt"][1] for b in blks])
        nv_lats = np.array([b["naive"][0] for b in blks])
        nv_lons = np.array([b["naive"][1] for b in blks])
        tt_lats = np.array([b["truetrack"][0] for b in blks])
        tt_lons = np.array([b["truetrack"][1] for b in blks])

        # Step-to-step displacement
        d_gt = np.hypot(np.diff(gt_lats), np.diff(gt_lons))
        d_nv = np.hypot(np.diff(nv_lats), np.diff(nv_lons))
        d_tt = np.hypot(np.diff(tt_lats), np.diff(tt_lons))
        print(f"Mean step dist (deg): GT={np.mean(d_gt):.6f}, Naive={np.mean(d_nv):.6f}, TT={np.mean(d_tt):.6f}")

        # Check if tt_lats is simply gt_lats + synthetic offset
        dlat_tt_gt = tt_lats - gt_lats
        dlon_tt_gt = tt_lons - gt_lons
        print("Sample TT-GT offsets (first 5):", list(zip(np.round(dlat_tt_gt[:5]*1e5, 2), np.round(dlon_tt_gt[:5]*1e5, 2))))
        print("Sample TT-GT offsets (middle 5):", list(zip(np.round(dlat_tt_gt[100:105]*1e5, 2), np.round(dlon_tt_gt[100:105]*1e5, 2))))
