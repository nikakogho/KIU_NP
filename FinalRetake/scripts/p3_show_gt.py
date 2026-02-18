from __future__ import annotations
import argparse
import os
import cv2
import numpy as np

from navigation.p3_io import load_ped_flow_npz, overlay_pedestrians, overlay_mask_edges


def _read_frame_at(cap: cv2.VideoCapture, idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {idx} from video.")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default=None, help="Path to pedestrian flow video")
    ap.add_argument("--gt", type=str, default=None, help="Path to .npz ground truth")
    ap.add_argument("--out", type=str, default="runs/p3_gt_overlay.png")
    ap.add_argument("--synthetic", action="store_true", help="Generate synthetic video+gt first")
    ap.add_argument("--corridor_width", type=int, default=120)
    ap.add_argument("--frames", type=str, default="0,mid,last", help='Indices: "0,mid,last" or "0,100,200"')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    video_path = args.video
    gt_path = args.gt

    if args.synthetic or (video_path is None or gt_path is None):
        from navigation.p3_synth_video import make_synth_corridor_ped_video

        os.makedirs("runs", exist_ok=True)
        gen = make_synth_corridor_ped_video(
            out_video="runs/p3_synth",
            out_gt="runs/p3_synth_gt.npz",
            corridor_width_px=int(args.corridor_width),
            seed=0,
        )
        video_path = gen["video"]
        gt_path = gen["gt"]

    gt = load_ped_flow_npz(gt_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    T = gt.ped_pos.shape[0]

    # parse frame indices
    tokens = [t.strip() for t in args.frames.split(",") if t.strip()]
    idxs = []
    for tok in tokens:
        if tok == "mid":
            idxs.append(T // 2)
        elif tok == "last":
            idxs.append(max(0, T - 1))
        else:
            idxs.append(int(tok))

    idxs = [int(np.clip(i, 0, T - 1)) for i in idxs]
    idxs = list(dict.fromkeys(idxs))  # unique, keep order

    frames = []
    for i in idxs:
        fr = _read_frame_at(cap, i)
        fr = overlay_mask_edges(fr, gt.ped_safe_mask255, edge_bgr=(0, 255, 0))
        fr = overlay_pedestrians(fr, gt.ped_pos[i], ped_radius_px=gt.ped_radius_px)
        cv2.putText(
            fr,
            f"frame={i}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        frames.append(fr)

    cap.release()

    # horizontal montage
    montage = np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]
    cv2.imwrite(args.out, montage)
    print(f"Saved: {args.out}")
    print(f"Video: {video_path}")
    print(f"GT:    {gt_path}")


if __name__ == "__main__":
    main()
