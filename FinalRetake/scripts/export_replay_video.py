from __future__ import annotations
import argparse
import os

from navigation.replay_render import export_video_from_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True, help="Replay .npz (from problem scripts)")
    ap.add_argument("--out", type=str, required=True, help="Output video path (mp4/avi; ext optional)")
    ap.add_argument("--fps", type=int, default=None, help="Override fps (default uses npz fps if present)")
    ap.add_argument("--no_trail", action="store_true", help="Disable trail drawing")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    out_path = export_video_from_npz(
        args.npz,
        out_path=args.out,
        fps=args.fps,
        trail=(not args.no_trail),
    )
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
