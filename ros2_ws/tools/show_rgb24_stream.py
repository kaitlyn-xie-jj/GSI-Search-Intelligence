#!/usr/bin/env python3
"""Display a fixed-size RGB24 stream received on stdin."""

import argparse
import sys

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--title", default="GSI UAV camera")
    parser.add_argument("--scale", type=int, default=1)
    args = parser.parse_args()
    if args.scale < 1:
        parser.error("--scale must be at least 1")
    frame_bytes = args.width * args.height * 3
    while True:
        data = sys.stdin.buffer.read(frame_bytes)
        if len(data) != frame_bytes:
            break
        rgb = np.frombuffer(data, dtype=np.uint8).reshape(
            args.height, args.width, 3
        )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if args.scale > 1:
            bgr = cv2.resize(
                bgr,
                (args.width * args.scale, args.height * args.scale),
                interpolation=cv2.INTER_NEAREST,
            )
        cv2.imshow(args.title, bgr)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
