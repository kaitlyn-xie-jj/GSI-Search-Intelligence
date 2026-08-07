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
    args = parser.parse_args()
    frame_bytes = args.width * args.height * 3
    while True:
        data = sys.stdin.buffer.read(frame_bytes)
        if len(data) != frame_bytes:
            break
        rgb = np.frombuffer(data, dtype=np.uint8).reshape(
            args.height, args.width, 3
        )
        cv2.imshow(args.title, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
