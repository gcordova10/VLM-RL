import os
import cv2
import argparse
import numpy as np

def concatenate_videos(input_paths, output_path):
    if not input_paths:
        print("Error: No input videos provided.")
        return

    caps = [cv2.VideoCapture(p) for p in input_paths]
    valid_caps = [cap for cap in caps if cap.isOpened()]
    
    if len(valid_caps) != len(caps):
        print("Error: Some videos could not be opened.")
        return

    # Use first video properties
    fps = valid_caps[0].get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 15.0
    width = int(valid_caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(valid_caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use high quality MJPG codec
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Concatenating {len(input_paths)} videos into {output_path}...")
    print(f"Resolution: {width}x{height} @ {fps} FPS")

    for i, cap in enumerate(valid_caps):
        print(f"Processing video {i+1}/{len(input_paths)}: {input_paths[i]}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Ensure frame size matches (just in case)
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            out.write(frame)
        cap.release()

    out.release()
    print(f"Success! Concatenated video saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concatenate multiple videos into one.')
    parser.add_argument('inputs', nargs='+', help='List of input video paths')
    parser.add_argument('--output', required=True, help='Output video path')

    args = parser.parse_args()
    concatenate_videos(args.inputs, args.output)
