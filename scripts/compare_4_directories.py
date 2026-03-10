import os
import cv2
import argparse
import numpy as np
from datetime import datetime
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def time_to_seconds(time_str):
    if not time_str:
        return None
    try:
        t = datetime.strptime(time_str, "%H:%M:%S")
        return t.hour * 3600 + t.minute * 60 + t.second
    except ValueError:
        try:
            t = datetime.strptime(time_str, "%M:%S")
            return t.minute * 60 + t.second
        except ValueError:
            return float(time_str)

def process_video_segment(caps, labels, target_w, target_h, fps, start_frame, end_frame, out):
    # Seek to start
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    while True:
        if end_frame is not None and frame_idx >= end_frame:
            break

        frames = []
        all_finished = True
        for cap in caps:
            ret, frame = cap.read()
            if ret:
                all_finished = False
                if frame.shape[0] != target_h or frame.shape[1] != target_w:
                    frame = cv2.resize(frame, (target_w, target_h))
            else:
                frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            frames.append(frame)

        if all_finished:
            break

        # Add labels
        for i, (frame, label) in enumerate(zip(frames, labels)):
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cv2.rectangle(frame, (5, 5), (15 + text_size[0], 15 + text_size[1]), (0, 0, 0), -1)
            cv2.putText(frame, label, (10, 10 + text_size[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Create grid
        top_row = np.hstack((frames[0], frames[1]))
        bottom_row = np.hstack((frames[2], frames[3]))
        combined = np.vstack((top_row, bottom_row))

        out.write(combined)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}...")

def main():
    parser = argparse.ArgumentParser(description='Merge 4 directories of videos into a single grid video sequence.')
    parser.add_argument('--dirs', nargs=4, required=True, help='4 Directories to compare')
    parser.add_argument('--labels', nargs=4, help='Optional 4 labels for the quadrants')
    parser.add_argument('--output', default='clips/grid_sequence_comparison.avi', help='Output path')
    parser.add_argument('--start', default="00:00:00", help='Start time in each video')
    parser.add_argument('--duration', type=float, default=10.0, help='Duration of each segment in seconds')
    parser.add_argument('--limit', type=int, default=15, help='Limit to first N common videos')

    args = parser.parse_args()
    
    # Get common files
    dir_files = []
    for d in args.dirs:
        files = set([f for f in os.listdir(d) if f.endswith('.avi')])
        dir_files.append(files)
    
    common_files = sorted(list(set.intersection(*dir_files)), key=natural_sort_key)
    if not common_files:
        print("No common .avi files found in all 4 directories.")
        return

    print(f"Found {len(common_files)} common files. Processing first {min(args.limit, len(common_files))}:")
    for f in common_files[:args.limit]:
        print(f"  - {f}")

    # Prepare labels
    if args.labels:
        base_labels = args.labels
    else:
        # Use folder names as labels
        base_labels = [os.path.basename(os.path.normpath(d)) for d in args.dirs]

    # Open first video to get properties
    first_path = os.path.join(args.dirs[0], common_files[0])
    cap_init = cv2.VideoCapture(first_path)
    fps = cap_init.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 15.0
    
    # Target resolution (using first folder's first video)
    target_w = int(cap_init.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_h = int(cap_init.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_init.release()

    combined_w = target_w * 2
    combined_h = target_h * 2
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output, fourcc, fps, (combined_w, combined_h))

    start_sec = time_to_seconds(args.start)
    start_frame = int(start_sec * fps)
    frames_per_segment = int(args.duration * fps)

    for filename in common_files[:args.limit]:
        print(f"\nProcessing {filename}...")
        caps = []
        labels = []
        for i, d in enumerate(args.dirs):
            path = os.path.join(d, filename)
            cap = cv2.VideoCapture(path)
            caps.append(cap)
            labels.append(f"{base_labels[i]} | {filename}")

        process_video_segment(caps, labels, target_w, target_h, fps, start_frame, start_frame + frames_per_segment, out)
        
        for cap in caps:
            cap.release()

    out.release()
    print(f"Success! Merged sequence video saved to: {args.output}")

if __name__ == "__main__":
    main()
