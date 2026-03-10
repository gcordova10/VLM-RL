import os
import cv2
import argparse
import numpy as np
from datetime import datetime

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

def merge_4_videos(paths, labels, output_path, start_time=None, end_time=None):
    caps = []
    for path in paths:
        if path.startswith('@'): path = path[1:]
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Could not open video {path}")
            return
        caps.append(cap)

    # Use first video to get properties
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 20.0 # Default if unknown
    
    # Get resolutions
    ws = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in caps]
    hs = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps]
    
    # Determine target resolution for each quadrant (use max width and height)
    target_w = max(ws)
    target_h = max(hs)
    
    combined_w = target_w * 2
    combined_h = target_h * 2

    start_sec = time_to_seconds(start_time) if start_time else 0
    end_sec = time_to_seconds(end_time) if end_time else None
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps) if end_sec else None

    # Seek to start
    if start_frame > 0:
        for cap in caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_w, combined_h))

    print(f"Merging 4 videos into 2x2 grid:")
    for i, (path, label) in enumerate(zip(paths, labels)):
        print(f"  Quadrant {i+1}: {label} ({path})")
    print(f"Output: {combined_w}x{combined_h} @ {fps} FPS")
    if start_time: print(f"Start: {start_time} ({start_frame}f)")
    if end_time: print(f"End: {end_time} ({end_frame}f)")

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
                # Resize to target quadrant size
                if frame.shape[0] != target_h or frame.shape[1] != target_w:
                    frame = cv2.resize(frame, (target_w, target_h))
            else:
                frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            frames.append(frame)

        if all_finished:
            break

        # Add labels
        for i, (frame, label) in enumerate(zip(frames, labels)):
            # Label
            font_scale = 1.0
            thickness = 2
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            # Draw rectangle for background text
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
            print(f"Processed {frame_idx - start_frame} frames...")

    for cap in caps:
        cap.release()
    out.release()
    print(f"Success! 2x2 grid video saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge 4 videos into a 2x2 grid with labels.')
    parser.add_argument('--tl', required=True, help='Top Left path')
    parser.add_argument('--tr', required=True, help='Top Right path')
    parser.add_argument('--bl', required=True, help='Bottom Left path')
    parser.add_argument('--br', required=True, help='Bottom Right path')
    parser.add_argument('--ltl', default='Top Left', help='Label for Top Left')
    parser.add_argument('--ltr', default='Top Right', help='Label for Top Right')
    parser.add_argument('--lbl', default='Bottom Left', help='Label for Bottom Left')
    parser.add_argument('--lbr', default='Bottom Right', help='Label for Bottom Right')
    parser.add_argument('--output', default='clips/grid_comparison.avi', help='Output path')
    parser.add_argument('--start', help='Start time (HH:MM:SS)')
    parser.add_argument('--end', help='End time (HH:MM:SS)')

    args = parser.parse_args()
    
    paths = [args.tl, args.tr, args.bl, args.br]
    labels = [args.ltl, args.ltr, args.lbl, args.lbr]
    
    merge_4_videos(paths, labels, args.output, args.start, args.end)
