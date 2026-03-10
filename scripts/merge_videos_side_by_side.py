import os
import cv2
import argparse
import numpy as np

def merge_videos(path_l, path_r, output_path):
    # Sanitize paths
    if path_l.startswith('@'): path_l = path_l[1:]
    if path_r.startswith('@'): path_r = path_r[1:]

    cap_l = cv2.VideoCapture(path_l)
    cap_r = cv2.VideoCapture(path_r)

    if not cap_l.isOpened() or not cap_r.isOpened():
        print("Error: Could not open one or both videos.")
        return

    # Get original properties
    fps = cap_l.get(cv2.CAP_PROP_FPS)
    w_l = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_l = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    w_r = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_r = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Final resolution: Sum of widths, Max of heights (WITHOUT RESIZING)
    target_h = max(h_l, h_r)
    combined_width = w_l + w_r

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, target_h))

    print(f"Merging with NATIVE RESOLUTION:")
    print(f"Left: {w_l}x{h_l} | Right: {w_r}x{h_r}")
    print(f"Output: {combined_width}x{target_h} @ {fps} FPS")

    frame_count = 0
    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()

        if not ret_l and not ret_r:
            break

        # Process Left (Add padding if shorter, NO RESIZE)
        if ret_l:
            if h_l < target_h:
                # Add black padding at the bottom to maintain original pixels
                canvas_l = np.zeros((target_h, w_l, 3), dtype=np.uint8)
                canvas_l[:h_l, :w_l] = frame_l
                frame_l = canvas_l
            
            # Label
            text_l = "BASELINE"
            font_scale = 1.0
            thickness = 2
            text_size = cv2.getTextSize(text_l, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cv2.putText(frame_l, text_l, (w_l - text_size[0] - 15, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        else:
            frame_l = np.zeros((target_h, w_l, 3), dtype=np.uint8)
            cv2.putText(frame_l, "BASELINE FINISHED", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # Process Right (Add padding if shorter, NO RESIZE)
        if ret_r:
            if h_r < target_h:
                canvas_r = np.zeros((target_h, w_r, 3), dtype=np.uint8)
                canvas_r[:h_r, :w_r] = frame_r
                frame_r = canvas_r
            
            # Label
            text_r = "CLG-Smooth"
            font_scale = 1.0
            thickness = 2
            text_size = cv2.getTextSize(text_r, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cv2.putText(frame_r, text_r, (w_r - text_size[0] - 15, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        else:
            frame_r = np.zeros((target_h, w_r, 3), dtype=np.uint8)
            cv2.putText(frame_r, "CLG-Smooth FINISHED", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # Combine side by side
        combined = np.hstack((frame_l, frame_r))
        out.write(combined)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...", end='\r')

    cap_l.release()
    cap_r.release()
    out.release()
    print(f"\nSuccess! Native resolution merged video saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge two videos side-by-side maintaining native resolution.')
    parser.add_argument('left', help='Path to left video')
    parser.add_argument('right', help='Path to right video')
    parser.add_argument('--output', default='/media/nemesis/disco4tb/Documents/VLM-RL/clips/merged_comparison.avi', help='Output path')

    args = parser.parse_args()
    merge_videos(args.left, args.right, args.output)
