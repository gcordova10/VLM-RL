import os
import cv2
import argparse
import numpy as np

def get_subfolders(base_path):
    if not os.path.exists(base_path): return []
    return sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])

def main():
    parser = argparse.ArgumentParser(description='Save Multi-Scenario Synchronized Grid to individual video files per scenario.')
    parser.add_argument('--base_dirs', nargs=4, required=True, help='4 Base directories')
    parser.add_argument('--filenames', nargs=4, required=True, help='4 Filenames (one for each quadrant)')
    parser.add_argument('--labels', nargs=4, default=["TL", "TR", "BL", "BR"], help='Labels for quadrants')
    parser.add_argument('--output_prefix', default='TF', help='Prefix for the output video files')
    parser.add_argument('--duration', type=float, default=None, help='Optional: limit each scenario to N seconds')
    parser.add_argument('--output_dir', default='video_metricas', help='Directory to save the videos')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created directory: {args.output_dir}")
    
    # Identify common subfolders (scenarios)
    scenario_sets = []
    for d in args.base_dirs:
        scenario_sets.append(set(get_subfolders(d)))
    
    common_scenarios = sorted(list(set.intersection(*scenario_sets)))
    
    if not common_scenarios:
        print("No common scenario subfolders found.")
        return

    print(f"Found {len(common_scenarios)} common scenarios. Saving individual videos to '{args.output_dir}/'...")
    
    # Get properties from the first valid video to setup the writer template
    fps, target_w, target_h = 15.0, 0, 0
    for i in range(4):
        path = os.path.join(args.base_dirs[i], common_scenarios[0], args.filenames[i])
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            target_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            target_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            break
    
    if target_w == 0:
        print("Error: Could not determine video properties.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    max_frames_per_scenario = int(args.duration * fps) if args.duration else float('inf')

    for s_idx, scenario in enumerate(common_scenarios):
        output_filename = os.path.join(args.output_dir, f"{args.output_prefix}_{scenario}.avi")
        print(f"Processing Scenario [{s_idx+1}/{len(common_scenarios)}]: {scenario} -> {output_filename}")
        
        # Initialize VideoWriter for THIS specific scenario
        out = cv2.VideoWriter(output_filename, fourcc, fps, (target_w * 2, target_h * 2))
        
        caps = []
        for i in range(4):
            path = os.path.join(args.base_dirs[i], scenario, args.filenames[i])
            caps.append(cv2.VideoCapture(path))

        frame_count = 0
        while frame_count < max_frames_per_scenario:
            grid_frames = []
            rets = []
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if ret:
                    if frame.shape[0] != target_h or frame.shape[1] != target_w:
                        frame = cv2.resize(frame, (target_w, target_h))
                    
                    # Add label
                    label = f"{args.labels[i]} | {scenario} | {args.filenames[i]}"
                    cv2.rectangle(frame, (2, 2), (frame.shape[1], 25), (0, 0, 0), -1)
                    cv2.putText(frame, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    cv2.putText(frame, "FINISHED", (target_w//3, target_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                grid_frames.append(frame)
                rets.append(ret)

            if not any(rets): break

            # Create 2x2 Grid
            top = np.hstack((grid_frames[0], grid_frames[1]))
            bottom = np.hstack((grid_frames[2], grid_frames[3]))
            combined = np.vstack((top, bottom))
            
            out.write(combined)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  Frame {frame_count}...", end='\r')

        for cap in caps: cap.release()
        out.release()
        print(f"\nFinished scenario: {scenario}")

    print(f"\nSuccess! All {len(common_scenarios)} videos saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
