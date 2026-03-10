import os
import cv2
import argparse
import numpy as np

def get_subfolders(base_path):
    return sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])

def main():
    parser = argparse.ArgumentParser(description='Interactive Multi-Scenario Synchronized Viewer.')
    parser.add_argument('--base_dirs', nargs=4, required=True, help='4 Base directories (e.g. tensorboard runs)')
    parser.add_argument('--filenames', nargs=4, required=True, help='4 Filenames to show (one for each quadrant)')
    parser.add_argument('--labels', nargs=4, default=["TL", "TR", "BL", "BR"], help='Labels for quadrants')
    
    args = parser.parse_args()
    
    # Identify common subfolders (scenarios)
    scenario_sets = []
    for d in args.base_dirs:
        scenario_sets.append(set(get_subfolders(d)))
    
    common_scenarios = sorted(list(set.intersection(*scenario_sets)))
    
    if not common_scenarios:
        print("No common scenario subfolders found in all 4 base directories.")
        return

    print(f"Found {len(common_scenarios)} common scenarios.")
    
    current_scenario_idx = 0
    speed = 1
    
    while current_scenario_idx < len(common_scenarios):
        scenario = common_scenarios[current_scenario_idx]
        print(f"Playing Scenario [{current_scenario_idx+1}/{len(common_scenarios)}]: {scenario}")
        
        caps = []
        labels = []
        for i in range(4):
            path = os.path.join(args.base_dirs[i], scenario, args.filenames[i])
            cap = cv2.VideoCapture(path)
            caps.append(cap)
            labels.append(f"{args.labels[i]} | {scenario} | {args.filenames[i]}")

        # Get target resolution from first valid cap
        target_w, target_h = 0, 0
        for cap in caps:
            if cap.isOpened():
                target_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                target_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                break
        
        if target_w == 0:
            print(f"Error: Could not open any videos for scenario {scenario}")
            current_scenario_idx += 1
            continue

        skip_scenario = False
        while True:
            frames = []
            rets = []
            
            # Read frames with speed support
            for _ in range(speed):
                batch_rets = []
                batch_frames = []
                for cap in caps:
                    ret, frame = cap.read()
                    batch_rets.append(ret)
                    batch_frames.append(frame)
                
                # We only care about the last read in the speed loop for display
                # but we must consume the frames
                if _ == speed - 1:
                    rets = batch_rets
                    frames = batch_frames
                
                if not any(batch_rets): break

            if not any(rets):
                print(f"Finished scenario {scenario}.")
                break # End of scenario

            # Process frames for grid
            grid_frames = []
            for i, (ret, frame) in enumerate(zip(rets, frames)):
                if not ret or frame is None:
                    frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    cv2.putText(frame, "FINISHED / NOT FOUND", (10, target_h//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    if frame.shape[0] != target_h or frame.shape[1] != target_w:
                        frame = cv2.resize(frame, (target_w, target_h))
                
                # Add label
                label = labels[i]
                font_scale = 0.6
                thickness = 1
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                cv2.rectangle(frame, (2, 2), (10 + text_size[0], 10 + text_size[1]), (0, 0, 0), -1)
                cv2.putText(frame, label, (5, 5 + text_size[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                grid_frames.append(frame)

            # Create 2x2 Grid
            top = np.hstack((grid_frames[0], grid_frames[1]))
            bottom = np.hstack((grid_frames[2], grid_frames[3]))
            combined = np.vstack((top, bottom))
            
            # Resize combined for display if too large
            display_frame = combined
            max_h = 900
            if combined.shape[0] > max_h:
                scale = max_h / combined.shape[0]
                display_frame = cv2.resize(combined, None, fx=scale, fy=scale)

            cv2.imshow('VLM-RL Multi-Scenario Player: [n] Next [1,2,3] Speed [q] Quit', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                for cap in caps: cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                skip_scenario = True
                break
            elif key == ord('1'): speed = 1
            elif key == ord('2'): speed = 5
            elif key == ord('3'): speed = 10

        for cap in caps: cap.release()
        if skip_scenario or True: # Auto-advance or wait? User asked to "encadenar"
            current_scenario_idx += 1

    cv2.destroyAllWindows()
    print("All scenarios played.")

if __name__ == "__main__":
    main()
