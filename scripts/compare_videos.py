import cv2
import os
import sys
import numpy as np
import argparse

def play_videos(baseline_dir, new_dir):
    # Get all .avi files in baseline_dir
    baseline_files = [f for f in os.listdir(baseline_dir) if f.endswith('.avi')]
    # Sort them naturally
    import re
    def natural_key(string_):
        return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
    baseline_files.sort(key=natural_key)

    i = 0
    while i < len(baseline_files):
        filename = baseline_files[i]
        baseline_path = os.path.join(baseline_dir, filename)
        new_path = os.path.join(new_dir, filename)

        if not os.path.exists(new_path):
            print(f"Skipping {filename}: Not found in {new_dir}")
            i += 1
            continue

        print(f"Playing pair: {filename}")
        
        cap_l = cv2.VideoCapture(baseline_path)
        cap_r = cv2.VideoCapture(new_path)

        # Get common dimensions
        height_l = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
        height_r = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))
        target_h = max(height_l, height_r)
        
        speed = 1
        skip_to_next = False
        jump_to_index = -1
        search_mode = False
        search_text = ""

        def process_frame(frame, target_h, label, filename, search_active, search_str):
            if frame is None:
                return None
            h, w, _ = frame.shape
            if h != target_h:
                scale = target_h / h
                new_w = int(w * scale)
                frame = cv2.resize(frame, (new_w, target_h))
            
            # Add labels and filename
            cv2.putText(frame, f"{label} | Speed: {speed}x", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, filename, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            if search_active:
                # Draw a search box overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, target_h - 100), (w, target_h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                cv2.putText(frame, f"SEARCH STEPS: {search_str}_", (20, target_h - 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, "Press ENTER to Search | ESC to Cancel", (20, target_h - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            
            return frame

        last_l = None
        last_r = None

        while True:
            # Handle speed by skipping frames (only if not searching)
            if not search_mode:
                for _ in range(speed):
                    ret_l, frame_l = cap_l.read()
                    ret_r, frame_r = cap_r.read()
                    if not ret_l and not ret_r:
                        break
            else:
                # Keep current frames when searching
                ret_l, ret_r = (last_l is not None), (last_r is not None)

            if not ret_l and not ret_r and not search_mode:
                break # Both finished

            if ret_l and not search_mode:
                last_l_raw = frame_l.copy()
                last_l = process_frame(frame_l, target_h, "BASELINE", filename, search_mode, search_text)
            elif last_l is not None:
                # Re-process last frame for search overlay
                temp_l = last_l_raw.copy() if not ret_l else frame_l.copy()
                last_l = process_frame(temp_l, target_h, "BASELINE", filename, search_mode, search_text)
                if not ret_l:
                    cv2.putText(last_l, "FINISHED", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            if ret_r and not search_mode:
                last_r_raw = frame_r.copy()
                last_r = process_frame(frame_r, target_h, "NEW", filename, search_mode, search_text)
            elif last_r is not None:
                # Re-process last frame for search overlay
                temp_r = last_r_raw.copy() if not ret_r else frame_r.copy()
                last_r = process_frame(temp_r, target_h, "NEW", filename, search_mode, search_text)
                if not ret_r:
                    cv2.putText(last_r, "FINISHED", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            if last_l is None or last_r is None:
                continue

            combined = np.hstack((last_l, last_r))
            cv2.imshow('Comparison: [1,2,3]=Speed | [n]=Next | [s]=Search | [q]=Quit', combined)

            key = cv2.waitKey(1 if search_mode else 1) & 0xFF
            
            if search_mode:
                if key == 13: # ENTER
                    target_filename = f"model_{search_text}_steps_eval.avi"
                    try:
                        idx = baseline_files.index(target_filename)
                        if os.path.exists(os.path.join(new_dir, target_filename)):
                            jump_to_index = idx
                            break
                        else:
                            print(f"Error: {target_filename} not found in NEW directory.")
                    except ValueError:
                        print(f"Error: {target_filename} not found in BASELINE directory.")
                    search_mode = False
                    search_text = ""
                elif key == 27: # ESC
                    search_mode = False
                    search_text = ""
                elif key == 8: # Backspace
                    search_text = search_text[:-1]
                elif ord('0') <= key <= ord('9'):
                    search_text += chr(key)
                continue

            if key == ord('q'):
                cap_l.release()
                cap_r.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                skip_to_next = True
                break
            elif key == ord('s'):
                search_mode = True
                search_text = ""
            elif key == ord('1'):
                speed = 1
            elif key == ord('2'):
                speed = 5
            elif key == ord('3'):
                speed = 10

        cap_l.release()
        cap_r.release()
        
        if jump_to_index != -1:
            i = jump_to_index
            continue
        
        if skip_to_next:
            i += 1
            continue

        print(f"Finished {filename}. Press 'n' or any key for next, 's' to search, 'q' to quit.")
        wait_key = cv2.waitKey(0) & 0xFF
        if wait_key == ord('q'):
            break
        elif wait_key == ord('s'):
            print("\n--- SEARCH ---")
            step_str = input("Enter model steps to jump to (e.g. 320000): ").strip()
            target_filename = f"model_{step_str}_steps_eval.avi"
            try:
                idx = baseline_files.index(target_filename)
                if os.path.exists(os.path.join(new_dir, target_filename)):
                    i = idx
                else:
                    print(f"Error: {target_filename} not found in NEW directory.")
                    i += 1
            except ValueError:
                print(f"Error: {target_filename} not found in BASELINE directory.")
                i += 1
        else:
            i += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play videos side by side.')
    parser.add_argument('baseline', help='Path to baseline videos directory')
    parser.add_argument('new', help='Path to new videos directory')
    args = parser.parse_args()

    play_videos(args.baseline, args.new)
