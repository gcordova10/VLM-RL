import os
import cv2
import argparse
from PIL import Image

def convert_avi_to_gif(input_path, output_path, scale=0.5, skip_frames=1):
    # Sanitize path
    if input_path.startswith('@'): input_path = input_path[1:]

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    
    frames = []
    
    print(f"Reading video: {input_path}")
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % skip_frames == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            if scale != 1.0:
                new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
                pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
            
            frames.append(pil_img)
        
        count += 1
        if count % 100 == 0:
            print(f"Read {count} frames...", end='\r', flush=True)

    cap.release()
    
    if not frames:
        print("\nError: No frames were captured.")
        return

    print(f"\nSaving GIF to: {output_path} (Frames: {len(frames)})")
    
    duration = int((1000.0 / fps) * skip_frames)
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=duration,
        loop=0
    )
    
    print(f"Success! GIF saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert AVI to GIF using Pillow.')
    parser.add_argument('input', help='Path to input AVI file')
    parser.add_argument('--output', help='Path to output GIF file')
    parser.add_argument('--scale', type=float, default=0.5, help='Scale factor')
    parser.add_argument('--skip', type=int, default=1, help='Skip frames factor')

    args = parser.parse_args()
    
    out_path = args.output if args.output else args.input.replace('.avi', '.gif')
    
    convert_avi_to_gif(args.input, out_path, args.scale, args.skip)
