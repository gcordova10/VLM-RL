import os
import cv2
import argparse

def time_to_seconds(t_str):
    parts = t_str.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])

def cut_video(input_path, start_time, end_time, output_dir):
    if input_path.startswith('@'):
        input_path = input_path[1:]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, f"cut_{filename}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Para ALTA CALIDAD en OpenCV:
    # Usamos MJPG (Motion JPEG) que es casi lossless pero genera archivos grandes,
    # o XVID con una configuración de bitrate alto si estuviera disponible.
    # MJPG suele ser la opción más segura para mantener cada pixel exacto.
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_sec = time_to_seconds(start_time)
    end_sec = time_to_seconds(end_time)
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    print(f"Cutting High Quality (OpenCV MJPG) from {start_time} to {end_time}...")
    print(f"Resolution: {width}x{height} @ {fps} FPS")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1
        if current_frame % 100 == 0:
            print(f"  Processed {current_frame - start_frame} frames...", end='\r')

    cap.release()
    out.release()
    print(f"\nSuccess! High-resolution cut saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cut video using OpenCV with High Quality.')
    parser.add_argument('input', help='Input video file')
    parser.add_argument('--start', default='00:00:00', help='Start (HH:MM:SS)')
    parser.add_argument('--end', default='00:00:10', help='End (HH:MM:SS)')
    parser.add_argument('--outdir', default='clips', help='Output directory')

    args = parser.parse_args()
    cut_video(args.input, args.start, args.end, args.outdir)
