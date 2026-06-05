import sys
import os
# Force Python to look in the current root directory for custom modules
sys.path.append(os.path.abspath('.'))

import torch
# --- FIX 1: CUDA 11.1 / Ampere GPU Hardware Patch ---
# Intercepts GPU matrix inversion and forces it to happen safely on the CPU
_original_inverse = torch.inverse
def _safe_inverse(x):
    return _original_inverse(x.cpu()).to(x.device)
torch.inverse = _safe_inverse
# ------------------------------------------------------

import cv2
import numpy as np
from mmcv import Config
from mmcv.utils import import_modules_from_strings
from mmdet3d.apis import init_model

# Argoverse 2 Class Colors (BGR format for OpenCV)
COLOR_MAPS_BGR = {
    0: (54, 137, 255),  # divider (Orange)
    1: (255, 0, 0),     # ped_crossing (Blue)
    2: (0, 0, 255),     # boundary (Red)
    3: (0, 255, 0)      # centerline (Green) - if applicable
}

# --- FIX 2: 3D to 2D Projection Function ---
def project_ego_to_image(pts_3d, img_w, img_h):
    """
    Simulates a standard dashboard camera to project 3D BEV points back onto the 2D video.
    You can tune the 'camera_height' and 'focal_length' to fit your specific video.
    """
    # TUNE THESE PARAMETERS FOR YOUR VIDEO:
    camera_height = 1.5  # Assume dashcam is 1.5 meters above the ground
    focal_length = 800   # Simulated focal length in pixels
    pitch_offset = 0.05  # Slight downward tilt of the camera
    
    projected_pts = []
    for pt in pts_3d:
        x_ego, y_ego = pt[0], pt[1]
        
        # Only project points that are in front of the camera (X > 0.5 meters)
        if x_ego < 0.5:
            continue
            
        # Convert from Ego 3D Coordinates to Camera Coordinates
        # Ego: X is forward, Y is left, Z is up (ground is 0)
        # Cam: Z is forward, X is right, Y is down
        z_cam = x_ego
        x_cam = -y_ego
        y_cam = camera_height - (x_ego * pitch_offset)
        
        # Pinhole projection math to get 2D pixel coordinates (u, v)
        u = int((x_cam * focal_length / z_cam) + (img_w / 2))
        v = int((y_cam * focal_length / z_cam) + (img_h / 2))
        
        # Keep points that are reasonably within or near the screen bounds
        if -500 < u < img_w + 500 and -500 < v < img_h + 500:
            projected_pts.append([u, v])
            
    return np.array(projected_pts, dtype=np.int32)
# -----------------------------------------

def main():
    # --- Configuration ---
    config_path = './projects/configs/maptrv2/maptrv2_av2_3d_r50_6ep.py' 
    checkpoint_path = './pretrained/maptrv2_av2_3d_r50_6ep.pth'
    video_path = './test_video.webm'  # Ensure this matches your video file path
    output_path = './outputs/maptrv2_demo_output2.mp4'
    device = 'cuda:0'
    score_thresh = 0.3
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Register MapTRv2 Modules ---
    cfg = Config.fromfile(config_path)
    custom_imports = cfg.get('custom_imports', None)
    if custom_imports is not None:
        import_modules_from_strings(**custom_imports)
    else:
        import_modules_from_strings(**dict(imports=['projects.mmdet3d_plugin.maptr.modules']))
        
    # --- Initialize Model ---
    print("Loading model weights... (This may take a moment)")
    model = init_model(config_path, checkpoint_path, device=device)
    model.eval()
    print("Model loaded successfully!")

    # --- Setup Video I/O ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    # Output video dimensions (Side-by-side: [Annotated Video Frame] | [BEV Map])
    target_h, target_w = 576, 1024
    bev_size = 600
    out_width = target_w + bev_size
    out_height = max(target_h, bev_size)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    print(f"Processing video. Output will be saved to: {output_path}")
    frame_count = 0

    # --- FIX 3: Simulated Camera Intrinsics ---
    # Stops the BEV hallucination by giving the network a realistic focal length
    focal_length_sim = 800.0  
    simulated_intrinsic = np.array([
        [focal_length_sim, 0.0, target_w / 2.0, 0.0],
        [0.0, focal_length_sim, target_h / 2.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    # --------------------------------------------

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        print(f"\rProcessing frame {frame_count}...", end="")

        # 1. Preprocess the input frame
        frame_resized = cv2.resize(frame, (target_w, target_h))
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # ImageNet Normalization
        norm_frame = (rgb_frame - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
        
        # 2. Fabricate the 6-camera input required by MapTRv2
        blank_frame = np.zeros_like(norm_frame)
        multi_view_frames = [norm_frame, blank_frame, blank_frame, blank_frame, blank_frame, blank_frame]
        
        # Shape: [batch(1), views(6), channels(3), H, W]
        img_tensor = torch.tensor(np.stack(multi_view_frames)).permute(0, 3, 1, 2).float().to(device)
        
        # 3. Fabricate dummy camera geometry matrices
        dummy_matrix = np.eye(4, dtype=np.float32)
        lidar2img_matrices = [dummy_matrix] * 6
        
        # 4. Wrap everything into the expected data-container dict
        data = {
            'img': [img_tensor],
            'img_metas': [[{
                'box_type_3d': None,
                'img_shape': [(target_h, target_w, 3) for _ in range(6)],
                'pad_shape': [(target_h, target_w, 3) for _ in range(6)],
                'scale_factor': [1.0] * 6,
                'flip': False,
                'pcd_horizontal_flip': False,
                'pcd_vertical_flip': False,
                'scene_token': 'custom_video_sequence', 
                'can_bus': np.zeros(18, dtype=np.float32), 
                'lidar2img': lidar2img_matrices,
                'camera_intrinsics': [simulated_intrinsic for _ in range(6)], 
                'camera2ego': [np.eye(4, dtype=np.float32) for _ in range(6)],
                'img_aug_matrix': [np.eye(4, dtype=np.float32) for _ in range(6)],   
                'lidar2ego': np.eye(4, dtype=np.float32),
                'ego2global': np.eye(4, dtype=np.float32),
                'lidar2global': np.eye(4, dtype=np.float32),
            }]]
        }
        
        # 5. Run Inference
        with torch.no_grad():
            results = model(return_loss=False, rescale=True, **data)
            
        # 6. Extract Predictions
        predictions = results[0]['pts_bbox']
        scores = predictions['scores_3d']
        labels = predictions['labels_3d']
        pts = predictions['pts_3d']
        
        valid_mask = scores > score_thresh
        
        # 7. Render BEV Map & 2D Perspective Annotations
        bev_canvas = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
        annotated_frame = frame_resized.copy() # Make a copy to draw the 2D lines on
        
        # Draw a simple car in the center of the BEV map
        center_x, center_y = bev_size // 2, bev_size // 2
        cv2.rectangle(bev_canvas, (center_x - 10, center_y - 20), (center_x + 10, center_y + 20), (255, 255, 255), -1)
        
        for score, label, polyline in zip(scores[valid_mask], labels[valid_mask], pts[valid_mask]):
            polyline = polyline.cpu().numpy()
            label_idx = label.item()
            color = COLOR_MAPS_BGR.get(label_idx, (255, 255, 255))
            
            # --- Draw the 3D BEV Line ---
            pixel_points = []
            for pt in polyline:
                # Map physical meters (e.g., -30m to +30m) to the BEV pixel canvas (0 to 600px)
                # Scale factor: 10 pixels per meter.
                x_px = int(center_x + pt[0] * 10)
                y_px = int(center_y - pt[1] * 10)
                pixel_points.append([x_px, y_px])
                
            pts_array = np.array(pixel_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(bev_canvas, [pts_array], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

            # --- Draw the 2D Annotated Line on the Video ---
            proj_pts = project_ego_to_image(polyline, target_w, target_h)
            if len(proj_pts) > 1: # Only draw if there are at least 2 valid points
                proj_pts_array = proj_pts.reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [proj_pts_array], isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA)

        # 8. Stitch images side-by-side
        final_canvas = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        
        # Place the ANNOTATED video frame on the left
        y_offset = (out_height - target_h) // 2
        final_canvas[y_offset:y_offset+target_h, :target_w] = annotated_frame
        
        # Place BEV map on the right
        final_canvas[:bev_size, target_w:] = bev_canvas
        
        # Write to video
        out_video.write(final_canvas)

    print("\nVideo processing complete!")
    cap.release()
    out_video.release()

if __name__ == '__main__':
    main()