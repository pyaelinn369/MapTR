import sys
import os
# Force Python to look in the current root directory for custom modules
sys.path.append(os.path.abspath('.'))

import torch
# --- FIX 1: CUDA 11.1 / Ampere GPU Hardware Patch ---
_original_inverse = torch.inverse
def _safe_inverse(x):
    return _original_inverse(x.cpu()).to(x.device)
torch.inverse = _safe_inverse
# ------------------------------------------------------

import cv2
import numpy as np
from pathlib import Path
from mmcv import Config
from mmcv.utils import import_modules_from_strings
from mmdet3d.apis import init_model

# --- FIX 2: Modern High-Level ROS Bag Reader ---
from rosbags.highlevel import AnyReader
# -------------------------------------------------

# Argoverse 2 Class Colors (BGR format for OpenCV)
COLOR_MAPS_BGR = {
    0: (54, 137, 255),  # divider (Orange)
    1: (255, 0, 0),     # ped_crossing (Blue)
    2: (0, 0, 255),     # boundary (Red)
    3: (0, 255, 0)      # centerline (Green)
}

# ==========================================
# ROS BAG TOPICS (Updated for your bag)
# ==========================================
BAG_PATH = './ROS_Bag_20260512_3Views_undis_V5.bag'

TOPIC_FRONT_WIDE = '/front_wide_undistort_camera/image_raw'
TOPIC_FRONT_NARROW = '/front_narrow_undistort_camera/image_raw'
TOPIC_REAR = '/rear_undistort_camera/image_raw'
# ==========================================

def project_ego_to_image(pts_3d, img_w, img_h):
    """Simulates a standard dashboard camera to project 3D BEV points back onto the 2D video."""
    camera_height = 1.5  
    focal_length = 800   
    pitch_offset = 0.05  
    
    projected_pts = []
    for pt in pts_3d:
        x_ego, y_ego = pt[0], pt[1]
        if x_ego < 0.5:
            continue
            
        z_cam = x_ego
        x_cam = -y_ego
        y_cam = camera_height - (x_ego * pitch_offset)
        
        u = int((x_cam * focal_length / z_cam) + (img_w / 2))
        v = int((y_cam * focal_length / z_cam) + (img_h / 2))
        
        if -500 < u < img_w + 500 and -500 < v < img_h + 500:
            projected_pts.append([u, v])
            
    return np.array(projected_pts, dtype=np.int32)

def decode_ros_image(msg, msgtype):
    """Converts a ROS Image message into an OpenCV BGR numpy array."""
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    if msg.encoding == 'rgb8':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif msg.encoding == 'bayer_rggb8':
        img = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
    return img

def main():
    config_path = './projects/configs/maptrv2/maptrv2_av2_3d_r50_6ep.py' 
    checkpoint_path = './pretrained/maptrv2_av2_3d_r50_6ep.pth'
    output_path = './outputs/maptrv2_rosbag_output.mp4'
    device = 'cuda:0'
    score_thresh = 0.3
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Register MapTRv2 Modules ---
    cfg = Config.fromfile(config_path)
    custom_imports = cfg.get('custom_imports', None)
    if custom_imports is not None:
        import_modules_from_strings(**custom_imports)
    else:
        import_modules_from_strings(**dict(imports=['projects.mmdet3d_plugin.maptr.modules']))
        
    print("Loading model weights... (This may take a moment)")
    model = init_model(config_path, checkpoint_path, device=device)
    model.eval()
    print("Model loaded successfully!")

    target_h, target_w = 576, 1024
    bev_size = 600
    out_width = target_w + bev_size
    out_height = max(target_h, bev_size)
    
    # Set FPS to 2 since you only have 5 frames total; makes it easier to view the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, 2.0, (out_width, out_height)) 

    focal_length_sim = 800.0  
    simulated_intrinsic = np.array([
        [focal_length_sim, 0.0, target_w / 2.0, 0.0],
        [0.0, focal_length_sim, target_h / 2.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)

    latest_frames = {
        TOPIC_FRONT_WIDE: None,
        TOPIC_FRONT_NARROW: None,
        TOPIC_REAR: None
    }

    print(f"Reading ROS Bag: {BAG_PATH}")
    frame_count = 0

    # --- NEW FIX: Wrap BAG_PATH in Path() to satisfy AnyReader typing ---
    with AnyReader([Path(BAG_PATH)]) as reader:
        connections = [x for x in reader.connections if x.topic in latest_frames.keys()]
        
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            # Natively deserialize without the buggy ROS1->CDR pipeline
            msg = reader.deserialize(rawdata, connection.msgtype)
            
            img = decode_ros_image(msg, connection.msgtype)
            latest_frames[connection.topic] = img
            
            # TRIGGER INFERENCE on Front Wide
            if connection.topic == TOPIC_FRONT_WIDE:
                
                # Make sure we have frames for the others
                if latest_frames[TOPIC_FRONT_NARROW] is None or latest_frames[TOPIC_REAR] is None:
                    continue
                    
                frame_count += 1
                print(f"\rProcessing synced frame {frame_count} / 5...", end="")

                # 1. Preprocess
                processed_views = []
                for topic in [TOPIC_FRONT_WIDE, TOPIC_FRONT_NARROW, TOPIC_REAR]:
                    frame = latest_frames[topic]
                    frame_resized = cv2.resize(frame, (target_w, target_h))
                    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    norm_frame = (rgb_frame - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
                    processed_views.append(norm_frame)
                
                # 2. Fabricate the 6-camera input 
                # Mapping: 0: Front-Wide, 1: Front-Narrow, 3: Rear. Others blank.
                blank_frame = np.zeros_like(processed_views[0])
                multi_view_frames = [
                    processed_views[0], # CAM_FRONT
                    processed_views[1], # CAM_FRONT_RIGHT (Using narrow here temporarily)
                    blank_frame,        # CAM_FRONT_LEFT
                    processed_views[2], # CAM_REAR
                    blank_frame,        # CAM_REAR_LEFT
                    blank_frame         # CAM_REAR_RIGHT
                ]
                
                img_tensor = torch.tensor(np.stack(multi_view_frames)).permute(0, 3, 1, 2).float().to(device)
                dummy_matrix = np.eye(4, dtype=np.float32)
                lidar2img_matrices = [dummy_matrix] * 6
                
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
                        'scene_token': 'custom_bag_sequence', 
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
                
                with torch.no_grad():
                    results = model(return_loss=False, rescale=True, **data)
                    
                predictions = results[0]['pts_bbox']
                scores = predictions['scores_3d']
                labels = predictions['labels_3d']
                pts = predictions['pts_3d']
                
                valid_mask = scores > score_thresh
                
                bev_canvas = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
                
                # Draw on the Front-Wide Camera
                front_frame_raw = latest_frames[TOPIC_FRONT_WIDE]
                annotated_frame = cv2.resize(front_frame_raw, (target_w, target_h)).copy()
                
                center_x, center_y = bev_size // 2, bev_size // 2
                cv2.rectangle(bev_canvas, (center_x - 10, center_y - 20), (center_x + 10, center_y + 20), (255, 255, 255), -1)
                
                for score, label, polyline in zip(scores[valid_mask], labels[valid_mask], pts[valid_mask]):
                    polyline = polyline.cpu().numpy()
                    label_idx = label.item()
                    color = COLOR_MAPS_BGR.get(label_idx, (255, 255, 255))
                    
                    pixel_points = []
                    for pt in polyline:
                        x_px = int(center_x + pt[0] * 10)
                        y_px = int(center_y - pt[1] * 10)
                        pixel_points.append([x_px, y_px])
                        
                    pts_array = np.array(pixel_points, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(bev_canvas, [pts_array], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

                    proj_pts = project_ego_to_image(polyline, target_w, target_h)
                    if len(proj_pts) > 1: 
                        proj_pts_array = proj_pts.reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [proj_pts_array], isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA)

                final_canvas = np.zeros((out_height, out_width, 3), dtype=np.uint8)
                y_offset = (out_height - target_h) // 2
                final_canvas[y_offset:y_offset+target_h, :target_w] = annotated_frame
                final_canvas[:bev_size, target_w:] = bev_canvas
                
                out_video.write(final_canvas)

    print("\nROS Bag processing complete! Output saved.")
    out_video.release()

if __name__ == '__main__':
    main()