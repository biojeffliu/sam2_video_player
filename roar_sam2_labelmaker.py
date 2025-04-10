#jl
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import io
import cv2
import json

class VideoSegmenter:
    def __init__(self, input_folder, output_folder="", input_format="", sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt", model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.input_format = input_format
        self.sam2_checkpoint = sam2_checkpoint
        self.model_cfg = model_cfg
        self.positive_points = {} # KEY: FRAME_NUMBER, VALUE: ((X_COORDINATE, Y_COORDINATE), OBJECT_ID)
        self.negative_points = {} # Same as above ^^^
        self.frame_masks = {}
        self.boxes = []
        self.drawing_box = False
        self.ix, self.iy = -1, -1

        self.device = self._get_device()
        self.predictor = self._build_predictor()

        self.current_frame = 0
        self.current_object_id = 1
        self.help_overlay_visible = False

        self.inference_state = self.predictor.init_state(video_path=self.input_folder)
    
    def _get_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
        if device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        return device

    def _build_predictor(self):
        predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint, device=self.device)
        return predictor

    def png_to_jpg_converter(self, folder_path):
        print(f"Converting {folder_path} PNG images to JPG, in place.")
        for idx, filename in enumerate(sorted(os.listdir(folder_path))):
            if filename.endswith(".PNG"):
                img_path = os.path.join(folder_path, filename)

                try:
                    img = Image.open(img_path)
                    new_name = f"{idx:05d}.jpg"
                    new_path = os.path.join(folder_path, new_name)

                    img.convert("RGB").save(new_path, "JPEG")

                    os.remove(img_path)
                    print(f"Converted {filename} to {new_name} and deleted the original PNG.")
                except Exception as e:
                    print(f"Error converting {filename}: {e}")
        print("Conversion complete!")
        print(sorted(os.listdir(folder_path)))

    # from notebooks/video_predictor_example.ipynb
    def show_mask(self, mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    # from notebooks/video_predictor_example.ipynb
    def show_points(self, coords, labels, ax, marker_size=200):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Append positive points to the dictionary for the current frame
            if self.current_frame not in self.positive_points:
                self.positive_points[self.current_frame] = []
            self.positive_points[self.current_frame].append(((x, y), self.current_object_id))
            print(f"Added positive point: ({x}, {y}) to frame {self.current_frame} with object ID {self.current_object_id}")
            cv2.drawMarker(param, (x, y), (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)

        elif event == cv2.EVENT_MBUTTONDOWN:
            # Append negative points to the dictionary for the current frame
            if self.current_frame not in self.negative_points:
                self.negative_points[self.current_frame] = []
            self.negative_points[self.current_frame].append(((x, y), self.current_object_id))
            print(f"Added negative point: ({x}, {y}) to frame {self.current_frame} with object ID {self.current_object_id}")
            cv2.drawMarker(param, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)

    def get_current_image(self):
        images = sorted([os.path.join(self.input_folder, img) for img in os.listdir(self.input_folder) if img.endswith((".jpg", ".png", ".jpeg"))])
        img_path = images[self.current_frame]
        img = cv2.imread(img_path)
        return img
    
    def show_help_overlay(self, img):
        help_text = """
        Controls:
        - Arrow Right (d)           : Next frame
        - Arrow Left (a)            : Previous frame
        - First frame (z)           : Jump to first frame
        - Last frame (c)            : Jump to last frame
        - Middle-click (M)          : Add negative point on the current object ID
        - Left-click (L)            : Add positive point on the current object ID
        - Help (h)                  : Toggle this help overlay
        - Quit (q)                  : Quit video player
        - Segment current frame (s) : (SAM2) Segments current frame on the current object ID
        - Propagate masklets (r)    : (SAM2) Propagates masklets on the current object ID
        - Next object ID (l)        : Increases object ID
        - Previous object ID (j)    : Decreases object ID
        """
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        y0, dy = 15, 20  
        overlay_height = 400 

        cv2.rectangle(img, (0, 0), (img.shape[1], overlay_height), (0, 0, 0), -1)

        for i, line in enumerate(help_text.split('\n')):
            y = y0 + i * dy
            cv2.putText(img, line.strip(), (10, y), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


    def segment_points(self, ann_frame_idx):
        print(f"Segmenting points for frame {self.current_frame} with object ID {self.current_object_id}")
        pos_all = self.positive_points.get(ann_frame_idx, [])
        neg_all = self.negative_points.get(ann_frame_idx, [])
        pos_points = [pt for (pt, obj_id) in pos_all if obj_id == self.current_object_id] # filter points for just current obj_id
        neg_points = [pt for (pt, obj_id) in neg_all if obj_id == self.current_object_id]

        if not pos_points and not neg_points:
            print(f"No points for frame {ann_frame_idx} with object ID {self.current_object_id}.")
            return None
        
        pos_points = np.array(pos_points, dtype=np.float32) if pos_points else np.empty((0,2), dtype=np.float32)
        neg_points = np.array(neg_points, dtype=np.float32) if neg_points else np.empty((0,2), dtype=np.float32)
        
        if len(pos_points) > 0 and len(neg_points) > 0:
            points = np.vstack([pos_points, neg_points])
            labels = np.array([1] * len(pos_points) + [0] * len(neg_points), np.int32)
        elif len(pos_points) > 0:
            points = pos_points
            labels = np.array([1] * len(pos_points), np.int32)
        else:
            points = neg_points
            labels = np.array([0] * len(neg_points), np.int32)
        
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=ann_frame_idx,
            obj_id=self.current_object_id,
            points=points,
            labels=labels,
        )
        
        mask_probabilities = torch.sigmoid(out_mask_logits)
        binary_mask = (mask_probabilities > 0.5).int()
        binary_mask_numpy = binary_mask.squeeze().cpu().numpy()
        object_mask = binary_mask_numpy * self.current_object_id
        self.frame_masks[ann_frame_idx] = object_mask
        return object_mask
    
    def propagate_masks(self):
        print(f"Propagating masks starting from frame {self.current_frame} for object ID {self.current_object_id}.")
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            mask_shape = torch.sigmoid(out_mask_logits[0]).squeeze().cpu().numpy().shape
            final_mask = np.zeros(mask_shape, dtype=np.int32)
            for i, out_obj_id in enumerate(out_obj_ids):
                mask_prob = torch.sigmoid(out_mask_logits[i])
                binary_mask = (mask_prob > 0.5).int().squeeze().cpu().numpy()
                # Important: objects with a higher object ID will overwrite those with lower ID, if the masks overlap.
                # Find a fix for this later (probably ignore and just put them in separate masks)
                final_mask[binary_mask > 0] = out_obj_id
            self.frame_masks[out_frame_idx] = final_mask
            video_segments[out_frame_idx] = final_mask
        print("Masks propagated for video.")
        self.predictor.reset_state(self.inference_state)
        return video_segments
        

    # testing
    # def propagate_masks(self):
    #     print(f"Propagating masks starting from frame {self.current_frame} for object ID {self.current_object_id}.")
    #     video_segments = {}
    #     for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
    #         video_segments[out_frame_idx] = {
    #             out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
    #             for i, out_obj_id in enumerate(out_obj_ids)
    #         }
    #     print(video_segments)

    def video_player(self):
        images = sorted([os.path.join(self.input_folder, img) for img in os.listdir(self.input_folder) if img.endswith((".jpg", ".png", ".jpeg"))])
        if not images:
            print("No images found in the folder.")
            return

        viewing_frame = 0

        cv2.namedWindow("Video Player")
        # cv2.setMouseCallback("Video Player", self.mouse_callback)

        print("Video Player loaded. Press `h` to toggle help overlay.")
        cv2.setMouseCallback("Video Player", self.mouse_callback)

        while True:
            img_path = images[viewing_frame]
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                break
            frame_info = f"Frame: {viewing_frame} / {len(images) - 1}"
            object_info = f"Object ID: {self.current_object_id}"
            info_text = f"{frame_info}   {object_info}"
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(img, info_text, (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            if self.help_overlay_visible:
                self.show_help_overlay(img)

            if self.current_frame in self.positive_points:
                for point, obj_id in self.positive_points[self.current_frame]:
                    cv2.drawMarker(img, point, (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=5, thickness=1)
            if self.current_frame in self.negative_points:
                for point, obj_id in self.negative_points[self.current_frame]:
                    cv2.drawMarker(img, point, (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=5, thickness=1)

            if viewing_frame in self.frame_masks:
                mask = self.frame_masks[viewing_frame]
                # Case 1: Single object segmentation (2D mask)
                if mask.ndim == 2:
                    binary_mask = (mask > 0).astype(np.uint8)
                    mask_vis = np.stack([binary_mask]*3, axis=-1)
                    mask_vis = mask_vis * np.array([180, 105, 255], dtype=np.uint8)
                # Case 2: Multiple objects (3D mask: shape (num_objects, H, W))
                elif mask.ndim == 3:
                    colors = [(180,105,255), (255,0,0), (0,255,0), (0,0,255), (255,255,0)]
                    H, W = mask.shape[1], mask.shape[2]
                    mask_vis = np.zeros((H, W, 3), dtype=np.uint8)
                    for i in range(mask.shape[0]):
                        binary_mask = (mask[i] > 0).astype(np.uint8)
                        color = colors[i % len(colors)]
                        colored_mask = np.stack([binary_mask]*3, axis=-1) * np.array(color, dtype=np.uint8)
                        mask_vis = cv2.addWeighted(mask_vis, 1.0, colored_mask, 0.5, 0)
                img = cv2.addWeighted(img, 1, mask_vis, 0.5, 0)




            # if viewing_frame in self.frame_masks:
            #     mask = self.frame_masks[viewing_frame]

            #     # Case 1: Single object segmentation (2D mask)
            #     if mask.ndim == 2:
            #         binary_mask = (mask > 0).astype(np.uint8)
            #         mask_vis = np.stack([binary_mask]*3, axis=-1)
            #         mask_vis = mask_vis * np.array([180, 105, 255], dtype=np.uint8)

            #         # Find contours to determine centroid
            #         contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #         for cnt in contours:
            #             M = cv2.moments(cnt)
            #             if M["m00"] != 0:
            #                 cX = int(M["m10"] / M["m00"])
            #                 cY = int(M["m01"] / M["m00"])
            #                 cv2.putText(img, str(self.current_object_id), (cX, cY), 
            #                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            #     # Case 2: Multiple objects (3D mask: shape (num_objects, H, W))
            #     elif mask.ndim == 3:
            #         colors = [(180, 105, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            #         H, W = mask.shape[1], mask.shape[2]
            #         mask_vis = np.zeros((H, W, 3), dtype=np.uint8)

            #         for i in range(mask.shape[0]):
            #             binary_mask = (mask[i] > 0).astype(np.uint8)
            #             color = colors[i % len(colors)]
            #             colored_mask = np.stack([binary_mask]*3, axis=-1) * np.array(color, dtype=np.uint8)
            #             mask_vis = cv2.addWeighted(mask_vis, 1.0, colored_mask, 0.5, 0)

            #             # Find contours and draw object IDs
            #             contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #             for cnt in contours:
            #                 M = cv2.moments(cnt)
            #                 if M["m00"] != 0:
            #                     cX = int(M["m10"] / M["m00"])
            #                     cY = int(M["m01"] / M["m00"])
            #                     cv2.putText(img, str(i + 1), (cX, cY), 
            #                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            #     img = cv2.addWeighted(img, 1, mask_vis, 0.5, 0)


            cv2.imshow("Video Player", img)


            key = cv2.waitKey(10) & 0xFF

            
            if key == ord('q'):
                break
            elif key == ord('d'): # next frame
                viewing_frame = min(viewing_frame + 1, len(images) - 1)
                self.current_frame = viewing_frame
            elif key == ord('a'): # previous frame
                viewing_frame = max(viewing_frame - 1, 0)
                self.current_frame = viewing_frame
            elif key == ord('z'):  #First frame
                viewing_frame = 0
                self.current_frame = viewing_frame  # Update current_frame
            elif key == ord('c'):  # Last frame
                viewing_frame = len(images) - 1
                self.current_frame = viewing_frame
            elif key == ord('h'):  # Toggle help overlay
                self.help_overlay_visible = not self.help_overlay_visible
            elif key == ord('s'): # segment current frame.
                # print(f"Segmenting points for frame {self.current_frame}")
                segmented_mask = self.segment_points(self.current_frame)
            elif key == ord('r'): # propagate prompts to get masklets for whole video
                # print(f"Propagating masks starting from frame {self.current_frame}")
                video_segments = self.propagate_masks()
            elif key == ord('j'): # decrease object id
                self.current_object_id = max(self.current_object_id - 1, 1)
            elif key == ord('l'): # increase object id
                self.current_object_id += 1
            # elif key == ord('o'): # save all current segmentations and masks
            #     if not self.output_folder:
            #         print("Output folder not provided. Please set output_folder.")
            #     else:
            #         serializable_masks = {str(k): v.tolist() for k, v in self.frame_masks.items()}
            #         output_path = os.path.join(self.output_folder, "segmentations.json")
            #         with open(output_path, "w") as f:
            #             json.dump(serializable_masks, f)
            #         print(f"Segmentations saved to {output_path}")


    def main(self):
        if self.input_format == "PNG":
            self.png_to_jpg_converter(self.input_folder)
        self.video_player()

if __name__ == "__main__":
    input_folder = "./videos/ims_2024_day6_run2_vimba_rear_filtered"
    output_folder = "./output/ims_2024_day6_run2_vimba_rear_filtered"
    input_format = ""
    segmenter = VideoSegmenter(input_folder, output_folder, input_format)
    segmenter.main()
