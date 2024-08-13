# Prediction interface for Cog ⚙️
# https://cog.run/python


import os
import io
import cv2
import time
import torch
import mimetypes
import subprocess
import numpy as np
from PIL import Image
from typing import List, Iterator
import matplotlib.pyplot as plt
from cog import BasePredictor, Input, Path, BaseModel, ConcatenateIterator

mimetypes.add_type("image/webp", ".webp")


DEVICE = "cuda"
MODEL_CACHE = "checkpoints"
BASE_URL = f"https://weights.replicate.delivery/default/sam-2/{MODEL_CACHE}/"


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        global build_sam2_video_predictor

        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError:
            print("sam2 not found. Installing...")
            os.system("pip install --no-build-isolation -e .")
            from sam2.build_sam import build_sam2_video_predictor

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
        model_files = ["sam2_hiera_large.pt"]
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        model_cfg = "sam2_hiera_l.yaml"
        sam2_checkpoint = f"{MODEL_CACHE}/sam2_hiera_large.pt"

        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

        # Enable bfloat16 and TF32 for better performance
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def show_anns(self, anns, obj_ids):
        if len(anns) == 0:
            return
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.zeros((*anns[0].shape[-2:], 4))
        img[:, :, 3] = 0

        cmap = plt.get_cmap("tab10")
        for ann, obj_id in zip(anns, obj_ids):
            m = ann.squeeze().astype(bool)
            color = np.array([*cmap(obj_id % 10)[:3], 0.6])
            img[m] = color
        ax.imshow(img)

    def parse_inputs(
        self,
        click_coordinates: str,
        click_labels: str,
        click_frames: str,
        click_object_ids: str,
    ) -> tuple:
        # Parse click coordinates
        click_list = [
            list(map(int, map(str.strip, click.strip()[1:-1].split(","))))
            for click in click_coordinates.strip().strip("[]").split("],")
        ]
        num_clicks = len(click_list)

        # Handle click labels
        # Strip whitespace and split by comma, then extend if necessary
        click_labels_list = [label.strip() for label in click_labels.split(",")]
        click_labels_list = (
            click_labels_list
            * ((num_clicks + len(click_labels_list) - 1) // len(click_labels_list))
        )[:num_clicks]

        # Handle click frames
        # Strip whitespace and split by comma, use default '0' if not provided, then extend if necessary
        click_frames_list = (
            [frame.strip() for frame in click_frames.split(",")]
            if click_frames
            else ["0"] * num_clicks
        )
        click_frames_list = (
            click_frames_list
            * ((num_clicks + len(click_frames_list) - 1) // len(click_frames_list))
        )[:num_clicks]

        # Handle click object IDs
        if click_object_ids:
            # Strip whitespace from each object ID
            object_ids_list = [obj_id.strip() for obj_id in click_object_ids.split(",")]
            # If not enough object IDs provided, extend with sequential labels
            if len(object_ids_list) < num_clicks:
                object_ids_list.extend(
                    f"object_{i}"
                    for i in range(len(object_ids_list) + 1, num_clicks + 1)
                )
        else:
            # If no object IDs provided, generate sequential labels
            object_ids_list = [f"object_{i}" for i in range(1, num_clicks + 1)]
        object_ids_list = object_ids_list[:num_clicks]

        # Map string labels to unique integer IDs
        # This is necessary because the underlying algorithm requires integer IDs,
        # but we want to allow users to input meaningful string labels
        label_to_id = {}
        id_counter = 1
        object_ids_int_list = []
        for label in object_ids_list:
            if label not in label_to_id:
                label_to_id[label] = id_counter
                id_counter += 1
            object_ids_int_list.append(label_to_id[label])

        # Convert all inputs to appropriate types
        # Ensure all data is in the correct format for processing
        click_list = [list(map(int, coords)) for coords in click_list]
        click_labels_list = list(map(int, click_labels_list))
        click_frames_list = list(map(int, click_frames_list))

        # Validation
        # Ensure all input lists have the same length to prevent inconsistencies in processing
        if not (
            len(click_list)
            == len(click_labels_list)
            == len(click_frames_list)
            == len(object_ids_list)
        ):
            raise ValueError(
                "Mismatch in the number of clicks, click types, click frames, or object IDs."
            )

    def predict(
        self,
        input_video: Path = Input(description="Input video file path"),
        # Segmentation inputs
        click_coordinates: str = Input(
            description="Click coordinates as '[x,y],[x,y],...'. Determines number of clicks."
        ),
        click_labels: str = Input(
            description="Click types (1=foreground, 0=background) as '1,1,0,1'. Auto-extends if shorter than coordinates.",
            default="1",
        ),
        click_frames: str = Input(
            description="Frame indices for clicks as '0,0,150,0'. Auto-extends if shorter than coordinates.",
            default="0",
        ),
        click_object_ids: str = Input(
            description="Object labels for clicks as 'person,dog,cat'. Auto-generates if missing or incomplete.",
            default="",
        ),
        # Output type
        mask_type: str = Input(
            default="binary",
            choices=["binary", "highlighted", "greenscreen"],
            description="Mask type: binary (B&W), highlighted (colored overlay), or greenscreen",
        ),
        # Output format
        output_video: bool = Input(
            default=False,
            description="True for video output, False for image sequence",
        ),
        # Video-specific options
        video_fps: int = Input(
            description="Video output frame rate (ignored for image sequence)",
            default=30,
            ge=1,
            le=60,
        ),
        # Image sequence-specific options
        output_format: str = Input(
            description="Image format for sequence (ignored for video)",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="JPG/WebP compression quality (0-100, ignored for PNG and video)",
            default=80,
            ge=0,
            le=100,
        ),
        # General output option
        output_frame_interval: int = Input(
            default=1,
            description="Output every Nth frame. 1=all frames, 2=every other, etc.",
        ),
    ) -> Iterator[Path]:
        # 1. Parse inputs
        click_list, click_labels_list, click_frames_list, object_ids_int_list = (
            self.parse_inputs(
                click_coordinates,
                click_labels,
                click_frames,
                click_object_ids,
            )
        )

        # 2. Create directories
        video_dir = Path("video_frames")
        video_dir.mkdir(exist_ok=True)
        output_dir = Path("predict_outputs")
        output_dir.mkdir(exist_ok=True)

        # 3. Extract video frames
        ffmpeg_command = (
            f"ffmpeg -i {input_video} -q:v 2 -start_number 0 {video_dir}/%05d.jpg"
        )
        subprocess.run(ffmpeg_command, shell=True, check=True)

        frame_names = sorted(
            [p for p in video_dir.glob("*.jpg")], key=lambda p: int(p.stem)
        )

        # 4. Initialize SAM predictor
        inference_state = self.predictor.init_state(video_path=str(video_dir))

        # 5. Process clicks and generate prompts
        prompts = {}
        for click, click_type, frame, obj_id in zip(
            click_list, click_labels_list, click_frames_list, object_ids_int_list
        ):
            x, y = click
            points = np.array([[x, y]], dtype=np.float32)
            labels = np.array([click_type], np.int32)

            if obj_id not in prompts:
                prompts[obj_id] = []
            prompts[obj_id].append((frame, points, labels))

        # 6. Perform segmentation
        video_segments = {}
        for obj_id, obj_prompts in prompts.items():
            for frame, points, labels in obj_prompts:
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=frame,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )

                if frame not in video_segments:
                    video_segments[frame] = {}
                video_segments[frame][obj_id] = out_mask_logits[0].cpu().numpy()

        # 7. Propagate masks
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.predictor.propagate_in_video(inference_state):
            if out_frame_idx not in video_segments:
                video_segments[out_frame_idx] = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                video_segments[out_frame_idx][out_obj_id] = (
                    out_mask_logits[i].cpu().numpy()
                )

        # 8. Generate and yield results
        is_video_output = output_video
        if is_video_output:
            output_format = "mp4"
        output_dir = Path("predict_outputs")
        output_dir.mkdir(exist_ok=True)

        if is_video_output:
            # Open the first frame to get its dimensions
            first_frame = Image.open(frame_names[0])
            frame_width, frame_height = first_frame.size

            video_output_path = output_dir / f"output_video.{output_format}"
            ffmpeg_command = (
                f"ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt rgb24 "
                f"-s {frame_width}x{frame_height} -r {video_fps} "
                f"-i - -c:v libx264 -pix_fmt yuv420p -preset fast -crf 23 {video_output_path}"
            )
            ffmpeg_process = subprocess.Popen(
                ffmpeg_command.split(), stdin=subprocess.PIPE, stderr=subprocess.PIPE
            )

        for out_frame_idx in range(0, len(frame_names), output_frame_interval):
            # Combine masks for all objects in the current frame
            combined_mask = np.zeros_like(
                next(iter(video_segments[out_frame_idx].values())).squeeze(),
                dtype=np.float32,
            )
            for out_mask in video_segments[out_frame_idx].values():
                combined_mask = np.maximum(combined_mask, out_mask.squeeze())

            # Apply threshold to get final binary mask
            final_mask = (combined_mask > 0.0).astype(np.uint8)

            if mask_type == "binary":
                output_image = Image.fromarray(final_mask * 255)
            elif mask_type == "highlighted":
                fig = plt.figure(figsize=(12, 8))
                plt.title(f"frame {out_frame_idx}")
                plt.imshow(Image.open(frame_names[out_frame_idx]))
                self.show_anns([final_mask], [1])
                buf = io.BytesIO()
                plt.savefig(
                    buf, format="png", dpi="figure", bbox_inches="tight", pad_inches=0
                )
                buf.seek(0)
                output_image = Image.open(buf)
                plt.close(fig)
            elif mask_type == "greenscreen":
                original_frame = Image.open(frame_names[out_frame_idx])
                mask_gray = Image.fromarray(final_mask * 255).convert("L")
                green_background = Image.new("RGB", original_frame.size, (0, 255, 0))
                output_image = Image.composite(
                    original_frame, green_background, mask_gray
                )

            if is_video_output:
                frame_data = np.array(output_image.convert("RGB"))
                ffmpeg_process.stdin.write(frame_data.tobytes())
            else:
                output_path = output_dir / f"frame_{out_frame_idx:05d}.{output_format}"
                self.save_image(
                    output_image, output_path, output_format, output_quality
                )
                yield output_path

        if is_video_output:
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
            yield video_output_path

        # Cleanup
        for file in video_dir.glob("*"):
            file.unlink()
        video_dir.rmdir()

    def save_image(self, image: Image.Image, path: Path, format: str, quality: int):
        save_params = {"format": format.upper()}
        if format.lower() != "png":
            save_params["quality"] = quality
            save_params["optimize"] = True
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, **save_params)
