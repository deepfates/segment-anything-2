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

    def predict(
        self,
        input_video: Path = Input(description="Path to the input video file"),
        click_coordinates: str = Input(
            description="List of click coordinates in format '[x,y],[x,y],...'"
        ),
        click_labels: str = Input(
            description="List of click types (1 for foreground, 0 for background), e.g., '1,1,0,1'"
        ),
        click_frames: str = Input(
            description="List of frame indices for each click, e.g., '0,0,150,0'"
        ),
        click_object_ids: str = Input(
            description="List of object IDs for each click, e.g., '1,1,1,2'"
        ),
        output_frame_interval: int = Input(
            default=1, description="Interval for output frame visualization"
        ),
        mask_type: str = Input(
            default="binary",
            choices=["binary", "highlighted"],
            description="Choose the type of mask to return",
        ),
        output_format: str = Input(
            description="The image file format of the generated output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="The image compression quality", default=80, ge=0, le=100
        ),
    ) -> Iterator[Path]:
        # 1. Parse inputs
        click_list = [
            list(map(int, click.split(",")))
            for click in click_coordinates.strip("[]").split("],[")
        ]
        click_type_list = list(map(int, click_labels.split(",")))
        frame_list = list(map(int, click_frames.split(",")))
        obj_id_list = list(map(int, click_object_ids.split(",")))

        if not (
            len(click_list)
            == len(click_type_list)
            == len(frame_list)
            == len(obj_id_list)
        ):
            raise ValueError(
                "The number of clicks, click types, click frames, and object IDs must be the same."
            )

        # 2. Extract video frames
        video_dir = "video_frames"
        os.makedirs(video_dir, exist_ok=True)
        ffmpeg_command = (
            f"ffmpeg -i {input_video} -q:v 2 -start_number 0 {video_dir}/%05d.jpg"
        )
        subprocess.run(ffmpeg_command, shell=True, check=True)

        frame_names = sorted(
            [
                p
                for p in os.listdir(video_dir)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
            ],
            key=lambda p: int(os.path.splitext(p)[0]),
        )

        # 3. Initialize SAM predictor
        inference_state = self.predictor.init_state(video_path=video_dir)

        # 4. Process clicks and generate prompts
        prompts = {}
        for click, click_type, frame, obj_id in zip(
            click_list, click_type_list, frame_list, obj_id_list
        ):
            x, y = click
            points = np.array([[x, y]], dtype=np.float32)
            labels = np.array([click_type], np.int32)

            if obj_id not in prompts:
                prompts[obj_id] = []
            prompts[obj_id].append((frame, points, labels))

        # 5. Perform segmentation
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

        # 6. Propagate masks
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

        # 7. Generate and yield results
        output_dir = Path("predict_outputs")
        output_dir.mkdir(exist_ok=True)

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
                output_path = (
                    output_dir / f"binary_mask_{out_frame_idx:05d}.{output_format}"
                )
                mask_image = Image.fromarray(final_mask * 255)
                self.save_image(mask_image, output_path, output_format, output_quality)
                yield output_path

            elif mask_type == "highlighted":
                output_path = (
                    output_dir
                    / f"highlighted_frame_{out_frame_idx:05d}.{output_format}"
                )
                fig = plt.figure(figsize=(12, 8))
                plt.title(f"frame {out_frame_idx}")
                plt.imshow(
                    Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
                )
                self.show_anns([final_mask], [1])  # Use a single color for all objects

                buf = io.BytesIO()
                plt.savefig(
                    buf, format="png", dpi="figure", bbox_inches="tight", pad_inches=0
                )
                buf.seek(0)

                img = Image.open(buf)
                self.save_image(img, output_path, output_format, output_quality)

                plt.close(fig)
                yield output_path

        # 8. Cleanup
        for file in os.listdir(video_dir):
            os.remove(os.path.join(video_dir, file))
        os.rmdir(video_dir)

    def save_image(self, image: Image.Image, path: Path, format: str, quality: int):
        save_params = {"format": format.upper()}
        if format.lower() != "png":
            save_params["quality"] = quality
            save_params["optimize"] = True
        image.save(path, **save_params)
