import numpy as np
import logging
import yaml
import time
import traceback
import cv2
# from dlclive import DLCLive
from pathlib import Path
from improv.actor import Actor
from collections import deque
# from .dlcProcessor import IndexAngles
# from deeplabcut.pose_estimation_pytorch import Task
# from deeplabcut.pose_estimation_pytorch.apis.analyze_videos import video_inference
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
log_file = "processor.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

class Processor:
    """ Applying DLC inference to each video frame
    """


    def setup(self):
        """Initializes all class variables."""

    
        logger.info("Beginning setup for Processor")

        '''# load the configuration file
        source_folder = Path(__file__).resolve().parent.parent

        with open(f'{source_folder}/config.yaml', 'r') as file:
            config = yaml.safe_load(file)'''


        train_dir = Path("/home/lucyliuu/deeplabcut_projects/nocover-lucy-2025-07-14/dlc-models-pytorch/iteration-1/nocoverJul14-trainset90shuffle2/train")
        pytorch_config_path = train_dir / "pytorch_config.yaml"
        snapshot_path = train_dir / "snapshot-best-290.pt"

        # for top-down models, otherwise None
        detector_snapshot_path = None

        # video and inference parameters
        max_num_animals = 1
        batch_size = 16
        detector_batch_size = 8

        # read model configuration
        model_cfg = read_config_as_dict(pytorch_config_path)

        self.pose_runner, detector_runner = get_inference_runners(
            model_config=model_cfg,
            snapshot_path=snapshot_path,
            max_individuals=max_num_animals,
            batch_size=batch_size,
            detector_batch_size=detector_batch_size,
            detector_path=detector_snapshot_path,
        )

    def process_video(self, video_path):
        """Run real-time inference on a video using OpenCV instead of improv queues."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Could not open video.")
            return

        self.frame_num = 0
        self.frames_log = 10
        self.time_start = time.perf_counter()
        self.predictions = []
        self.latencies = []
        self.dlc_latencies = []
        self.grab_latencies = []
        self.put_latencies = []

        logger.info("Starting video frame-by-frame processing...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.perf_counter()

            # Run DLC inference
            dlc_start = time.perf_counter()
            self.prediction = self.pose_runner.inference([frame])
            self.predictions.append(self.prediction)
            dlc_end = time.perf_counter()

            self.frame_num += 1
            self.dlc_latencies.append(dlc_end - dlc_start)
            self.grab_latencies.append(dlc_end - start_time)

            # Logging
            if self.frame_num % self.frames_log == 0:
                total_time = dlc_end - self.time_start
                logger.info(f"Frame number: {self.frame_num}")
                logger.info(f"Overall Average FPS: {round(self.frames_log / total_time,2)}")
                logger.info(f'DLC Inference Time Avg latency: {np.mean(self.dlc_latencies):.4f}')
                logger.info(f'Grab Time Avg latency: {np.mean(self.grab_latencies):.4f}')
                self.time_start = time.perf_counter()

    
        cap.release()

if __name__ == "__main__":
    processor = Processor()
    processor.setup()

    video_path = "/home/lucyliuu/deeplabcut_projects/nocover-lucy-2025-07-14/videos/video_9.mp4"
    processor.process_video(video_path)
