import os
import sys
import time
import logging
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    def __init__(self, log_dir, name="med3dvlm"):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.file_logger = logging.getLogger(name)
        self.file_logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(log_dir, "train.log"))
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self.file_logger.addHandler(handler)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self.file_logger.addHandler(console)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag, tag_dict, step):
        self.writer.add_scalars(main_tag, tag_dict, step)

    def log_text(self, tag, text, step):
        self.writer.add_text(tag, text, step)

    def info(self, msg):
        self.file_logger.info(msg)

    def warning(self, msg):
        self.file_logger.warning(msg)

    def close(self):
        self.writer.close()


class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is not None:
            self.elapsed += time.time() - self.start_time
            self.start_time = None
        return self.elapsed

    def reset(self):
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
