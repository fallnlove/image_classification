import numpy as np

from src.metrics.tracker import MetricsTracker


class WanDBWriter:
    def __init__(self, project_name: str, run_name: str = None, run_id: str = None):
        """
        Initialize the WanDBWriter

        Input:
            project_name (str): Name of the project
            run_name (str): Name of the run
            run_id (str): ID of the run
        """
        try:
            import wandb

            wandb.init(project=project_name, name=run_name, id=run_id, resume="allow")

            self.wandb = wandb

        except ImportError:
            raise ImportError("Please install wandb to use this writer")

        self.step = 0
        self.mode = ""

    def step(self):
        """
        Step the writer
        """

        self.step += 1

    def train(self):
        """
        Set the writer to train mode
        """

        self.mode = "train"

    def eval(self):
        """
        Set the writer to eval mode
        """

        self.mode = "eval"

    def log_scalar(self, name: str, value: float):
        """
        Log a scalar value

        Input:
            name (str): Name of the scalar
            value (float): Value of the scalar
        """

        self.wandb.log({f"{name}_{self.mode}": value}, step=self.step)

    def log_image(self, name: str, image: np.ndarray):
        """
        Log an image

        Input:
            name (str): Name of the image
            image (np.ndarray): Image to log
        """

        self.wandb.log({f"{name}_{self.mode}": self.wandb.Image(image)}, step=self.step)

    def log_metrics(self, metrics: MetricsTracker):
        """
        Log metrics

        Input:
            metrics (MetricsTracker): Metrics to log
        """

        for key in metrics.keys():
            self.log_scalar(key, metrics.get(key))
