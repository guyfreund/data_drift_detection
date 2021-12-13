import argparse

from src.pipeline.interfaces.imanager import IManager
from src.pipeline.data_generation.data_generation_manager import DataGenerationManager
from src.pipeline.data_drift_detection.data_drift_detection_manager import DataDriftDetectionManager
from src.pipeline.model.model_trainining_manager import ModelTrainingManager
from src.pipeline.constants import PipelineMode


class PipelineManager(IManager):
    def __init__(self, pipeline_mode: PipelineMode):
        self._mode = pipeline_mode
        self._data_generation_manager = DataGenerationManager()
        self._data_drift_detection_manager = DataDriftDetectionManager()
        self._model_training_manager = ModelTrainingManager()

    def manage(self):
        if self._mode == PipelineMode.Training:
            self._model_training_manager.manage()
        else:
            self._data_generation_manager.manage()
            self._data_drift_detection_manager.manage()


def args_handler():
    parser = argparse.ArgumentParser(description='Running the pipeline manager')
    parser.add_argument('-m', '--mode', default=PipelineMode.Training, type=int, help='Pipeline mode: 0=Training, 1=Monitoring')
    return parser.parse_args()


if __name__ == '__main__':
    args = args_handler()
    mode = args.mode
    pipeline_manager = PipelineManager(pipeline_mode=mode)
    pipeline_manager.manage()
