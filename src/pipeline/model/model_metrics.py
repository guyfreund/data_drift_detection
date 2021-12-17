from src.pipeline.model.constants import ModelMetricType
from src.pipeline.model.interfaces.imodel_metric import IModelMetric


class Accuracy(IModelMetric):
    def __init__(self, value: float):
        self._value = value
        self._mtype = ModelMetricType.Accuracy

    @property
    def value(self):
        return self._value

    @property
    def mtype(self):
        return self._mtype


class Precision(IModelMetric):
    def __init__(self, value: float):
        self._value = value
        self._mtype = ModelMetricType.Precision

    @property
    def value(self):
        return self._value

    @property
    def mtype(self):
        return self._mtype


class Recall(IModelMetric):
    def __init__(self, value: float):
        self._value = value
        self._mtype = ModelMetricType.Recall

    @property
    def value(self):
        return self._value

    @property
    def mtype(self):
        return self._mtype


class F1(IModelMetric):
    def __init__(self, value: float):
        self._value = value
        self._mtype = ModelMetricType.F1

    @property
    def value(self):
        return self._value

    @property
    def mtype(self):
        return self._mtype


class AUC(IModelMetric):
    def __init__(self, value: float):
        self._value = value
        self._mtype = ModelMetricType.AUC

    @property
    def value(self):
        return self._value

    @property
    def mtype(self):
        return self._mtype
