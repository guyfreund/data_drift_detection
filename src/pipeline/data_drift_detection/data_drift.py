class DataDrift:
    def __init__(self, is_drifted: bool):
        self._is_drifted = is_drifted

    @property
    def is_drifted(self):
        return self._is_drifted


class ModelBasedDataDrift(DataDrift):
    def __init__(self, is_drifted: bool):
        super().__init__(is_drifted=is_drifted)


class StatisticalBasedDataDrift(DataDrift):
    def __init__(self, is_drifted: bool):
        super().__init__(is_drifted=is_drifted)


class MeanDataDrift(DataDrift):
    def __init__(self, is_drifted: bool):
        super().__init__(is_drifted=is_drifted)


class VarianceDataDrift(DataDrift):
    def __init__(self, is_drifted: bool):
        super().__init__(is_drifted=is_drifted)


class NumNullsDataDrift(DataDrift):
    def __init__(self, is_drifted: bool):
        super().__init__(is_drifted=is_drifted)
