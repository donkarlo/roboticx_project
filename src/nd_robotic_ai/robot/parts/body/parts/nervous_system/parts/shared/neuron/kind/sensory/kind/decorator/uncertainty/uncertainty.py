class Uncertainty:
    """
    check notability
    """
    def __init__(self):
        # uncertainty sources
        self._noise = None
        self._bias = None
        self._scale_factor_error = None
        self._none_linearity = None
        self._none_quantization = None
        self._latencey = None
        self._environmental_effects = None