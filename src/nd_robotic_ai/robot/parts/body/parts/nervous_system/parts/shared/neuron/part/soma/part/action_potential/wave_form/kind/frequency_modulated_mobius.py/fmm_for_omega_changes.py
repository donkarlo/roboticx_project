import numpy
import matplotlib.pyplot as pyplot


class FmmSignal:
    def __init__(self, baseline: float, amplitude: float, alpha: float, beta: float,
                 time_values: numpy.ndarray) -> None:
        self.baseline = baseline
        self.amplitude = amplitude
        self.alpha = alpha
        self.beta = beta
        self.time_values = time_values

    def compute_signal(self, omega: float) -> numpy.ndarray:
        phase_values = self.beta + 2.0 * numpy.arctan(
            omega * numpy.tan((self.time_values - self.alpha) / 2.0)
        )
        signal_values = self.baseline + self.amplitude * numpy.cos(phase_values)
        return signal_values


class FmmOmegaPlotter:
    def __init__(self) -> None:
        self.time_values = numpy.linspace(0.0, 2.0 * numpy.pi, 1000)
        self.omega_values = numpy.arange(0.0, 1.01, 0.1)

        self.fmm_signal = FmmSignal(
            baseline=0.0,
            amplitude=1.0,
            alpha=0.0,
            beta=-numpy.pi / 2.0,
            time_values=self.time_values,
        )

    def plot(self) -> None:
        figure, axis = pyplot.subplots(figsize=(10, 6))

        for omega in self.omega_values:
            signal_values = self.fmm_signal.compute_signal(omega)
            axis.plot(self.time_values, signal_values, label=f"omega={omega:.1f}")

        axis.axhline(0.0, linewidth=0.8)
        axis.set_xlabel("t")
        axis.set_ylabel("v(t)")
        axis.set_title("FMM signal with fixed parameters and changing omega")
        axis.legend()
        axis.grid(True)

        pyplot.show()


if __name__ == "__main__":
    plotter = FmmOmegaPlotter()
    plotter.plot()