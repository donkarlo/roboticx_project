import numpy
import matplotlib.pyplot as pyplot


class FmmSignal:
    def __init__(self, baseline: float, amplitude: float, alpha: float, time_values: numpy.ndarray) -> None:
        self.baseline = baseline
        self.amplitude = amplitude
        self.alpha = alpha
        self.time_values = time_values

    def compute_signal(self, beta: float, omega: float) -> numpy.ndarray:
        phase_values = beta + 2.0 * numpy.arctan(
            omega * numpy.tan((self.time_values - self.alpha) / 2.0)
        )
        signal_values = self.baseline + self.amplitude * numpy.cos(phase_values)
        return signal_values


class FmmBetaPlotter:
    def __init__(self) -> None:
        self.time_values = numpy.linspace(0.0, 2.0 * numpy.pi, 1000)
        self.beta_degrees = numpy.arange(0.0, 361.0, 15.0)
        self.beta_values = numpy.deg2rad(self.beta_degrees)
        self.omega = 1.0

        self.fmm_signal = FmmSignal(
            baseline=0.0,
            amplitude=1.0,
            alpha=0.0,
            time_values=self.time_values,
        )

    def plot(self) -> None:
        figure, axis = pyplot.subplots(figsize=(12, 7))

        for beta_degree, beta_value in zip(self.beta_degrees, self.beta_values):
            signal_values = self.fmm_signal.compute_signal(
                beta=beta_value,
                omega=self.omega,
            )
            axis.plot(
                self.time_values,
                signal_values,
                label=f"beta={int(beta_degree)}°",
            )

        axis.axhline(0.0, linewidth=0.8)
        axis.set_xlabel("t")
        axis.set_ylabel("v(t)")
        axis.set_title("FMM signal with fixed omega=1 and changing beta")
        axis.legend(ncol=3, fontsize=8)
        axis.grid(True)

        pyplot.show()


if __name__ == "__main__":
    plotter = FmmBetaPlotter()
    plotter.plot()