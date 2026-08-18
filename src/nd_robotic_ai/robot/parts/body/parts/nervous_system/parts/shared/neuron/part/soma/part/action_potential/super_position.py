from __future__ import annotations

import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy
from scipy.optimize import differential_evolution
from scipy.optimize import least_squares


class FmmComponentParameters:
    def __init__(self, amplitude: float, alpha: float, beta: float, omega: float) -> None:
        self.amplitude = amplitude
        self.alpha = alpha
        self.beta = beta
        self.omega = omega

    def get_as_list(self) -> List[float]:
        return [
            self.amplitude,
            self.alpha,
            self.beta,
            self.omega
        ]

    def create_modified(self, amplitude_scale: float, alpha_shift: float, beta_shift: float,
                        omega_scale: float) -> FmmComponentParameters:
        shifted_alpha = self.alpha + alpha_shift
        shifted_beta = self.beta + beta_shift

        while shifted_alpha >= 2.0 * math.pi:
            shifted_alpha = shifted_alpha - 2.0 * math.pi

        while shifted_alpha < 0.0:
            shifted_alpha = shifted_alpha + 2.0 * math.pi

        while shifted_beta >= 2.0 * math.pi:
            shifted_beta = shifted_beta - 2.0 * math.pi

        while shifted_beta < 0.0:
            shifted_beta = shifted_beta + 2.0 * math.pi

        modified_omega = self.omega * omega_scale

        if modified_omega < 0.001:
            modified_omega = 0.001

        if modified_omega > 1.0:
            modified_omega = 1.0

        parameters = FmmComponentParameters(
            self.amplitude * amplitude_scale,
            shifted_alpha,
            shifted_beta,
            modified_omega
        )

        return parameters


class SingleFmmSignalParameters:
    def __init__(self, baseline: float, component_parameters: FmmComponentParameters) -> None:
        self.baseline = baseline
        self.component_parameters = component_parameters

    def get_as_list(self) -> List[float]:
        return [
            self.baseline,
            self.component_parameters.amplitude,
            self.component_parameters.alpha,
            self.component_parameters.beta,
            self.component_parameters.omega
        ]


class TwoComponentFmmSignalParameters:
    def __init__(self, baseline: float, first_component_parameters: FmmComponentParameters,
                 second_component_parameters: FmmComponentParameters) -> None:
        self.baseline = baseline
        self.first_component_parameters = first_component_parameters
        self.second_component_parameters = second_component_parameters


class FmmComponent:
    def __init__(self, parameters: FmmComponentParameters) -> None:
        self.parameters = parameters

    def evaluate(self, time_points: numpy.ndarray) -> numpy.ndarray:
        half_angle_difference = 0.5 * (time_points - self.parameters.alpha)

        phase = self.parameters.beta + 2.0 * numpy.arctan2(
            self.parameters.omega * numpy.sin(half_angle_difference),
            numpy.cos(half_angle_difference)
        )

        values = self.parameters.amplitude * numpy.cos(phase)
        return values


class SingleFmmSignal:
    def __init__(self, parameters: SingleFmmSignalParameters) -> None:
        self.parameters = parameters
        self.component = FmmComponent(parameters.component_parameters)

    def evaluate(self, time_points: numpy.ndarray) -> numpy.ndarray:
        values = self.parameters.baseline + self.component.evaluate(time_points)
        return values


class ActionPotentialLikeFmmSignal:
    def __init__(self, parameters: TwoComponentFmmSignalParameters, resting_potential: float,
                 reference_time: float) -> None:
        self.parameters = parameters
        self.resting_potential = resting_potential
        self.reference_time = reference_time
        self.first_component = FmmComponent(parameters.first_component_parameters)
        self.second_component = FmmComponent(parameters.second_component_parameters)
        self.resting_offset = self.create_resting_offset()

    def create_resting_offset(self) -> float:
        reference_time_points = numpy.array([self.reference_time], dtype=float)
        reference_value = self.evaluate_without_resting_correction(reference_time_points)[0]
        resting_offset = self.resting_potential - float(reference_value)
        return resting_offset

    def evaluate_without_resting_correction(self, time_points: numpy.ndarray) -> numpy.ndarray:
        values = self.parameters.baseline + self.first_component.evaluate(time_points) + self.second_component.evaluate(
            time_points)
        return values

    def evaluate(self, time_points: numpy.ndarray) -> numpy.ndarray:
        values = self.evaluate_without_resting_correction(time_points) + self.resting_offset
        return values


class WeightedCombinedActionPotentialSignal:
    def __init__(self, resting_potential: float, first_signal: ActionPotentialLikeFmmSignal,
                 second_signal: ActionPotentialLikeFmmSignal, first_weight: float, second_weight: float) -> None:
        self.resting_potential = resting_potential
        self.first_signal = first_signal
        self.second_signal = second_signal
        self.first_weight = first_weight
        self.second_weight = second_weight

    def evaluate_first_wave(self, time_points: numpy.ndarray) -> numpy.ndarray:
        values = self.first_signal.evaluate(time_points)
        return values

    def evaluate_second_wave(self, time_points: numpy.ndarray) -> numpy.ndarray:
        values = self.second_signal.evaluate(time_points)
        return values

    def evaluate(self, time_points: numpy.ndarray) -> numpy.ndarray:
        first_wave_values = self.first_signal.evaluate(time_points)
        second_wave_values = self.second_signal.evaluate(time_points)

        first_deviation = first_wave_values - self.resting_potential
        second_deviation = second_wave_values - self.resting_potential

        values = self.resting_potential + self.first_weight * first_deviation + self.second_weight * second_deviation
        return values


class ActionPotentialTemplateFactory:
    def create_template_parameters(self) -> TwoComponentFmmSignalParameters:
        first_component_parameters = FmmComponentParameters(
            53.03187484,
            5.06442008,
            2.40657480,
            0.10937460
        )

        second_component_parameters = FmmComponentParameters(
            34.65401532,
            5.39860485,
            5.39098286,
            0.13734080
        )

        parameters = TwoComponentFmmSignalParameters(
            -50.51165188,
            first_component_parameters,
            second_component_parameters
        )

        return parameters


class RandomActionPotentialParameterFactory:
    def __init__(self) -> None:
        self.random_generator = numpy.random.default_rng()
        self.template_factory = ActionPotentialTemplateFactory()

    def create_random_parameters(self) -> TwoComponentFmmSignalParameters:
        template_parameters = self.template_factory.create_template_parameters()

        baseline_shift = float(self.random_generator.uniform(-4.0, 4.0))

        first_amplitude_scale = float(self.random_generator.uniform(0.85, 1.15))
        first_alpha_shift = float(self.random_generator.uniform(-0.20, 0.20))
        first_beta_shift = float(self.random_generator.uniform(-0.20, 0.20))
        first_omega_scale = float(self.random_generator.uniform(0.85, 1.15))

        second_amplitude_scale = float(self.random_generator.uniform(0.85, 1.15))
        second_alpha_shift = float(self.random_generator.uniform(-0.20, 0.20))
        second_beta_shift = float(self.random_generator.uniform(-0.20, 0.20))
        second_omega_scale = float(self.random_generator.uniform(0.85, 1.15))

        first_component_parameters = template_parameters.first_component_parameters.create_modified(
            first_amplitude_scale,
            first_alpha_shift,
            first_beta_shift,
            first_omega_scale
        )

        second_component_parameters = template_parameters.second_component_parameters.create_modified(
            second_amplitude_scale,
            second_alpha_shift,
            second_beta_shift,
            second_omega_scale
        )

        parameters = TwoComponentFmmSignalParameters(
            template_parameters.baseline + baseline_shift,
            first_component_parameters,
            second_component_parameters
        )

        return parameters


class FmmParameterVectorFactory:
    def create_single_fmm_parameters_from_vector(self, parameter_vector: numpy.ndarray) -> SingleFmmSignalParameters:
        baseline = float(parameter_vector[0])
        amplitude = float(parameter_vector[1])
        alpha = float(parameter_vector[2])
        beta = float(parameter_vector[3])
        omega = float(parameter_vector[4])

        component_parameters = FmmComponentParameters(
            amplitude,
            alpha,
            beta,
            omega
        )

        parameters = SingleFmmSignalParameters(
            baseline,
            component_parameters
        )

        return parameters


class SingleFmmFitObjective:
    def __init__(self, time_points: numpy.ndarray, target_values: numpy.ndarray) -> None:
        self.time_points = time_points
        self.target_values = target_values
        self.parameter_vector_factory = FmmParameterVectorFactory()

    def compute_residuals(self, parameter_vector: numpy.ndarray) -> numpy.ndarray:
        parameters = self.parameter_vector_factory.create_single_fmm_parameters_from_vector(parameter_vector)
        signal = SingleFmmSignal(parameters)
        predicted_values = signal.evaluate(self.time_points)
        residuals = self.target_values - predicted_values
        return residuals

    def compute_sum_of_squared_errors(self, parameter_vector: numpy.ndarray) -> float:
        residuals = self.compute_residuals(parameter_vector)
        squared_errors = residuals ** 2
        sum_of_squared_errors = float(numpy.sum(squared_errors))
        return sum_of_squared_errors


class SingleFmmParameterBounds:
    def __init__(self, target_values: numpy.ndarray) -> None:
        target_minimum = float(numpy.min(target_values))
        target_maximum = float(numpy.max(target_values))
        signal_range = target_maximum - target_minimum

        if signal_range <= 0.0:
            signal_range = 1.0

        self.baseline_lower = target_minimum - signal_range
        self.baseline_upper = target_maximum + signal_range
        self.amplitude_lower = 0.0
        self.amplitude_upper = 2.0 * signal_range
        self.alpha_lower = 0.0
        self.alpha_upper = 2.0 * math.pi
        self.beta_lower = 0.0
        self.beta_upper = 2.0 * math.pi
        self.omega_lower = 0.001
        self.omega_upper = 1.0

    def get_differential_evolution_bounds(self) -> List[Tuple[float, float]]:
        bounds = [
            (self.baseline_lower, self.baseline_upper),
            (self.amplitude_lower, self.amplitude_upper),
            (self.alpha_lower, self.alpha_upper),
            (self.beta_lower, self.beta_upper),
            (self.omega_lower, self.omega_upper)
        ]

        return bounds

    def get_least_squares_bounds(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        lower_bounds = numpy.array(
            [
                self.baseline_lower,
                self.amplitude_lower,
                self.alpha_lower,
                self.beta_lower,
                self.omega_lower
            ],
            dtype=float
        )

        upper_bounds = numpy.array(
            [
                self.baseline_upper,
                self.amplitude_upper,
                self.alpha_upper,
                self.beta_upper,
                self.omega_upper
            ],
            dtype=float
        )

        return lower_bounds, upper_bounds


class SingleFmmFitResult:
    def __init__(self, parameters: SingleFmmSignalParameters, sum_of_squared_errors: float,
                 root_mean_squared_error: float, maximum_absolute_error: float) -> None:
        self.parameters = parameters
        self.sum_of_squared_errors = sum_of_squared_errors
        self.root_mean_squared_error = root_mean_squared_error
        self.maximum_absolute_error = maximum_absolute_error

    def create_signal(self) -> SingleFmmSignal:
        signal = SingleFmmSignal(self.parameters)
        return signal

    def print_summary(self) -> None:
        print("Best five parameters for one FMM wave:")
        print("M      =", self.parameters.baseline)
        print("A      =", self.parameters.component_parameters.amplitude)
        print("alpha  =", self.parameters.component_parameters.alpha)
        print("beta   =", self.parameters.component_parameters.beta)
        print("omega  =", self.parameters.component_parameters.omega)
        print("SSE    =", self.sum_of_squared_errors)
        print("RMSE   =", self.root_mean_squared_error)
        print("MAX AE =", self.maximum_absolute_error)


class SingleFmmWaveFitter:
    def __init__(self, maximum_iterations: int, random_seed: int) -> None:
        self.maximum_iterations = maximum_iterations
        self.random_seed = random_seed
        self.parameter_vector_factory = FmmParameterVectorFactory()

    def fit(self, time_points: numpy.ndarray, target_values: numpy.ndarray) -> SingleFmmFitResult:
        objective = SingleFmmFitObjective(time_points, target_values)
        bounds = SingleFmmParameterBounds(target_values)

        global_result = differential_evolution(
            objective.compute_sum_of_squared_errors,
            bounds.get_differential_evolution_bounds(),
            maxiter=self.maximum_iterations,
            seed=self.random_seed,
            polish=False
        )

        local_result = least_squares(
            objective.compute_residuals,
            global_result.x,
            bounds=bounds.get_least_squares_bounds(),
            max_nfev=5000
        )

        best_parameter_vector = local_result.x
        best_parameters = self.parameter_vector_factory.create_single_fmm_parameters_from_vector(best_parameter_vector)
        residuals = objective.compute_residuals(best_parameter_vector)

        best_error = float(numpy.sum(residuals ** 2))
        root_mean_squared_error = float(numpy.sqrt(numpy.mean(residuals ** 2)))
        maximum_absolute_error = float(numpy.max(numpy.abs(residuals)))

        result = SingleFmmFitResult(
            best_parameters,
            best_error,
            root_mean_squared_error,
            maximum_absolute_error
        )

        return result


class FmmPlotData:
    def __init__(self, time_points: numpy.ndarray, first_wave_values: numpy.ndarray, second_wave_values: numpy.ndarray,
                 combined_wave_values: numpy.ndarray, fitted_wave_values: numpy.ndarray, first_wave_parameter_text: str,
                 second_wave_parameter_text: str, combined_wave_parameter_text: str,
                 fitted_wave_parameter_text: str) -> None:
        self.time_points = time_points
        self.first_wave_values = first_wave_values
        self.second_wave_values = second_wave_values
        self.combined_wave_values = combined_wave_values
        self.fitted_wave_values = fitted_wave_values
        self.first_wave_parameter_text = first_wave_parameter_text
        self.second_wave_parameter_text = second_wave_parameter_text
        self.combined_wave_parameter_text = combined_wave_parameter_text
        self.fitted_wave_parameter_text = fitted_wave_parameter_text


class FmmWavePlotter:
    def plot(self, plot_data: FmmPlotData) -> None:
        figure, axes = plt.subplots(3, 1, figsize=(13, 14), sharex=True)

        axes[0].plot(plot_data.time_points, plot_data.first_wave_values)
        axes[0].set_title(
            "First random action-potential-like FMM wave\n" + plot_data.first_wave_parameter_text,
            fontsize=9
        )
        axes[0].set_ylabel("Membrane potential")
        axes[0].grid(True)

        axes[1].plot(plot_data.time_points, plot_data.second_wave_values)
        axes[1].set_title(
            "Second random action-potential-like FMM wave\n" + plot_data.second_wave_parameter_text,
            fontsize=9
        )
        axes[1].set_ylabel("Membrane potential")
        axes[1].grid(True)

        axes[2].plot(plot_data.time_points, plot_data.combined_wave_values,
                     label="Combined action-potential-like signal")
        axes[2].plot(plot_data.time_points, plot_data.fitted_wave_values, linestyle="--",
                     label="Fitted single five-parameter FMM wave")
        axes[2].set_title(
            "Combined signal and fitted single FMM wave\n"
            + plot_data.combined_wave_parameter_text
            + "\n"
            + plot_data.fitted_wave_parameter_text,
            fontsize=9
        )
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Membrane potential")
        axes[2].legend()
        axes[2].grid(True)

        figure.tight_layout()
        plt.show()


class ExampleApplication:
    def __init__(self) -> None:
        self.sample_count = 1000
        self.time_start = 0.0
        self.time_end = 2.0 * math.pi
        self.resting_potential = -70.0
        self.reference_time = 0.0
        self.first_weight = 0.55
        self.second_weight = 0.45
        self.random_parameter_factory = RandomActionPotentialParameterFactory()

    def create_time_points(self) -> numpy.ndarray:
        time_points = numpy.linspace(
            self.time_start,
            self.time_end,
            self.sample_count,
            endpoint=False
        )

        return time_points

    def create_combined_signal(self) -> WeightedCombinedActionPotentialSignal:
        first_parameters = self.random_parameter_factory.create_random_parameters()
        second_parameters = self.random_parameter_factory.create_random_parameters()

        first_signal = ActionPotentialLikeFmmSignal(
            first_parameters,
            self.resting_potential,
            self.reference_time
        )

        second_signal = ActionPotentialLikeFmmSignal(
            second_parameters,
            self.resting_potential,
            self.reference_time
        )

        combined_signal = WeightedCombinedActionPotentialSignal(
            self.resting_potential,
            first_signal,
            second_signal,
            self.first_weight,
            self.second_weight
        )

        return combined_signal

    def create_two_component_parameter_text(self, signal: ActionPotentialLikeFmmSignal) -> str:
        parameters = signal.parameters
        first_component = parameters.first_component_parameters
        second_component = parameters.second_component_parameters

        first_line = (
                "M="
                + str(round(parameters.baseline, 4))
                + ", A1="
                + str(round(first_component.amplitude, 4))
                + ", alpha1="
                + str(round(first_component.alpha, 4))
                + ", beta1="
                + str(round(first_component.beta, 4))
                + ", omega1="
                + str(round(first_component.omega, 4))
        )

        second_line = (
                "A2="
                + str(round(second_component.amplitude, 4))
                + ", alpha2="
                + str(round(second_component.alpha, 4))
                + ", beta2="
                + str(round(second_component.beta, 4))
                + ", omega2="
                + str(round(second_component.omega, 4))
        )

        parameter_text = first_line + "\n" + second_line
        return parameter_text

    def create_single_component_parameter_text(self, signal: SingleFmmSignal) -> str:
        parameters = signal.parameters
        component = parameters.component_parameters

        parameter_text = (
                "Fitted: M="
                + str(round(parameters.baseline, 4))
                + ", A="
                + str(round(component.amplitude, 4))
                + ", alpha="
                + str(round(component.alpha, 4))
                + ", beta="
                + str(round(component.beta, 4))
                + ", omega="
                + str(round(component.omega, 4))
        )

        return parameter_text

    def create_combined_parameter_text(self) -> str:
        parameter_text = (
                "Combined: resting_potential="
                + str(round(self.resting_potential, 4))
                + ", first_weight="
                + str(round(self.first_weight, 4))
                + ", second_weight="
                + str(round(self.second_weight, 4))
        )

        return parameter_text

    def print_signal_summary(self, signal_name: str, signal_values: numpy.ndarray) -> None:
        print(signal_name)
        print("minimum =", float(numpy.min(signal_values)))
        print("maximum =", float(numpy.max(signal_values)))
        print("initial =", float(signal_values[0]))

    def run(self) -> None:
        time_points = self.create_time_points()
        combined_signal = self.create_combined_signal()

        first_wave_values = combined_signal.evaluate_first_wave(time_points)
        second_wave_values = combined_signal.evaluate_second_wave(time_points)
        combined_wave_values = combined_signal.evaluate(time_points)

        self.print_signal_summary("First wave", first_wave_values)
        self.print_signal_summary("Second wave", second_wave_values)
        self.print_signal_summary("Combined wave", combined_wave_values)

        fitter = SingleFmmWaveFitter(
            500,
            42
        )

        fit_result = fitter.fit(time_points, combined_wave_values)
        fit_result.print_summary()

        fitted_single_signal = fit_result.create_signal()
        fitted_wave_values = fitted_single_signal.evaluate(time_points)

        first_wave_parameter_text = self.create_two_component_parameter_text(combined_signal.first_signal)
        second_wave_parameter_text = self.create_two_component_parameter_text(combined_signal.second_signal)
        combined_wave_parameter_text = self.create_combined_parameter_text()
        fitted_wave_parameter_text = self.create_single_component_parameter_text(fitted_single_signal)

        plot_data = FmmPlotData(
            time_points,
            first_wave_values,
            second_wave_values,
            combined_wave_values,
            fitted_wave_values,
            first_wave_parameter_text,
            second_wave_parameter_text,
            combined_wave_parameter_text,
            fitted_wave_parameter_text
        )

        plotter = FmmWavePlotter()
        plotter.plot(plot_data)


if __name__ == "__main__":
    application = ExampleApplication()
    application.run()