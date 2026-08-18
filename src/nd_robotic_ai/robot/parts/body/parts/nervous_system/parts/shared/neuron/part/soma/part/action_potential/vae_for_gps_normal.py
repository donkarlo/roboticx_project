from __future__ import annotations

import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from nd_sociomind.experiment.parts.oldest.uav1_300k_normal_time_position_modality import (
    Uav1300kNormalTimePositionModality,
)


class SamplingLayer(tf.keras.layers.Layer):
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        latent_mean, latent_log_variance = inputs
        noise = tf.random.normal(shape=tf.shape(latent_mean))
        latent_vector = latent_mean + tf.exp(0.5 * latent_log_variance) * noise

        return latent_vector


class GpsStandardizer:
    def __init__(self) -> None:
        self.mean_values = None
        self.standard_deviation_values = None

    def fit(self, gps_points: np.ndarray) -> None:
        self.mean_values = np.mean(gps_points, axis=0)
        self.standard_deviation_values = np.std(gps_points, axis=0)

        self.standard_deviation_values[self.standard_deviation_values < 1e-8] = 1.0

    def transform(self, gps_points: np.ndarray) -> np.ndarray:
        if self.mean_values is None:
            raise ValueError("The standardizer must be fitted before transformation.")

        if self.standard_deviation_values is None:
            raise ValueError("The standardizer must be fitted before transformation.")

        normalized_points = (gps_points - self.mean_values) / self.standard_deviation_values

        return normalized_points.astype(np.float32)

    def fit_transform(self, gps_points: np.ndarray) -> np.ndarray:
        self.fit(gps_points)
        normalized_points = self.transform(gps_points)

        return normalized_points

    def inverse_transform(self, normalized_points: np.ndarray) -> np.ndarray:
        if self.mean_values is None:
            raise ValueError("The standardizer must be fitted before inverse transformation.")

        if self.standard_deviation_values is None:
            raise ValueError("The standardizer must be fitted before inverse transformation.")

        gps_points = normalized_points * self.standard_deviation_values + self.mean_values

        return gps_points.astype(np.float32)


class GpsLoader:
    def load(self) -> np.ndarray:
        data_slice = slice(0, 50000)
        memory = Uav1300kNormalTimePositionModality(data_slice)
        data = memory.get_np_positions()

        gps_points = data[:, 0:3]

        valid_rows = ~np.isnan(gps_points).any(axis=1)
        gps_points = gps_points[valid_rows]

        valid_rows = ~np.isinf(gps_points).any(axis=1)
        gps_points = gps_points[valid_rows]

        if gps_points.shape[0] == 0:
            raise ValueError("No GPS rows remained after preprocessing.")

        return gps_points.astype(np.float32)


class GpsTrainValidationSplitter:
    def __init__(self, validation_fraction: float, random_seed: int) -> None:
        self.validation_fraction = validation_fraction
        self.random_seed = random_seed

    def split(self, gps_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.validation_fraction <= 0.0:
            raise ValueError("Validation fraction must be greater than 0.")

        if self.validation_fraction >= 1.0:
            raise ValueError("Validation fraction must be less than 1.")

        sample_count = len(gps_points)
        validation_count = int(sample_count * self.validation_fraction)

        if validation_count < 1:
            raise ValueError("Validation set is empty. Increase validation_fraction.")

        training_count = sample_count - validation_count

        if training_count < 1:
            raise ValueError("Training set is empty. Decrease validation_fraction.")

        random_generator = np.random.default_rng(self.random_seed)
        indices = np.arange(sample_count)
        random_generator.shuffle(indices)

        validation_indices = indices[:validation_count]
        training_indices = indices[validation_count:]

        training_points = gps_points[training_indices]
        validation_points = gps_points[validation_indices]

        return training_points.astype(np.float32), validation_points.astype(np.float32)


class Fmm1ParameterBounder(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()

        self.baseline_scale = tf.constant(30.0, dtype=tf.float32)

        self.amplitude_minimum = tf.constant(40.0, dtype=tf.float32)
        self.amplitude_maximum = tf.constant(80.0, dtype=tf.float32)

        self.alpha_minimum = tf.constant(6.00, dtype=tf.float32)
        self.alpha_maximum = tf.constant(2.0 * math.pi, dtype=tf.float32)

        self.beta_minimum = tf.constant(3.75, dtype=tf.float32)
        self.beta_maximum = tf.constant(4.25, dtype=tf.float32)

        self.omega_minimum = tf.constant(0.09, dtype=tf.float32)
        self.omega_maximum = tf.constant(0.18, dtype=tf.float32)

    def call(self, raw_parameters: tf.Tensor) -> tf.Tensor:
        raw_baseline = raw_parameters[:, 0:1]
        raw_amplitude = raw_parameters[:, 1:2]
        raw_alpha = raw_parameters[:, 2:3]
        raw_beta = raw_parameters[:, 3:4]
        raw_omega = raw_parameters[:, 4:5]

        baseline = self.baseline_scale * tf.tanh(raw_baseline)

        amplitude = self.amplitude_minimum + (
                self.amplitude_maximum - self.amplitude_minimum
        ) * tf.sigmoid(raw_amplitude)

        alpha = self.alpha_minimum + (
                self.alpha_maximum - self.alpha_minimum
        ) * tf.sigmoid(raw_alpha)

        beta = self.beta_minimum + (
                self.beta_maximum - self.beta_minimum
        ) * tf.sigmoid(raw_beta)

        omega = self.omega_minimum + (
                self.omega_maximum - self.omega_minimum
        ) * tf.sigmoid(raw_omega)

        parameters = tf.concat(
            [
                baseline,
                amplitude,
                alpha,
                beta,
                omega,
            ],
            axis=1,
        )

        return parameters


class Fmm1WaveLayer(tf.keras.layers.Layer):
    def __init__(self, time_points: np.ndarray) -> None:
        super().__init__()
        self.time_points = tf.constant(time_points.astype(np.float32), dtype=tf.float32)

    def call(self, parameters: tf.Tensor) -> tf.Tensor:
        baseline = parameters[:, 0:1]
        amplitude = parameters[:, 1:2]
        alpha = parameters[:, 2:3]
        beta = parameters[:, 3:4]
        omega = parameters[:, 4:5]

        half_angle_difference = 0.5 * (self.time_points[tf.newaxis, :] - alpha)

        phase = beta + 2.0 * tf.atan2(
            omega * tf.sin(half_angle_difference),
            tf.cos(half_angle_difference),
        )

        values = baseline + amplitude * tf.cos(phase)
        values = values - values[:, 0:1]

        return values


class GpsToFmm1VariationalAutoencoder(tf.keras.Model):
    def __init__(self, time_points: np.ndarray) -> None:
        super().__init__()

        self.encoder_network = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(3,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
        ])

        self.latent_mean_layer = tf.keras.layers.Dense(5)
        self.latent_log_variance_layer = tf.keras.layers.Dense(5)

        self.sampling_layer = SamplingLayer()

        self.decoder_network = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(5,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(3),
        ])

        self.parameter_bounder = Fmm1ParameterBounder()
        self.wave_layer = Fmm1WaveLayer(time_points)

    def encode(self, gps_points: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        hidden_values = self.encoder_network(gps_points, training=training)

        latent_mean = self.latent_mean_layer(hidden_values)
        latent_log_variance = self.latent_log_variance_layer(hidden_values)

        latent_log_variance = tf.clip_by_value(
            latent_log_variance,
            -6.0,
            6.0,
        )

        return latent_mean, latent_log_variance

    def call(self, gps_points: tf.Tensor, training: bool = False) -> Tuple[
        tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        latent_mean, latent_log_variance = self.encode(
            gps_points,
            training=training,
        )

        if training:
            latent_vector = self.sampling_layer(
                (
                    latent_mean,
                    latent_log_variance,
                )
            )
        else:
            latent_vector = latent_mean

        reconstructed_gps_points = self.decoder_network(
            latent_vector,
            training=training,
        )

        fmm_parameters = self.parameter_bounder(latent_vector)
        fmm_waves = self.wave_layer(fmm_parameters)

        return reconstructed_gps_points, latent_mean, latent_log_variance, latent_vector, fmm_parameters, fmm_waves


class Trainer:
    def __init__(self, model: GpsToFmm1VariationalAutoencoder, learning_rate: float, kl_weight: float) -> None:
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.kl_weight = kl_weight

    def calculate_losses(self, gps_points: tf.Tensor, training: bool) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        reconstructed_gps_points, latent_mean, latent_log_variance, latent_vector, fmm_parameters, fmm_waves = self.model(
            gps_points,
            training=training,
        )

        reconstruction_loss = tf.reduce_mean(
            tf.square(gps_points - reconstructed_gps_points)
        )

        kl_loss = -0.5 * tf.reduce_mean(
            1.0
            + latent_log_variance
            - tf.square(latent_mean)
            - tf.exp(latent_log_variance)
        )

        total_loss = reconstruction_loss + self.kl_weight * kl_loss

        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, gps_points: tf.Tensor) -> Tuple[float, float, float]:
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss = self.calculate_losses(
                gps_points=gps_points,
                training=True,
            )

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return float(total_loss), float(reconstruction_loss), float(kl_loss)

    def validation_step(self, gps_points: tf.Tensor) -> Tuple[float, float, float]:
        total_loss, reconstruction_loss, kl_loss = self.calculate_losses(
            gps_points=gps_points,
            training=False,
        )

        return float(total_loss), float(reconstruction_loss), float(kl_loss)

    def create_dataset(self, gps_points: np.ndarray, batch_size: int, shuffle: bool) -> tf.data.Dataset:
        if len(gps_points) == 0:
            raise ValueError("GPS dataset is empty.")

        dataset = tf.data.Dataset.from_tensor_slices(gps_points.astype(np.float32))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(gps_points))

        dataset = dataset.batch(batch_size)

        return dataset

    def train(self, training_gps_points: np.ndarray, validation_gps_points: np.ndarray, epoch_count: int,
              batch_size: int) -> None:
        training_dataset = self.create_dataset(
            gps_points=training_gps_points,
            batch_size=batch_size,
            shuffle=True,
        )

        validation_dataset = self.create_dataset(
            gps_points=validation_gps_points,
            batch_size=batch_size,
            shuffle=False,
        )

        for epoch_index in range(epoch_count):
            training_total_losses = []
            training_reconstruction_losses = []
            training_kl_losses = []

            validation_total_losses = []
            validation_reconstruction_losses = []
            validation_kl_losses = []

            for batch_points in training_dataset:
                total_loss, reconstruction_loss, kl_loss = self.train_step(batch_points)

                training_total_losses.append(total_loss)
                training_reconstruction_losses.append(reconstruction_loss)
                training_kl_losses.append(kl_loss)

            for batch_points in validation_dataset:
                total_loss, reconstruction_loss, kl_loss = self.validation_step(batch_points)

                validation_total_losses.append(total_loss)
                validation_reconstruction_losses.append(reconstruction_loss)
                validation_kl_losses.append(kl_loss)

            print(
                "epoch =",
                epoch_index + 1,
                "train_total_loss =",
                float(np.mean(training_total_losses)),
                "train_gps_reconstruction_mse =",
                float(np.mean(training_reconstruction_losses)),
                "train_kl =",
                float(np.mean(training_kl_losses)),
                "validation_total_loss =",
                float(np.mean(validation_total_losses)),
                "validation_gps_reconstruction_mse =",
                float(np.mean(validation_reconstruction_losses)),
                "validation_kl =",
                float(np.mean(validation_kl_losses)),
            )


class NumpyFmm1WaveFactory:
    def create_raw_wave(self, time_points: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        baseline = float(parameters[0])
        amplitude = float(parameters[1])
        alpha = float(parameters[2])
        beta = float(parameters[3])
        omega = float(parameters[4])

        half_angle_difference = 0.5 * (time_points - alpha)

        phase = beta + 2.0 * np.arctan2(
            omega * np.sin(half_angle_difference),
            np.cos(half_angle_difference),
        )

        values = baseline + amplitude * np.cos(phase)

        return values

    def create_display_wave(self, time_points: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        raw_values = self.create_raw_wave(time_points, parameters)
        display_values = raw_values - raw_values[0]

        return display_values


class PlotTitleFactory:
    def create_title(self, gps_point: np.ndarray, parameters: np.ndarray, index: int) -> str:
        x_value = float(gps_point[0])
        y_value = float(gps_point[1])
        z_value = float(gps_point[2])

        baseline = float(parameters[0])
        amplitude = float(parameters[1])
        alpha = float(parameters[2])
        beta = float(parameters[3])
        omega = float(parameters[4])

        title = (
                "sample "
                + str(index)
                + "\n"
                + "x="
                + str(round(x_value, 2))
                + ", y="
                + str(round(y_value, 2))
                + ", z="
                + str(round(z_value, 2))
                + "\n"
                + "M="
                + str(round(baseline, 2))
                + ", A="
                + str(round(amplitude, 2))
                + "\n"
                + "alpha="
                + str(round(alpha, 3))
                + ", beta="
                + str(round(beta, 3))
                + ", omega="
                + str(round(omega, 3))
        )

        return title


class WaveShapeReporter:
    def report(self, waves: np.ndarray) -> None:
        action_potential_like_count = 0

        for wave_index in range(waves.shape[0]):
            wave = waves[wave_index]

            peak_index = int(np.argmax(wave))
            trough_index = int(np.argmin(wave))

            peak_value = float(wave[peak_index])
            trough_value = float(wave[trough_index])

            is_action_potential_like = False

            if peak_value > 0.0:
                if trough_value < 0.0:
                    if peak_index < trough_index:
                        is_action_potential_like = True

            if is_action_potential_like:
                action_potential_like_count = action_potential_like_count + 1

        print("")
        print("Action-potential-like diagnostic")
        print("count =", action_potential_like_count)
        print("total =", int(waves.shape[0]))


class Plotter:
    def __init__(self, plot_time_end: float) -> None:
        self.plot_time_end = plot_time_end
        self.title_factory = PlotTitleFactory()
        self.wave_factory = NumpyFmm1WaveFactory()

    def plot_fmm_waves_as_table(self, gps_points: np.ndarray, parameters: np.ndarray, wave_count: int) -> None:
        time_points = np.linspace(
            0.0,
            self.plot_time_end,
            1000,
            endpoint=False,
        )

        selected_count = min(wave_count, len(parameters))

        figure = plt.figure(figsize=(22, 9))

        for index in range(selected_count):
            axes = figure.add_subplot(2, 5, index + 1)

            wave = self.wave_factory.create_display_wave(
                time_points,
                parameters[index],
            )

            title = self.title_factory.create_title(
                gps_points[index],
                parameters[index],
                index,
            )

            axes.plot(time_points, wave)
            axes.axhline(0.0, linestyle="--")
            axes.set_title(title, fontsize=8)
            axes.set_xlabel("internal time")
            axes.set_ylabel("V(t)")
            axes.grid(True)

        figure.tight_layout()
        plt.show()

    def print_parameter_ranges(self, parameters: np.ndarray) -> None:
        print("")
        print("First FMM1 parameters")
        print("M, A, alpha, beta, omega")
        print(parameters[0])

        print("")
        print("FMM1 parameter ranges")
        print("M min/max     =", float(np.min(parameters[:, 0])), float(np.max(parameters[:, 0])))
        print("A min/max     =", float(np.min(parameters[:, 1])), float(np.max(parameters[:, 1])))
        print("alpha min/max =", float(np.min(parameters[:, 2])), float(np.max(parameters[:, 2])))
        print("beta min/max  =", float(np.min(parameters[:, 3])), float(np.max(parameters[:, 3])))
        print("omega min/max =", float(np.min(parameters[:, 4])), float(np.max(parameters[:, 4])))


class Application:
    def __init__(self, epoch_count: int, batch_size: int, learning_rate: float, kl_weight: float, wave_count: int,
                 plot_time_end: float, time_point_count: int, validation_fraction: float, random_seed: int) -> None:
        self.epoch_count = epoch_count
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.wave_count = wave_count
        self.plot_time_end = plot_time_end
        self.time_point_count = time_point_count
        self.validation_fraction = validation_fraction
        self.random_seed = random_seed

    def run(self) -> None:
        gps_loader = GpsLoader()
        gps_points = gps_loader.load()

        splitter = GpsTrainValidationSplitter(
            validation_fraction=self.validation_fraction,
            random_seed=self.random_seed,
        )

        training_gps_points, validation_gps_points = splitter.split(gps_points)

        standardizer = GpsStandardizer()
        normalized_training_gps_points = standardizer.fit_transform(training_gps_points)
        normalized_validation_gps_points = standardizer.transform(validation_gps_points)
        normalized_all_gps_points = standardizer.transform(gps_points)

        print("GPS points shape =", gps_points.shape)
        print("Training GPS points shape =", training_gps_points.shape)
        print("Validation GPS points shape =", validation_gps_points.shape)
        print("Normalized training GPS points shape =", normalized_training_gps_points.shape)
        print("Normalized validation GPS points shape =", normalized_validation_gps_points.shape)

        time_points = np.linspace(
            0.0,
            self.plot_time_end,
            self.time_point_count,
            endpoint=False,
        ).astype(np.float32)

        model = GpsToFmm1VariationalAutoencoder(
            time_points=time_points,
        )

        trainer = Trainer(
            model=model,
            learning_rate=self.learning_rate,
            kl_weight=self.kl_weight,
        )

        trainer.train(
            training_gps_points=normalized_training_gps_points,
            validation_gps_points=normalized_validation_gps_points,
            epoch_count=self.epoch_count,
            batch_size=self.batch_size,
        )

        all_tensor = tf.convert_to_tensor(
            normalized_all_gps_points.astype(np.float32)
        )

        reconstructed_gps_points, latent_mean, latent_log_variance, latent_vector, fmm_parameters, fmm_waves = model(
            all_tensor,
            training=False,
        )

        all_parameters = fmm_parameters.numpy()
        all_waves = fmm_waves.numpy()

        selected_indices = np.linspace(
            0,
            len(normalized_all_gps_points) - 1,
            self.wave_count,
            dtype=int,
        )

        selected_gps_points = gps_points[selected_indices]
        selected_parameters = all_parameters[selected_indices]

        print("")
        print("Selected GPS points")
        print(selected_gps_points)

        print("")
        print("Selected FMM1 parameters")
        print(selected_parameters)

        reporter = WaveShapeReporter()
        reporter.report(
            waves=all_waves,
        )

        plotter = Plotter(
            plot_time_end=self.plot_time_end,
        )

        plotter.print_parameter_ranges(all_parameters)

        plotter.plot_fmm_waves_as_table(
            gps_points=selected_gps_points,
            parameters=selected_parameters,
            wave_count=self.wave_count,
        )


if __name__ == "__main__":
    epoch_count = 15
    batch_size = 2048
    learning_rate = 0.001
    kl_weight = 0.0001
    wave_count = 10
    plot_time_end = 2.0 * math.pi
    time_point_count = 1000
    validation_fraction = 0.2
    random_seed = 42

    application = Application(
        epoch_count=epoch_count,
        batch_size=batch_size,
        learning_rate=learning_rate,
        kl_weight=kl_weight,
        wave_count=wave_count,
        plot_time_end=plot_time_end,
        time_point_count=time_point_count,
        validation_fraction=validation_fraction,
        random_seed=random_seed,
    )

    application.run()