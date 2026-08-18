from __future__ import annotations

import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from nd_sociomind.experiment.parts.oldest.uav1_normal_lidar_time_ranges_modality import (
    Uav1NormalLidarTimeRangesModality,
)


class SamplingLayer(tf.keras.layers.Layer):
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        latent_mean, latent_log_variance = inputs
        noise = tf.random.normal(shape=tf.shape(latent_mean))
        latent_vector = latent_mean + tf.exp(0.5 * latent_log_variance) * noise

        return latent_vector


class LidarStandardizer:
    def __init__(self) -> None:
        self.mean_values = None
        self.standard_deviation_values = None

    def fit(self, lidar_ranges: np.ndarray) -> None:
        self.mean_values = np.mean(lidar_ranges, axis=0)
        self.standard_deviation_values = np.std(lidar_ranges, axis=0)

        self.standard_deviation_values[self.standard_deviation_values < 1e-8] = 1.0

    def transform(self, lidar_ranges: np.ndarray) -> np.ndarray:
        if self.mean_values is None:
            raise ValueError("The standardizer must be fitted before transformation.")

        if self.standard_deviation_values is None:
            raise ValueError("The standardizer must be fitted before transformation.")

        normalized_ranges = (lidar_ranges - self.mean_values) / self.standard_deviation_values

        return normalized_ranges.astype(np.float32)

    def fit_transform(self, lidar_ranges: np.ndarray) -> np.ndarray:
        self.fit(lidar_ranges)
        normalized_ranges = self.transform(lidar_ranges)

        return normalized_ranges

    def inverse_transform(self, normalized_ranges: np.ndarray) -> np.ndarray:
        if self.mean_values is None:
            raise ValueError("The standardizer must be fitted before inverse transformation.")

        if self.standard_deviation_values is None:
            raise ValueError("The standardizer must be fitted before inverse transformation.")

        lidar_ranges = normalized_ranges * self.standard_deviation_values + self.mean_values

        return lidar_ranges.astype(np.float32)


class LidarLoader:
    def __init__(self, maximum_range: float) -> None:
        self.maximum_range = maximum_range

    def load(self) -> np.ndarray:
        data_slice = slice(0, 50000)
        memory = Uav1NormalLidarTimeRangesModality(data_slice)

        data = memory.get_np_time_ranges()

        print("Raw LiDAR data shape =", data.shape)

        if data.shape[1] == 720:
            lidar_ranges = data
        elif data.shape[1] > 720:
            lidar_ranges = data[:, -720:]
        else:
            raise ValueError("LiDAR data must have at least 720 columns.")

        lidar_ranges = np.nan_to_num(
            lidar_ranges,
            nan=self.maximum_range,
            posinf=self.maximum_range,
            neginf=0.0,
        )

        lidar_ranges[lidar_ranges < 0.0] = 0.0
        lidar_ranges[lidar_ranges > self.maximum_range] = self.maximum_range

        if lidar_ranges.shape[0] == 0:
            raise ValueError("No LiDAR rows remained after preprocessing.")

        return lidar_ranges.astype(np.float32)


class LidarTrainValidationSplitter:
    def __init__(self, validation_fraction: float, random_seed: int) -> None:
        self.validation_fraction = validation_fraction
        self.random_seed = random_seed

    def split(self, lidar_ranges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.validation_fraction <= 0.0:
            raise ValueError("Validation fraction must be greater than 0.")

        if self.validation_fraction >= 1.0:
            raise ValueError("Validation fraction must be less than 1.")

        sample_count = len(lidar_ranges)
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

        training_ranges = lidar_ranges[training_indices]
        validation_ranges = lidar_ranges[validation_indices]

        return training_ranges.astype(np.float32), validation_ranges.astype(np.float32)


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


class LidarToFmm1VariationalAutoencoder(tf.keras.Model):
    def __init__(self, time_points: np.ndarray) -> None:
        super().__init__()

        self.encoder_network = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(720,)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
        ])

        self.latent_mean_layer = tf.keras.layers.Dense(5)
        self.latent_log_variance_layer = tf.keras.layers.Dense(5)

        self.sampling_layer = SamplingLayer()

        self.decoder_network = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(5,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(720),
        ])

        self.parameter_bounder = Fmm1ParameterBounder()
        self.wave_layer = Fmm1WaveLayer(time_points)

    def encode(self, lidar_ranges: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        hidden_values = self.encoder_network(lidar_ranges, training=training)

        latent_mean = self.latent_mean_layer(hidden_values)
        latent_log_variance = self.latent_log_variance_layer(hidden_values)

        latent_log_variance = tf.clip_by_value(
            latent_log_variance,
            -6.0,
            6.0,
        )

        return latent_mean, latent_log_variance

    def call(self, lidar_ranges: tf.Tensor, training: bool = False) -> Tuple[
        tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        latent_mean, latent_log_variance = self.encode(
            lidar_ranges,
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

        reconstructed_lidar_ranges = self.decoder_network(
            latent_vector,
            training=training,
        )

        fmm_parameters = self.parameter_bounder(latent_vector)
        fmm_waves = self.wave_layer(fmm_parameters)

        return reconstructed_lidar_ranges, latent_mean, latent_log_variance, latent_vector, fmm_parameters, fmm_waves


class Trainer:
    def __init__(self, model: LidarToFmm1VariationalAutoencoder, learning_rate: float, kl_weight: float) -> None:
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.kl_weight = kl_weight

    def calculate_losses(self, lidar_ranges: tf.Tensor, training: bool) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        reconstructed_lidar_ranges, latent_mean, latent_log_variance, latent_vector, fmm_parameters, fmm_waves = self.model(
            lidar_ranges,
            training=training,
        )

        reconstruction_loss = tf.reduce_mean(
            tf.square(lidar_ranges - reconstructed_lidar_ranges)
        )

        kl_loss = -0.5 * tf.reduce_mean(
            1.0
            + latent_log_variance
            - tf.square(latent_mean)
            - tf.exp(latent_log_variance)
        )

        total_loss = reconstruction_loss + self.kl_weight * kl_loss

        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, lidar_ranges: tf.Tensor) -> Tuple[float, float, float]:
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss = self.calculate_losses(
                lidar_ranges=lidar_ranges,
                training=True,
            )

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return float(total_loss), float(reconstruction_loss), float(kl_loss)

    def validation_step(self, lidar_ranges: tf.Tensor) -> Tuple[float, float, float]:
        total_loss, reconstruction_loss, kl_loss = self.calculate_losses(
            lidar_ranges=lidar_ranges,
            training=False,
        )

        return float(total_loss), float(reconstruction_loss), float(kl_loss)

    def create_dataset(self, lidar_ranges: np.ndarray, batch_size: int, shuffle: bool) -> tf.data.Dataset:
        if len(lidar_ranges) == 0:
            raise ValueError("LiDAR dataset is empty.")

        dataset = tf.data.Dataset.from_tensor_slices(lidar_ranges.astype(np.float32))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(lidar_ranges))

        dataset = dataset.batch(batch_size)

        return dataset

    def train(self, training_lidar_ranges: np.ndarray, validation_lidar_ranges: np.ndarray, epoch_count: int,
              batch_size: int) -> None:
        training_dataset = self.create_dataset(
            lidar_ranges=training_lidar_ranges,
            batch_size=batch_size,
            shuffle=True,
        )

        validation_dataset = self.create_dataset(
            lidar_ranges=validation_lidar_ranges,
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

            for batch_ranges in training_dataset:
                total_loss, reconstruction_loss, kl_loss = self.train_step(batch_ranges)

                training_total_losses.append(total_loss)
                training_reconstruction_losses.append(reconstruction_loss)
                training_kl_losses.append(kl_loss)

            for batch_ranges in validation_dataset:
                total_loss, reconstruction_loss, kl_loss = self.validation_step(batch_ranges)

                validation_total_losses.append(total_loss)
                validation_reconstruction_losses.append(reconstruction_loss)
                validation_kl_losses.append(kl_loss)

            print(
                "epoch =",
                epoch_index + 1,
                "train_total_loss =",
                float(np.mean(training_total_losses)),
                "train_lidar_reconstruction_mse =",
                float(np.mean(training_reconstruction_losses)),
                "train_kl =",
                float(np.mean(training_kl_losses)),
                "validation_total_loss =",
                float(np.mean(validation_total_losses)),
                "validation_lidar_reconstruction_mse =",
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
    def create_title(self, lidar_vector: np.ndarray, parameters: np.ndarray, index: int) -> str:
        minimum_range = float(np.min(lidar_vector))
        maximum_range = float(np.max(lidar_vector))
        mean_range = float(np.mean(lidar_vector))

        baseline = float(parameters[0])
        amplitude = float(parameters[1])
        alpha = float(parameters[2])
        beta = float(parameters[3])
        omega = float(parameters[4])

        title = (
                "sample "
                + str(index)
                + "\n"
                + "min="
                + str(round(minimum_range, 2))
                + ", max="
                + str(round(maximum_range, 2))
                + ", mean="
                + str(round(mean_range, 2))
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

    def plot_fmm_waves_as_table(self, lidar_ranges: np.ndarray, parameters: np.ndarray, wave_count: int) -> None:
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
                lidar_ranges[index],
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
                 plot_time_end: float, time_point_count: int, maximum_range: float, validation_fraction: float,
                 random_seed: int) -> None:
        self.epoch_count = epoch_count
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.wave_count = wave_count
        self.plot_time_end = plot_time_end
        self.time_point_count = time_point_count
        self.maximum_range = maximum_range
        self.validation_fraction = validation_fraction
        self.random_seed = random_seed

    def run(self) -> None:
        lidar_loader = LidarLoader(
            maximum_range=self.maximum_range,
        )

        lidar_ranges = lidar_loader.load()

        splitter = LidarTrainValidationSplitter(
            validation_fraction=self.validation_fraction,
            random_seed=self.random_seed,
        )

        training_lidar_ranges, validation_lidar_ranges = splitter.split(lidar_ranges)

        standardizer = LidarStandardizer()
        normalized_training_lidar_ranges = standardizer.fit_transform(training_lidar_ranges)
        normalized_validation_lidar_ranges = standardizer.transform(validation_lidar_ranges)
        normalized_all_lidar_ranges = standardizer.transform(lidar_ranges)

        print("LiDAR ranges shape =", lidar_ranges.shape)
        print("Training LiDAR ranges shape =", training_lidar_ranges.shape)
        print("Validation LiDAR ranges shape =", validation_lidar_ranges.shape)
        print("Normalized training LiDAR ranges shape =", normalized_training_lidar_ranges.shape)
        print("Normalized validation LiDAR ranges shape =", normalized_validation_lidar_ranges.shape)

        time_points = np.linspace(
            0.0,
            self.plot_time_end,
            self.time_point_count,
            endpoint=False,
        ).astype(np.float32)

        model = LidarToFmm1VariationalAutoencoder(
            time_points=time_points,
        )

        trainer = Trainer(
            model=model,
            learning_rate=self.learning_rate,
            kl_weight=self.kl_weight,
        )

        trainer.train(
            training_lidar_ranges=normalized_training_lidar_ranges,
            validation_lidar_ranges=normalized_validation_lidar_ranges,
            epoch_count=self.epoch_count,
            batch_size=self.batch_size,
        )

        all_tensor = tf.convert_to_tensor(
            normalized_all_lidar_ranges.astype(np.float32)
        )

        reconstructed_lidar_ranges, latent_mean, latent_log_variance, latent_vector, fmm_parameters, fmm_waves = model(
            all_tensor,
            training=False,
        )

        all_parameters = fmm_parameters.numpy()
        all_waves = fmm_waves.numpy()

        selected_indices = np.linspace(
            0,
            len(normalized_all_lidar_ranges) - 1,
            self.wave_count,
            dtype=int,
        )

        selected_lidar_ranges = lidar_ranges[selected_indices]
        selected_parameters = all_parameters[selected_indices]

        print("")
        print("Selected LiDAR range summaries")
        print("min =", np.min(selected_lidar_ranges, axis=1))
        print("max =", np.max(selected_lidar_ranges, axis=1))
        print("mean =", np.mean(selected_lidar_ranges, axis=1))

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
            lidar_ranges=selected_lidar_ranges,
            parameters=selected_parameters,
            wave_count=self.wave_count,
        )


if __name__ == "__main__":
    epoch_count = 15
    batch_size = 512
    learning_rate = 0.001
    kl_weight = 0.0001
    wave_count = 10
    plot_time_end = 2.0 * math.pi
    time_point_count = 1000
    maximum_range = 4.0
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
        maximum_range=maximum_range,
        validation_fraction=validation_fraction,
        random_seed=random_seed,
    )

    application.run()