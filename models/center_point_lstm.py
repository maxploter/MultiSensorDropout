from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Latents:
	"""Args:
	    sensors_h: (num_sensors, B, hidden_size)
	    output: (num_sensors, B, H*W, C)
	"""
	sensors_h: torch.Tensor  # Sensor, B, Vector
	sensors_c: torch.Tensor  # Sensor, B, Vector
	output: torch.Tensor  # Sensor, B, H*W, C

	def detach(self):
		"""Detach all states for BPTT"""
		return Latents(
			self.sensors_h.detach(),
			self.sensors_c.detach(),
			self.output.detach(),
		)

	def update(self, sensor_id, sensor_h, sensor_c, new_output):
		# Clone tensors to avoid in-place operations
		sensors_h_new = self.sensors_h.clone()
		sensors_c_new = self.sensors_c.clone()
		output_new = self.output.clone()

		# Update specific sensor's data directly
		sensors_h_new[sensor_id] = sensor_h
		sensors_c_new[sensor_id] = sensor_c
		output_new[sensor_id] = new_output

		return Latents(
			sensors_h=sensors_h_new,
			sensors_c=sensors_c_new,
			output=output_new,
		)

	def get_sensor_states(self, sensor_id):
		"""Returns (h, c) for specified sensor_id"""
		return self.sensors_h[sensor_id], self.sensors_c[sensor_id]


class CenterPointLSTM(nn.Module):
	def __init__(self, num_latents, latent_dim, feature_channels, feature_size, num_sensors):
		super().__init__()

		self.latent_dim = latent_dim
		self.feature_size = feature_size  # (H, W) of backbone features

		self.projection = nn.Linear(feature_channels, feature_channels // 2)

		input_feature_vector = feature_size[0] * feature_size[1] * feature_channels // 2

		self.lstm_hidden_size_1 = feature_size[0] * feature_size[1] * feature_channels // 8

		# ConvLSTM components
		self.sensor_lstm_cells = nn.ModuleList([
			nn.LSTMCell(input_size=input_feature_vector,
			            hidden_size=self.lstm_hidden_size_1,
			            )
			for _ in range(num_sensors)
		])

		# Learnable initial hidden/cell states
		self.init_sensor_h = nn.Parameter(torch.randn(num_sensors, 1, self.lstm_hidden_size_1), requires_grad=True)
		self.init_sensor_c = nn.Parameter(torch.randn(num_sensors, 1, self.lstm_hidden_size_1), requires_grad=True)
		self.latents = nn.Parameter(torch.randn(num_sensors, 1, feature_size[0] * feature_size[1], feature_channels),
		                            requires_grad=True)  # S, B, H * W, C

		self.mlp = nn.Sequential(
			nn.Linear(feature_channels // 8, feature_channels // 4),
			nn.ReLU(),
			nn.Linear(feature_channels // 4, feature_channels)
		)

	def _init_latents(self, batch_size):
		"""Initialize latent states for a batch"""
		return Latents(
			sensors_h=self.init_sensor_h.expand(-1, batch_size, -1),
			sensors_c=self.init_sensor_c.expand(-1, batch_size, -1),
			output=self.latents.expand(-1, batch_size, -1, -1),  # Sensor, B, Latents, Latent_dim
		)

	def forward(self, data, sensor_id, latents=None):
		B = data.shape[0] if data is not None else latents.sensor_h.shape[1]

		if latents is None:
			latents = self._init_latents(B)

		if data is None:
			# when no information we use previous hidden state as input
			data = latents.output[sensor_id]

		data = self.projection(data)
		data = data.reshape(B, -1)  # [B, C * H * W]

		h, c = latents.get_sensor_states(sensor_id)

		lstm_cell = self.sensor_lstm_cells[sensor_id]
		sensor_h_next, sensor_c_next = lstm_cell(data, (h, c))
		sensor_h_next = sensor_h_next + h  # Residual connection

		sensor_out = sensor_h_next.reshape(B, latents.output.shape[2], -1)  # B, H*W, C
		sensor_out = self.mlp(sensor_out)

		updated_latents = latents.update(sensor_id, sensor_h_next, sensor_c_next, sensor_out)

		S, B, L, C = updated_latents.output.shape

		return updated_latents.output.reshape(B, S * L, C), updated_latents
