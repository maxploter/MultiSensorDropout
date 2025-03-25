from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conv_lstm import ConvLSTMCell


@dataclass
class Latents:
	"""
	Args:
			sensors_h: (num_sensors, depth, B, C, H, W)
			sensors_c: (num_sensors, depth, B, C, H, W)
			output: (num_sensors, B, H*W, C)
	"""
	sensors_h: torch.Tensor  # Sensor, depth, B, hidden_size
	sensors_c: torch.Tensor  # Sensor, depth, B, hidden_size
	output: torch.Tensor  # Sensor, B, H*W, C

	def detach(self):
		"""Detach all states for BPTT"""
		return Latents(
			self.sensors_h.detach(),
			self.sensors_c.detach(),
			self.output.detach(),
		)

	def update(self, sensor_id, new_sensor_h, new_sensor_c, new_output):
		"""
		Update the latents for a given sensor.
		new_sensor_h, new_sensor_c should have shape (depth, B, hidden_size)
		new_output should have shape (B, H*W, C)
		"""
		# Clone tensors to avoid in-place modifications
		sensors_h_new = self.sensors_h.clone()
		sensors_c_new = self.sensors_c.clone()
		output_new = self.output.clone()

		# Update the specific sensor's states across all layers
		sensors_h_new[sensor_id] = new_sensor_h
		sensors_c_new[sensor_id] = new_sensor_c
		output_new[sensor_id] = new_output

		return Latents(
			sensors_h=sensors_h_new,
			sensors_c=sensors_c_new,
			output=output_new,
		)

	def get_sensor_states(self, sensor_id, layer):
		"""Returns (h, c) for the specified sensor and layer."""
		return self.sensors_h[sensor_id, layer], self.sensors_c[sensor_id, layer]


class CenterPointConvLSTM(nn.Module):
	def __init__(self, num_latents, latent_dim, feature_channels, feature_size, num_sensors, depth=1):
		super().__init__()

		self.latent_dim = latent_dim
		self.feature_size = feature_size  # (H, W) of backbone features
		self.depth = depth

		self.conv_lstm_input_size = feature_channels

		self.conv_lstm_hidden_size = feature_channels // 2

		# ConvLSTM components: for each sensor, we have a list of LSTM cells for each layer.
		self.sensor_conv_lstm_cells = nn.ModuleList([
			nn.ModuleList([
				ConvLSTMCell(
					input_dim=self.conv_lstm_input_size if i == 0 else self.conv_lstm_hidden_size,
					hidden_dim=self.conv_lstm_hidden_size,
					kernel_size=(3,3),
					bias=True
				)
				for i in range(self.depth)
			])
			for _ in range(num_sensors)
		])

		self.init_sensor_h = nn.Parameter(torch.randn(num_sensors, self.depth, 1, self.conv_lstm_hidden_size, *self.feature_size), requires_grad=True)
		self.init_sensor_c = nn.Parameter(torch.randn(num_sensors, self.depth, 1, self.conv_lstm_hidden_size, *self.feature_size), requires_grad=True)
		# Note: output has no layer dimension
		self.latents = nn.Parameter(torch.randn(num_sensors, 1, *self.feature_size, feature_channels),
		                            requires_grad=True)  # S, B, H, W, C

		self.mlp = nn.Sequential(
			nn.Linear(self.conv_lstm_hidden_size, self.conv_lstm_hidden_size * 2),
			nn.ReLU(),
			nn.Linear(self.conv_lstm_hidden_size * 2, feature_channels)
		)

	def _init_latents(self, batch_size):
		"""
		Initialize latent states for a batch.
		Here we replicate the initial state for each layer.
		"""
		# Expand initial h and c from shape (num_sensors, 1, hidden_size)
		# to (num_sensors, depth, batch_size, hidden_size)
		init_h = self.init_sensor_h.expand(-1, -1, batch_size, -1, -1, -1)
		init_c = self.init_sensor_c.expand(-1, -1, batch_size, -1, -1, -1)
		# Expand output (no depth dimension)
		init_output = self.latents.expand(-1, batch_size, -1, -1, -1)
		return Latents(
			sensors_h=init_h,
			sensors_c=init_c,
			output=init_output,
		)

	def forward(self, data, sensor_id, latents=None):
		"""
		data: input tensor (B, feature_channels) or appropriate shape to be fed into projection.
		sensor_id: index of the sensor for which to run the LSTM.
		latents: a Latents object holding states.
		"""
		B = data.shape[0] if data is not None else latents.sensors_h.shape[2]

		if latents is None:
			latents = self._init_latents(B)

		if data is None:
			# When no new information is provided, use the previous output for that sensor.
			data = latents.output[sensor_id]

		lstm_cells = self.sensor_conv_lstm_cells[sensor_id]

		data = data.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W

		# Process first layer using its corresponding states.
		h0, c0 = latents.get_sensor_states(sensor_id, 0)
		sensor_h, sensor_c = lstm_cells[0](data, (h0, c0))
		# To store the updated states for all layers
		new_h_states = [sensor_h]
		new_c_states = [sensor_c]

		# Process subsequent layers; store each layer's hidden and cell states.
		for layer in range(1, self.depth):
			# For each layer, get the initial state for that layer from latents.
			h_init, c_init = latents.get_sensor_states(sensor_id, layer)
			# Use previous layer’s output as input.
			sensor_h, sensor_c = lstm_cells[layer](sensor_h, (h_init, c_init))
			new_h_states.append(sensor_h)
			new_c_states.append(sensor_c)

		# Stack the states so that they have shape (depth, B, hidden_size)
		new_h_tensor = torch.stack(new_h_states, dim=0)
		new_c_tensor = torch.stack(new_c_states, dim=0)

		# Compute sensor output from the last layer's hidden state.
		sensor_out = sensor_h.permute(0, 2, 3, 1) # B, C, H, W -> B, H, W, C

		sensor_out = self.mlp(sensor_out)

		# Update the latents for the given sensor across all layers.
		updated_latents = latents.update(sensor_id, new_h_tensor, new_c_tensor, sensor_out)

		S, B, H, W, C = updated_latents.output.shape
		combined_output = updated_latents.output.reshape(B, S * H * W, C)
		return combined_output, updated_latents
