import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterPointLSTM(nn.Module):
	def __init__(self, num_latents, latent_dim, feature_channels, feature_size, num_sensors):
		super().__init__()

		self.latent_dim = latent_dim
		self.feature_size = feature_size  # (H, W) of backbone features

		input_feature_vector = feature_size[0] * feature_size[1] * feature_channels

		self.lstm_hidden_size_1 = feature_size[0] * feature_size[1] * feature_channels // 4

		# ConvLSTM components
		self.sensor_lstm_cells = nn.ModuleList([
			nn.LSTM(input_size=input_feature_vector, hidden_size=self.lstm_hidden_size_1,
			        batch_first=True)
			for _ in range(num_sensors)
		])

		self.lstm_hidden_size_2 = feature_size[0] * feature_size[1] * self.latent_dim

		self.state_lstm = nn.LSTM(
			input_size=self.lstm_hidden_size_1,
			hidden_size=self.lstm_hidden_size_2,
			batch_first=True
		)

		# Learnable initial hidden/cell states
		self.init_h_1 = nn.Parameter(torch.randn(num_sensors, self.lstm_hidden_size_1), requires_grad=True)
		self.init_c_1 = nn.Parameter(torch.randn(num_sensors, self.lstm_hidden_size_1), requires_grad=True)
		self.init_h_2 = nn.Parameter(torch.randn(self.lstm_hidden_size_2), requires_grad=True)
		self.init_c_2 = nn.Parameter(torch.randn(self.lstm_hidden_size_2), requires_grad=True)

	def forward(self, data, sensor_id, latents=None):
		if data is not None:
			B, H, W, C = data.shape
			data = data.reshape(B, H * W * C)  # [B, C * H * W]

			# Initialize states
			if latents is None:
				latents = [
					self.init_h_1.expand(B, -1, -1),
					self.init_c_1.expand(B, -1, -1),
					self.init_h_2.expand(B, -1),
					self.init_c_2.expand(B, -1),
					torch.zeros((B, H*W, self.latent_dim), device=self.init_h_1.device)
				]

			h, c = latents[0][:,sensor_id], latents[1][:,sensor_id]

			# Run LSTM
			lstm_cell = self.sensor_lstm_cells[sensor_id]
			output_1, (h_1_next, c_1_next) = lstm_cell(data, (h, c))

			latents[0] = latents[0].clone()
			latents[1] = latents[1].clone()
			latents[0][:, sensor_id] = h_1_next
			latents[1][:, sensor_id] = c_1_next

		else:
			output_1 = latents[0][:,sensor_id]
		# Run state LSTM
		output_2, (h_2_next, c_2_next) = self.state_lstm(output_1, (latents[2], latents[3]))

		latents[2] = h_2_next
		latents[3] = c_2_next

		B, Q, _ = latents[4].shape
		output_2 = output_2.reshape(B, Q, -1)

		# Sum output from different sensors
		latents[4] = latents[4] + output_2

		return latents[4], latents