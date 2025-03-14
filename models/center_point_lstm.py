import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conv_lstm import ConvLSTMCell


class CenterPointLSTM(nn.Module):
	def __init__(self, num_latents, latent_dim, feature_channels, feature_size):
		super().__init__()
		self.num_latents = num_latents
		self.latent_dim = latent_dim
		self.feature_size = feature_size  # (H, W) of backbone features

		# ConvLSTM components
		self.conv_lstm_cell = ConvLSTMCell(
			input_dim=feature_channels,
			hidden_dim=latent_dim,
			kernel_size=(3, 3),
			bias=True
		)

		# Projection from hidden state to object queries
		self.query_conv = nn.Conv2d(latent_dim, num_latents * latent_dim, kernel_size=1)

		# Learnable initial hidden/cell states
		self.init_h = nn.Parameter(torch.randn(1, latent_dim, feature_size[0], feature_size[1]))
		self.init_c = nn.Parameter(torch.randn(1, latent_dim, feature_size[0], feature_size[1]))

	def forward(self, data, latents=None):
		if data is not None:
			B, H, W, C = data.shape
			data = data.permute(0, 3, 1, 2)  # [B, C, H, W]

			# Initialize states
			if latents is None:
				h = self.init_h.expand(B, -1, -1, -1)
				c = self.init_c.expand(B, -1, -1, -1)
			else:
				h, c = latents

			# Update ConvLSTM states
			h_next, c_next = self.conv_lstm_cell(data, (h, c))

			# Generate object queries from hidden state
			queries = self.query_conv(h_next)  # [B, num_latents*latent_dim, H, W]
			queries = queries.view(B, self.num_latents, self.latent_dim, H, W)
			queries = queries.mean(dim=[3, 4])  # [B, num_latents, latent_dim]

			return queries, (h_next, c_next)
		else:
			# No data - return previous states or initial states
			if latents is None:
				B = 1  # Default batch size when no data
				return (self.init_h.expand(B, -1, -1, -1),
				        self.init_c.expand(B, -1, -1, -1))
			return latents