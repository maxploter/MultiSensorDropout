import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conv_lstm import ConvLSTMCell
from models.perceiver import GEGLU


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

		# Learnable initial hidden/cell states
		self.init_h = nn.Parameter(torch.randn(1, latent_dim, feature_size[0], feature_size[1]))
		self.init_c = nn.Parameter(torch.randn(1, latent_dim, feature_size[0], feature_size[1]))

		# Transformer-style feedforward network with GELU activation
		self.ffn = nn.Sequential(
			nn.LayerNorm(latent_dim),
			nn.Linear(latent_dim, latent_dim * 4 * 2),
			GEGLU(),
			nn.Linear(latent_dim * 4, latent_dim),
		)

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
			# (b, hidden_dim, H, W)
			h_next, c_next = self.conv_lstm_cell(data, (h, c))

			# Generate object queries from hidden state
			h = h_next + h
		else:
			# No data, just propagate latents
			h, c_next = latents
			B, _, H, W = h.shape

		h = h.permute(0, 2, 3, 1)  # [B, H, W, latent_dim]

		h = self.ffn(h) + h

		h_next = h.permute(0, 3, 1, 2)  # [B, latent_dim, H, W]

		h = h.reshape(B, H * W, self.latent_dim)  # [B, num_latents, H*W, latent_dim]
		return h, (h_next, c_next)