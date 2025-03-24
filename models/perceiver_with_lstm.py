from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.perceiver import PreNorm, Attention, FeedForward, cache_fn

@dataclass
class Latents:
	sensors_h: torch.Tensor # B, Sensor, Vector
	sensors_c: torch.Tensor # B, Sensor, Vector
	queries: torch.Tensor # B, Sensor, Vector

	def detach(self):
		"""Detach all states for BPTT"""
		return Latents(
			self.sensors_h.detach(),
			self.sensors_c.detach(),
			self.queries.detach(),
		)

	def update(self, sensor_id, sensor_h, sensor_c, queries):
		sensors_h_for_update = self.sensors_h.clone()
		sensors_c_for_update = self.sensors_c.clone()
		sensors_h_for_update[:, sensor_id] = sensor_h
		sensors_c_for_update[:, sensor_id] = sensor_c
		return Latents(
			sensors_h=sensors_h_for_update,
			sensors_c=sensors_c_for_update,
			queries = queries
		)


class PerceiverWithLstm(nn.Module):
	def __init__(self,
	             num_latents,
	             latent_dim,
	             feature_channels,
	             feature_size,
	             num_sensors,
	             latent_heads,
	             latent_dim_head,
	             attn_dropout,
	             ff_dropout,
	             self_per_cross_attn,
	             ):
		super().__init__()

		self.num_latents = num_latents
		self.latent_dim = latent_dim
		self.feature_size = feature_size  # (H, W) of backbone features
		self.projection_before_lstm = nn.Linear(feature_channels, feature_channels // 2)

		input_feature_vector = feature_size[0] * feature_size[1] * feature_channels // 2

		self.lstm_hidden_size_1 = feature_size[0] * feature_size[1] * feature_channels // 4

		# ConvLSTM components
		self.sensor_lstm_cells = nn.ModuleList([
			nn.LSTM(input_size=input_feature_vector, hidden_size=self.lstm_hidden_size_1,
			        batch_first=True)
			for _ in range(num_sensors)
		])

		self.projection_to_latents = nn.Linear(feature_channels // 4, latent_dim)

		get_latent_attn = lambda: PreNorm(latent_dim,
		                                  Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head,
		                                            dropout=attn_dropout))
		get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

		get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

		cache_args = {'_cache': False}
		self.self_attns = nn.ModuleList([])
		for block_ind in range(self_per_cross_attn):
			self.self_attns.append(nn.ModuleList([
				get_latent_attn(**cache_args, key=block_ind),
				get_latent_ff(**cache_args, key=block_ind)
			]))

		# Learnable initial hidden/cell states
		self.init_sensor_h = nn.Parameter(torch.randn(num_sensors, self.lstm_hidden_size_1), requires_grad=True)
		self.init_sensor_c = nn.Parameter(torch.randn(num_sensors, self.lstm_hidden_size_1), requires_grad=True)
		self.init_queries = nn.Parameter(torch.randn(num_latents, latent_dim))

	def _init_latents(self, batch_size):
		"""Initialize latent states for a batch"""
		return Latents(
			sensors_h=self.init_sensor_h.expand(batch_size, -1, -1),
			sensors_c=self.init_sensor_c.expand(batch_size, -1, -1),
			queries=self.init_queries.expand(batch_size, -1, -1),
		)

	def forward(self, data, sensor_id, latents=None):
		B = data.shape[0] if data is not None else latents.sensor_h.shape[1]

		if latents is None:
			latents = self._init_latents(B)

		if data is not None:
			B, H, W, C = data.shape
			data = self.projection_before_lstm(data)
			data = data.reshape(B, -1)  # [B, C * H * W]

			h, c = latents.sensors_h[:, sensor_id], latents.sensors_c[:, sensor_id]

			# Run LSTM
			lstm_cell = self.sensor_lstm_cells[sensor_id]

			x, (sensor_h_next, sensor_c_next) = lstm_cell(data, (h, c))

		else:
			x = latents.sensors_h[:, sensor_id]
			sensor_h_next = latents.sensors_h[:, sensor_id]
			sensor_c_next = latents.sensors_c[:, sensor_id]

		x = x.reshape(B, self.feature_size[0]*self.feature_size[1], -1)
		x = self.projection_to_latents(x)

		q = latents.queries

		for self_attn, self_ff in self.self_attns:
			q = self_attn(q, context=x) + q
			q = self_ff(q) + q

		updated_latents = latents.update(sensor_id, sensor_h_next, sensor_c_next, q)

		return q, updated_latents