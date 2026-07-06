"""Meera VITS — Variational Inference Text-to-Speech model."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm


class TextEncoder(nn.Module):
    """Transformer-based text encoder."""

    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        x = x.transpose(1, 2)  # (B, C, T)
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class Encoder(nn.Module):
    """FFT-style encoder with relative positional encoding."""

    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(EncoderLayer(hidden_channels, filter_channels, n_heads, kernel_size, p_dropout))
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, x_mask):
        for layer in self.layers:
            x = layer(x, x_mask)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, kernel_size, p_dropout):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.ff = FFN(hidden_channels, filter_channels, kernel_size, p_dropout)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        # Self-attention
        residual = x
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = self.attn(x, x, x_mask)
        x = self.dropout(x) + residual

        # FFN
        residual = x
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = self.ff(x, x_mask)
        x = self.dropout(x) + residual

        return x * x_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, out_channels, n_heads, p_dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = channels // n_heads

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        B, C, T_q = q.shape
        _, _, T_k = k.shape

        q = q.view(B, self.n_heads, self.head_dim, T_q).transpose(2, 3)
        k = k.view(B, self.n_heads, self.head_dim, T_k).transpose(2, 3)
        v = v.view(B, self.n_heads, self.head_dim, T_k).transpose(2, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))

        out = torch.matmul(attn, v).transpose(2, 3).contiguous().view(B, C, T_q)
        return self.conv_o(out)


class FFN(nn.Module):
    def __init__(self, channels, filter_channels, kernel_size, p_dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(filter_channels, channels, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv1(x * x_mask)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x * x_mask)
        return x * x_mask


class PosteriorEncoder(nn.Module):
    """Encodes mel-spectrogram to latent z."""

    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, n_layers, dilation_rate=1):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WaveNetLike(hidden_channels, kernel_size, dilation_rate, n_layers)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class WaveNetLike(nn.Module):
    """Simplified WaveNet-style network."""

    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.layers.append(
                weight_norm(nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding))
            )
        self.proj = nn.Conv1d(hidden_channels, hidden_channels, 1)

    def forward(self, x, x_mask):
        for layer in self.layers:
            res = x
            x_in = layer(x * x_mask)
            a, b = x_in.split(x_in.size(1) // 2, dim=1)
            x = torch.tanh(a) * torch.sigmoid(b)
            x = x + res
        return self.proj(x * x_mask)


class Generator(nn.Module):
    """HiFi-GAN based vocoder/decoder."""

    def __init__(self, initial_channel, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes):
        super().__init__()
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = weight_norm(nn.Conv1d(initial_channel, upsample_initial_channel, 7, padding=3))

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(nn.ConvTranspose1d(ch, ch // 2, k, stride=u, padding=(k - u) // 2))
            )
            ch = ch // 2
            for j, (rk, rd) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, rk, rd))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)

            xs = None
            for j in range(len(self.resblocks) // self.num_upsamples * i,
                           len(self.resblocks) // self.num_upsamples * (i + 1)):
                if xs is None:
                    xs = self.resblocks[j](x)
                else:
                    xs += self.resblocks[j](x)
            x = xs / len(self.resblocks) * self.num_upsamples

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for d in dilation:
            self.convs1.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=self._get_padding(kernel_size, d))))
            self.convs2.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=self._get_padding(kernel_size, 1))))

    def _get_padding(self, kernel_size, dilation):
        return (kernel_size * dilation - dilation) // 2

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class MeeraVITS(nn.Module):
    """Full VITS model for Meera's voice."""

    def __init__(self, n_vocab, spec_channels, inter_channels, hidden_channels,
                 filter_channels, n_heads, n_layers, kernel_size, p_dropout,
                 resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes, **kwargs):
        super().__init__()
        self.enc_p = TextEncoder(n_vocab, inter_channels, hidden_channels,
                                 filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 16)
        self.dec = Generator(inter_channels, resblock_kernel_sizes, resblock_dilation_sizes,
                             upsample_rates, upsample_initial_channel, upsample_kernel_sizes)

    def forward(self, x, x_lengths, y, y_lengths):
        # Text encoding
        _, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        # Posterior encoding (from mel)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths)

        # Decode to waveform
        o = self.dec(z)

        return o, m_p, logs_p, m_q, logs_q, x_mask, y_mask, z

    @torch.no_grad()
    def infer(self, x, x_lengths, noise_scale=0.667, length_scale=1.0):
        """Generate speech from text."""
        _, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        # Sample from prior
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        # Expand based on length_scale (simple repeat for now)
        o = self.dec(z_p)
        return o

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


if __name__ == "__main__":
    # Quick test
    model = MeeraVITS(
        n_vocab=256, spec_channels=80, inter_channels=192, hidden_channels=192,
        filter_channels=768, n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.1,
        resblock_kernel_sizes=[3, 7, 11], resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[8, 8, 2, 2], upsample_initial_channel=512, upsample_kernel_sizes=[16, 16, 4, 4],
    )
    print(f"MeeraVITS — {model.count_params() / 1e6:.1f}M parameters")
