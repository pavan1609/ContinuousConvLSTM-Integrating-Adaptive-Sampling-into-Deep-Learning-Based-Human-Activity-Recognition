from typing import Optional, Sequence

from torch import nn
import torch
import torch.nn.functional as F

from models.gamma_quant import gammaFunction
from models.gamma_quant import UniformQuantizerSTE
from models.test_continuous_conv import ContinuousConv2d


class DeepConvLSTM(nn.Module):
    """
    DeepConvLSTM with optional frequency-adaptive front-end.

    conv_type:
      - "standard": single fixed Conv2d stack
      - "continuous": coordinate-conditioned ("MLP kernel") conv stack with
        frequency-specific branches + optional multirate training inside forward()
      - "standard_multibranch": baseline multi-rate single-checkpoint model:
        one fixed-kernel Conv2d stack per supported rate + optional multirate training
      - "continuous_single": single-branch ContinuousConv stack (no per-rate branches);
        sampling-rate agnostic -- train at one rate, infer at any rate

    standard_padding:
      - "valid": no padding (legacy behavior)
      - "same": time padding so temporal length is preserved per conv layer
    """

    def __init__(
        self,
        channels,
        classes,
        window_size,
        conv_kernels=64,
        conv_kernel_size=5,
        lstm_units=128,
        lstm_layers=2,
        dropout=0.5,
        gamma_quant=None,
        quant_bits=4,
        apply_gamma="global",
        gamma_type="id",
        conv_type="standard",
        conv_rank=8,
        conv_mlp_hidden_dim=32,
        conv_rank_by_rate: Optional[dict] = None,
        conv_mlp_hidden_dim_by_rate: Optional[dict] = None,
        supported_sample_rates: Sequence[int] = (50, 25, 12, 6),
        multirate_training: bool = True,
        kernel_support_s: Optional[float] = None,
        standard_padding: str = "valid",
        temporal_head: Optional[str] = None,
    ):
        super().__init__()

        self.gamma_quant = gamma_quant
        self.channels = int(channels)
        self.classes = int(classes)
        self.window_size = int(window_size)
        self.conv_kernels = int(conv_kernels)
        self.conv_kernel_size = int(conv_kernel_size)
        self.lstm_units = int(lstm_units)
        self.lstm_layers = int(lstm_layers)
        self.dropout_p = float(dropout)
        self.conv_type = str(conv_type)

        # Backward-compatible separation of convolution front-end and temporal head.
        # Legacy aliases still work:
        #   standard_cnn           -> standard conv + CNN head
        #   continuous_cnn         -> continuous conv + CNN head
        #   continuous_single_cnn  -> continuous_single conv + CNN head
        if temporal_head is None:
            if self.conv_type.endswith("_cnn"):
                temporal_head = "cnn"
                self.conv_type = self.conv_type[:-4]
            else:
                temporal_head = "lstm"

        self.temporal_head = str(temporal_head).lower().strip()
        if self.temporal_head not in ("lstm", "cnn"):
            raise ValueError(f"temporal_head must be 'lstm' or 'cnn', got: {temporal_head}")


        def _rate_cfg_int(mapping, fs, default):
            if mapping is None:
                return int(default)
            if not isinstance(mapping, dict):
                return int(default)

            fs_int = int(fs)
            candidate_keys = [
                fs_int,
                str(fs_int),
                f"{fs_int}hz",
                f"{fs_int}Hz",
            ]

            for key in candidate_keys:
                if key in mapping:
                    return int(mapping[key])

            return int(default)

        self.conv_rank = int(conv_rank)
        self.conv_mlp_hidden_dim = int(conv_mlp_hidden_dim)
        self.conv_rank_by_rate = conv_rank_by_rate
        self.conv_mlp_hidden_dim_by_rate = conv_mlp_hidden_dim_by_rate

        standard_padding = str(standard_padding)
        if standard_padding not in ("valid", "same"):
            raise ValueError(f"standard_padding must be 'valid' or 'same', got: {standard_padding}")
        self.standard_padding = standard_padding
        pad_t = (self.conv_kernel_size // 2) if (self.standard_padding == "same") else 0
        self._std_pad = (pad_t, 0)

        self.supported_sample_rates = [int(x) for x in supported_sample_rates]
        self.multirate_training = bool(multirate_training) and (self.conv_type in ("continuous", "standard_multibranch"))
        # continuous_single: never does internal multirate training

        # Debug: used by train.py / evaluation to print branch usage stats
        self.last_sample_rate = None  # int | None
        self.last_effective_kernel_size = None
        self.last_effective_kernel_sizes_per_layer = None

        # In our datasets, configs use ~1-second windows where window_size == sampling_rate.
        # So conv_kernel_size / window_size is a reasonable "seconds" support default.
        base_fs = float(self.window_size) if float(self.window_size) > 0 else 50.0
        if kernel_support_s is not None:
            self.kernel_support_s = float(kernel_support_s)
        else:
            self.kernel_support_s = float(self.conv_kernel_size) / base_fs

        if self.conv_type == "standard":
            self.conv1 = nn.Conv2d(1, conv_kernels, (conv_kernel_size, 1), padding=self._std_pad)
            self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1), padding=self._std_pad)
            self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1), padding=self._std_pad)
            self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1), padding=self._std_pad)

        elif self.conv_type == "continuous":
            self.branches = nn.ModuleDict()
            for fs in self.supported_sample_rates:
                fs_rank = _rate_cfg_int(conv_rank_by_rate, fs, conv_rank)
                fs_hidden = _rate_cfg_int(conv_mlp_hidden_dim_by_rate, fs, conv_mlp_hidden_dim)
                self.branches[str(int(fs))] = nn.ModuleList(
                    [
                        ContinuousConv2d(
                            1,
                            conv_kernels,
                            (conv_kernel_size, 1),
                            rank=fs_rank,
                            mlp_hidden_dim=fs_hidden,
                            padding="same",
                            kernel_support_s=self.kernel_support_s,
                        ),
                        ContinuousConv2d(
                            conv_kernels,
                            conv_kernels,
                            (conv_kernel_size, 1),
                            rank=fs_rank,
                            mlp_hidden_dim=fs_hidden,
                            padding="same",
                            kernel_support_s=self.kernel_support_s,
                        ),
                        ContinuousConv2d(
                            conv_kernels,
                            conv_kernels,
                            (conv_kernel_size, 1),
                            rank=fs_rank,
                            mlp_hidden_dim=fs_hidden,
                            padding="same",
                            kernel_support_s=self.kernel_support_s,
                        ),
                        ContinuousConv2d(
                            conv_kernels,
                            conv_kernels,
                            (conv_kernel_size, 1),
                            rank=fs_rank,
                            mlp_hidden_dim=fs_hidden,
                            padding="same",
                            kernel_support_s=self.kernel_support_s,
                        ),
                    ]
                )

        elif self.conv_type == "standard_multibranch":
            self.branches = nn.ModuleDict()
            for fs in self.supported_sample_rates:
                self.branches[str(int(fs))] = nn.ModuleList(
                    [
                        nn.Conv2d(1, conv_kernels, (conv_kernel_size, 1), padding=self._std_pad),
                        nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1), padding=self._std_pad),
                        nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1), padding=self._std_pad),
                        nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1), padding=self._std_pad),
                    ]
                )

        elif self.conv_type == "continuous_single":
            self.single_branch = nn.ModuleList([
                ContinuousConv2d(1, conv_kernels, (conv_kernel_size, 1), rank=conv_rank, mlp_hidden_dim=conv_mlp_hidden_dim, padding="same", kernel_support_s=self.kernel_support_s),
                ContinuousConv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1), rank=conv_rank, mlp_hidden_dim=conv_mlp_hidden_dim, padding="same", kernel_support_s=self.kernel_support_s),
                ContinuousConv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1), rank=conv_rank, mlp_hidden_dim=conv_mlp_hidden_dim, padding="same", kernel_support_s=self.kernel_support_s),
                ContinuousConv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1), rank=conv_rank, mlp_hidden_dim=conv_mlp_hidden_dim, padding="same", kernel_support_s=self.kernel_support_s),
            ])

        else:
            raise ValueError(f"Unknown conv_type: {self.conv_type}")

        if self.temporal_head == "cnn":
            # Pure CNN head: Conv stack -> temporal mean pooling -> FC classifier.
            # No LSTM module is registered, so checkpoints do not contain LSTM weights.
            self.lstm = None
            self.classifier = None
            self.cnn_classifier = nn.Linear(self.channels * self.conv_kernels, self.classes)
        else:
            self.lstm = nn.LSTM(
                self.channels * self.conv_kernels,
                self.lstm_units,
                num_layers=self.lstm_layers,
            )
            self.classifier = nn.Linear(self.lstm_units, self.classes)
            self.cnn_classifier = None

        self.dropout = nn.Dropout(self.dropout_p)
        self.activation = nn.ReLU()

        self.apply_gamma = apply_gamma
        if self.gamma_quant is not None:
            if apply_gamma == "global":
                self.gamma_function = gammaFunction(init=gamma_type, offset=0.0)
            elif apply_gamma == "local":
                self.gamma_functions = nn.ModuleList(
                    [gammaFunction(init=gamma_type, offset=0.0) for _ in range(self.channels)]
                )
            else:
                raise ValueError(f"Unknown apply_gamma mode: {apply_gamma}")

            self.quantizer = UniformQuantizerSTE(n_bits=quant_bits)

    def _snap_sample_rate(self, T: int) -> int:
        return int(min(self.supported_sample_rates, key=lambda fs: abs(int(fs) - int(T))))

    def _resample_time(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        # x: [B, T, C] -> [B, target_len, C]
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,C], got {tuple(x.shape)}")
        B, T, C = x.shape
        if int(T) == int(target_len):
            return x

        xt = x.permute(0, 2, 1).contiguous().view(B * C, 1, T)
        xt = F.interpolate(xt, size=int(target_len), mode="linear", align_corners=False)
        xt = xt.view(B, C, int(target_len)).permute(0, 2, 1).contiguous()
        return xt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        if x.ndim != 3:
            raise ValueError(f"DeepConvLSTM expected [B,T,C], got shape {tuple(x.shape)}")

        self.last_sample_rate = None  # reset every forward
        B, T, C = x.shape

        sample_rate = None
        if self.conv_type in ("continuous", "standard_multibranch", "continuous_single"):
            if self.training and self.multirate_training:
                idx = torch.randint(low=0, high=len(self.supported_sample_rates), size=(1,), device=x.device).item()
                fs = int(self.supported_sample_rates[idx])
                x = self._resample_time(x, target_len=fs)
                sample_rate = fs
            else:
                fs = self._snap_sample_rate(T)
                if int(T) != int(fs):
                    x = self._resample_time(x, target_len=int(fs))
                sample_rate = int(fs)

            self.last_sample_rate = int(sample_rate)

        x = x.unsqueeze(1)  # [B, 1, T, C]

        if self.gamma_quant == "quantize":
            x = self.quantizer(x)
        elif self.gamma_quant == "quantize_and_gamma":
            if hasattr(self, "gamma_functions"):
                x_split = torch.split(x, 1, dim=-1)
                x = torch.cat([self.gamma_functions[i](x_split[i]) for i in range(len(x_split))], dim=-1)
                x = self.quantizer(x)
            else:
                x = self.quantizer(self.gamma_function(x))

        if self.conv_type == "standard":
            x = self.activation(self.conv1(x))
            x = self.activation(self.conv2(x))
            x = self.activation(self.conv3(x))
            x = self.activation(self.conv4(x))

        elif self.conv_type == "continuous":
            branch = self.branches[str(int(sample_rate))]
            x = self.activation(branch[0](x, sample_rate=sample_rate))
            x = self.activation(branch[1](x, sample_rate=sample_rate))
            x = self.activation(branch[2](x, sample_rate=sample_rate))
            x = self.activation(branch[3](x, sample_rate=sample_rate))

            self.last_effective_kernel_sizes_per_layer = [
                int(branch[0].last_k_t),
                int(branch[1].last_k_t),
                int(branch[2].last_k_t),
                int(branch[3].last_k_t),
            ]
            self.last_effective_kernel_size = int(branch[0].last_k_t)

        elif self.conv_type == "standard_multibranch":
            branch = self.branches[str(int(sample_rate))]
            x = self.activation(branch[0](x))
            x = self.activation(branch[1](x))
            x = self.activation(branch[2](x))
            x = self.activation(branch[3](x))

        elif self.conv_type == "continuous_single":
            branch = self.single_branch
            x = self.activation(branch[0](x, sample_rate=sample_rate))
            x = self.activation(branch[1](x, sample_rate=sample_rate))
            x = self.activation(branch[2](x, sample_rate=sample_rate))
            x = self.activation(branch[3](x, sample_rate=sample_rate))
            self.last_effective_kernel_sizes_per_layer = [int(branch[0].last_k_t), int(branch[1].last_k_t), int(branch[2].last_k_t), int(branch[3].last_k_t)]
            self.last_effective_kernel_size = int(branch[0].last_k_t)

        else:
            raise ValueError(f"Unknown conv_type: {self.conv_type}")

        # [B, K, T, C] -> [T, B, C, K] -> [T, B, C*K]
        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        if self.temporal_head == "cnn":
            # x: [T, B, C*K] -> mean over time -> [B, C*K]
            x = x.mean(dim=0)
            x = self.dropout(x)
            return self.cnn_classifier(x)

        x, _ = self.lstm(x)
        x = x[-1, :, :]
        x = x.view(-1, self.lstm_units)
        x = self.dropout(x)
        return self.classifier(x)
