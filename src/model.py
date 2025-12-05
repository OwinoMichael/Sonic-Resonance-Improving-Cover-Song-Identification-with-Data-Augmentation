#!/usr/bin/env python3

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.loss import CenterLoss, FocalLoss, HardTripletLoss, ArcFaceLoss  # Import ArcFaceLoss
from src.module.conformer import ConformerEncoder
from src.module.layers import Conv1d, Linear
from src.pytorch_utils import get_latest_model, get_model_with_epoch


class AttentiveStatisticsPooling(torch.nn.Module):
    """
    Implement an attentive statistic pooling layer for each channel.
    """

    def __init__(self, channels, output_channels) -> None:
        """
        Implement an attentive statistic pooling layer for each channel.

        It returns the concatenated mean and std of the input tensor.

        Arguments:
        ---------
          channels: int
            The number of input channels.
          output_channels: int
            The number of output channels.

        """
        super().__init__()

        self._eps = 1e-12
        self._linear = Linear(channels * 3, channels)
        self._tanh = torch.nn.Tanh()
        self._conv = Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self._final_layer = torch.nn.Linear(channels * 2, output_channels, bias=False)
        logging.info(
            "Init AttentiveStatisticsPooling with %s->%s",
            channels,
            output_channels,
        )

    @staticmethod
    def _compute_statistics(x: torch.Tensor, m: torch.Tensor, eps: float, dim: int = 2):
        mean = (m * x).sum(dim)
        std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
        return mean, std

    def forward(self, x: torch.Tensor):
        """
        Calculate mean and std for a batch (input tensor).

        Args:
        ----
          x: torch.Tensor
            Tensor of shape [N, L, C].

        """
        x = x.transpose(1, 2)
        L = x.shape[-1]
        lengths = torch.ones(x.shape[0], device=x.device)
        mask = self.length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)
        total = mask.sum(dim=2, keepdim=True).float()

        mean, std = self._compute_statistics(x, mask / total, self._eps)
        mean = mean.unsqueeze(2).repeat(1, 1, L)
        std = std.unsqueeze(2).repeat(1, 1, L)
        attn = torch.cat([x, mean, std], dim=1)
        attn = self._conv(
            self._tanh(self._linear(attn.transpose(1, 2)).transpose(1, 2)),
        )

        attn = attn.masked_fill(mask == 0, float("-inf"))  # Filter out zero-padding
        attn = F.softmax(attn, dim=2)
        mean, std = self._compute_statistics(x, attn, self._eps)
        # returns pooled statistics
        return self._final_layer(torch.cat((mean, std), dim=1))

    def forward_with_mask(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None,
    ):
        """
        Calculate mean and std for a batch (input tensor).

        Not used in CoverHunter.

        Args:
        ----
          x: torch.Tensor
            Tensor of shape [N, C, L].
          lengths:
            The length of the masks (to be verified).

        """
        L = x.shape[-1]

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        # Make binary mask of shape [N, 1, L]
        mask = self.length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.

        # torch.std is unstable for backward computation
        # https://github.com/pytorch/pytorch/issues/4320
        total = mask.sum(dim=2, keepdim=True).float()
        mean, std = self._compute_statistics(x, mask / total, self._eps)

        mean = mean.unsqueeze(2).repeat(1, 1, L)
        std = std.unsqueeze(2).repeat(1, 1, L)
        attn = torch.cat([x, mean, std], dim=1)

        # Apply layers
        attn = self.conv(self._tanh(self._linear(attn, lengths)))

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=2)
        mean, std = self._compute_statistics(x, attn, self._eps)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        return pooled_stats.unsqueeze(2)

    @staticmethod
    def length_to_mask(
        length: torch.Tensor,
        max_len: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Create a binary mask for each sequence.

        Args:
        ----
          length: torch.LongTensor
            Containing the length of each sequence in the batch. Must be 1D.
          max_len: int
            Max length for the mask, also the size of the second dimension.
          dtype: torch.dtype, default: None
            The dtype of the generated mask.
          device: torch.device, default: None
            The device to put the mask variable.

        Returns:
        -------
          mask: tensor
            The binary mask.

        """
        assert len(length.shape) == 1

        if max_len is None:
            max_len = length.max().long().item()  # using arange to generate mask
        mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len,
        ) < length.unsqueeze(1)

        if dtype is None:
            dtype = length.dtype

        if device is None:
            device = length.device

        return torch.as_tensor(mask, dtype=dtype, device=device)


class Model(torch.nn.Module):

    def __init__(self, hp: Dict) -> None:
        """
        Implement Csi Backbone Model.

        Steps:
          Batch-norm
          Conformer-Encoder(x6 or x4)
          Global-Avg-Pool
          Resize and linear
          Bn-neck

        Loss:
          Focal-loss(ce) + [Triplet OR ArcFace] + Center-loss.
          
        The model automatically uses Triplet or ArcFace based on hp config:
          - If hp has 'triplet' key: uses Triplet Loss
          - If hp has 'arcface' key: uses ArcFace Loss
          - Can have both (will use both losses)
          - Must have at least one

        Args:
        ----
          hp: dict
            The hyperparameters.

        """
        super().__init__()
        self._hp = hp
        self._epoch = 0
        self._step = 0
        self._global_cmvn = torch.nn.BatchNorm1d(hp["input_dim"])
        self._encoder = ConformerEncoder(
            input_size=hp["input_dim"],
            output_size=hp["encoder"]["output_dims"],
            linear_units=hp["encoder"]["attention_dim"],
            num_blocks=hp["encoder"]["num_blocks"],
        )

        # support the option of setting encoder:output_dims != embed_dim
        if hp["encoder"]["output_dims"] != hp["embed_dim"]:
            self._embed_lo = torch.nn.Linear(
                hp["encoder"]["output_dims"], hp["embed_dim"],
            )
        else:
            self._embed_lo = None

        # Bottleneck
        self._bottleneck = torch.nn.BatchNorm1d(hp["embed_dim"])
        self._bottleneck.bias.requires_grad_(False)  # no shift

        self._pool_layer = AttentiveStatisticsPooling(
            hp["embed_dim"], output_channels=hp["embed_dim"],
        )
        
        self._ce_layer = torch.nn.Linear(
            hp["embed_dim"], hp["foc"]["output_dims"], bias=False,
        )

        # Loss Functions
        self._foc_loss = FocalLoss(
            alpha=None,
            gamma=self._hp["foc"]["gamma"],
            num_cls=self._hp["foc"]["output_dims"],
            device=hp["device"],
        )
        
        # Conditionally initialize Triplet Loss
        self._triplet_loss = None
        if "triplet" in hp:
            self._triplet_loss = HardTripletLoss(margin=hp["triplet"]["margin"])
            logging.info(
                "Initialized Triplet Loss (margin=%.2f, weight=%.2f)",
                hp["triplet"]["margin"],
                hp["triplet"]["weight"]
            )
        
        # Conditionally initialize ArcFace Loss
        self._arcface_loss = None
        if "arcface" in hp:
            self._arcface_loss = ArcFaceLoss(
                num_classes=self._hp["foc"]["output_dims"],
                embedding_size=hp["embed_dim"],
                s=hp["arcface"]["s"],
                m=hp["arcface"]["m"],
            )
            logging.info(
                "Initialized ArcFace Loss (s=%.1f, m=%.2f, weight=%.2f)",
                hp["arcface"]["s"],
                hp["arcface"]["m"],
                hp["arcface"]["weight"]
            )
        
        # Validate: Must have at least one metric loss
        if self._triplet_loss is None and self._arcface_loss is None:
            raise ValueError(
                "Config must specify at least one of 'triplet' or 'arcface' loss!"
            )
        
        self._center_loss = CenterLoss(
            num_classes=self._hp["foc"]["output_dims"],
            feat_dim=hp["embed_dim"],
            device=hp["device"],
        )
        
        logging.info("Model size: %.3fM\n", self.model_size() / 1000 / 1000)

    def load_model_parameters(
        self, model_dir, epoch_num=-1, device="mps", advanced=False,
    ):
        """
        Load parameters from pt model, and return model epoch.

        If advanced is set, the model can have different variables from saved.
        This is useful when switching between Triplet and ArcFace models.
        """
        if epoch_num == -1:
            model_path, epoch_num = get_latest_model(model_dir, "g_")
        else:
            model_path = get_model_with_epoch(model_dir, "g_", epoch_num)
            assert model_path, f"Error:model with epoch {epoch_num} not found"

        state_dict_g = torch.load(model_path, map_location=device, weights_only=False)["generator"]
        
        if advanced:
            model_dict = self.state_dict()
            valid_dict = {
                k: v for k, v in state_dict_g.items() if k in model_dict
            }
            model_dict.update(valid_dict)
            self.load_state_dict(model_dict)
            
            # Log what wasn't loaded
            for k in model_dict:
                if k not in state_dict_g:
                    logging.warning("%s not initialized (using random weights)", k)
            
            # Log what was in checkpoint but not in model
            for k in state_dict_g:
                if k not in model_dict:
                    logging.info("%s from checkpoint not used (architecture changed)", k)
        else:
            self.load_state_dict(state_dict_g)

        self.eval()
        self._epoch = epoch_num
        logging.info(
            "Successful init model with epoch-%d, device:%s\n",
            self._epoch,
            device
        )
        return self._epoch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        feat[b, frame_size, feat_size] -> embed[b, embed_dim]
        """
        assert x.dtype == torch.float32, "Input tensor must be of type float32"

        x = self._global_cmvn(x.transpose(1, 2)).transpose(1, 2)
        xs_lens = (
            torch.full([x.size(0)], fill_value=x.size(1), dtype=torch.long)
            .to(x.device)
            .long()
        )
        x, _ = self._encoder(x, xs_lens=xs_lens, decoding_chunk_size=-1)
        if self._embed_lo is not None:
            x = self._embed_lo(x)  # Project to embed_dim
        x = self._pool_layer(x)
        return x

    def compute_loss(
        self, anchor: torch.Tensor, label: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute loss based on configured loss functions.
        
        Always computes Focal Loss.
        Computes Triplet Loss if configured.
        Computes ArcFace Loss if configured.
        Computes Center Loss if weight > 0.
        """
        f_t = self.forward(anchor)

        # Focal Loss (always computed)
        f_i = self._bottleneck(f_t)
        foc_pred = self._ce_layer(f_i)
        foc_loss = self._foc_loss(foc_pred, label)
        loss = foc_loss * self._hp["foc"]["weight"]
        loss_dict = {"foc_loss": foc_loss}

        # Triplet Loss (if configured)
        if self._triplet_loss is not None:
            tri_weight = self._hp["triplet"]["weight"]
            if tri_weight > 0.01:
                tri_loss = self._triplet_loss(f_t, label)
                loss += tri_loss * tri_weight
                loss_dict.update({"tri_loss": tri_loss})

        # ArcFace Loss (if configured)
        if self._arcface_loss is not None:
            arcface_weight = self._hp["arcface"]["weight"]
            if arcface_weight > 0.01:
                arcface_loss = self._arcface_loss(f_t, label)
                loss += arcface_loss * arcface_weight
                loss_dict.update({"arcface_loss": arcface_loss})

        # Center Loss (optional)
        cen_weight = self._hp["center"]["weight"]
        if cen_weight > 0.0:
            cen_loss = self._center_loss(f_t, label)
            loss += cen_loss * cen_weight
            loss_dict.update({"cen_loss": cen_loss})

        return loss, loss_dict


    @torch.jit.ignore
    def inference(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            embed = self.forward(feat)
            embed_foc = self._ce_layer(embed)
        return embed, embed_foc

    # @torch.jit.export
    def get_embed_length(self) -> int:
        return self._hp["embed_dim"]

    def model_size(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # Unused
    #def dump_torch_script(self, dump_path) -> None:
    #    script_model = torch.jit.script(self)
    #    script_model.save(dump_path)
    #    logging.info(f"Export model successfully, see {dump_path}")

    # Unused
    #@torch.jit.export
    #def compute_embed(self, feat: torch.Tensor) -> torch.Tensor:
    #    with torch.no_grad():
    #        return self.forward(feat)
