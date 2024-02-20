# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy

import torch
from diffusers import UNet2DConditionModel


class ClockworkWrapper(torch.nn.Module):
    """
    Base clockwork diffusion [1]_ wrapper implementing identity adaptor
    clockwork.

    Parameters
    ----------
    unet : UNet2DConditionModel
        UNet to wrap with clockwork.
    clock : int, default=4
        Clock to use for clockwork. Every clock step, will use a full UNet
        pass, every other step will use the adaptor.

    References
    ----------
    .. [1] Habibian et al., Clockwork Diffusion: Efficient Generation With
        Model-Step Distillation, 2023

    """

    def __init__(self, unet: UNet2DConditionModel, clock: int = 4):
        super().__init__()

        assert isinstance(unet, UNet2DConditionModel), (
            "ClockworkWrapper expects a UNet of type UNet2DConditionModel. "
            f"You passed {unet.__class__}."
        )
        self.unet = unet
        self.down_blocks = self.unet.down_blocks
        self.mid_block = self.unet.mid_block
        self.up_blocks = self.unet.up_blocks

        # the second to last upsampling block, where we conduct feature adaptation
        self.adaptor_block = copy.deepcopy(self.up_blocks[-2])
        self.adaptor_block.forward = self._forward_adaptor
        # indexing trick to match state shapes
        self.adaptor_block.resnets = self.adaptor_block.resnets[0:1]

        # clockwork related attributes
        self.clock = clock
        self._time = 0
        # NOTE: the name "r_out" is a reference to the variable name in the
        #       architecture of our paper, figure 3
        self.cached_features_r_out = None
        self.is_adaptor_graph = False
        self.is_full_unet_graph = True
        # NOTE: while UNet already uses full UNet pass by default, calling the
        #       switch_graph_to_full_unet adds a forward hook to cache features
        #       necessary for the following adaptor call 
        self._switch_graph_to_full_unet()  

        # bubble up attributes used by diffusers pipeline
        self.config = self.unet.config
        self.device = self.unet.device
        self.add_embedding = getattr(self.unet, "add_embedding", None)

    def forward(self, *args, **kwargs) -> dict:
        if self._use_full_unet and self.is_adaptor_graph:
            self._switch_graph_to_full_unet()
        elif self._use_adaptor and self.is_full_unet_graph:
            self._switch_graph_to_adaptor()

        outputs = self.unet.forward(*args, **kwargs)

        # NOTE: we automatically _tick to advance the clock requires the user
        #        to reset the clock before each call to a pipeline
        self._tick()
        return outputs

    def reset(self) -> None:
        """Reset clockwork time to 0."""
        self._time = 0

    @property
    def _use_full_unet(self):
        """
        Property indicating whether clockwork wrapper should be using a full
        UNet pass at this diffusion sampling step.
    
        """
        return self._time % self.clock == 0

    @property
    def _use_adaptor(self):
        """
        Property indicating whether clockwork wrapper should be using an
        adaptor pass at this diffusion sampling step.
    
        """
        return not self._use_full_unet

    def _tick(self) -> None:
        """Tick to advance clockwork time by 1."""
        self._time += 1

    def _forward_adaptor(self, *args, **kwargs) -> torch.Tensor:
        """
        Adaptor forward. In this case identity which returns the previously
        cached feature. Should be overwritten to implement learnable adaptors.

        """
        return self.cached_features_r_out

    def _switch_graph_to_adaptor(self) -> None:
        """Switch forward graph to UNet with adaptor in the lower-res block."""
        # revert the changes made by full graph
        self.hook_cache_features_zK.remove()
        # bypass Down 2, 3, 4
        self.unet.down_blocks = self.down_blocks[0:1]
        # bypass Mid
        self.unet.mid_block = None
        # bypass Up 4, 3
        self.unet.up_blocks = self.up_blocks[-2:]
        # adaptor instead of Up 2
        self.unet.up_blocks[0] = self.adaptor_block

        # switch flags
        self.is_adaptor_graph = True
        self.is_full_unet_graph = False

    def _switch_graph_to_full_unet(self) -> None:
        """Switch forward graph to full UNet."""
        self.unet.down_blocks = self.down_blocks
        self.unet.mid_block = self.mid_block
        self.unet.up_blocks = self.up_blocks

        def handler_cache_features_r_out(module, inp, out):
            """Forward hook caching the output of the full UNet upsampler"""
            assert self._use_full_unet, (
                "You are trying to cache the UNet intermediate features while "
                "not using the full UNet graph."
            )
            self.cached_features_r_out = out.detach()

        self.hook_cache_features_zK = self.unet.up_blocks[-2].register_forward_hook(
            handler_cache_features_r_out
        )

        # switch flags
        self.is_adaptor_graph = False
        self.is_full_unet_graph = True
