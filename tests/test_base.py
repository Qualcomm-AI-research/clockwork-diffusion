import pytest
import torch
from diffusers import UNet2DConditionModel

from clockwork import ClockworkWrapper


@pytest.fixture
def unet_inputs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = torch.randn(1, 4, 64, 64, device=device)
    timestep = torch.randint(low=0, high=1000, size=(1,), device=device)
    encoder_hidden_states = torch.randn(1, 10, 1280, device=device)
    return sample, timestep, encoder_hidden_states, device


class TestClockworkWrapper(object):
    def test_init(self):
        unet = UNet2DConditionModel()
        ClockworkWrapper(unet, clock=2)

    def test_clock(self, unet_inputs):
        sample, timestep, encoder_hidden_states, device = unet_inputs

        unet = UNet2DConditionModel().to(device)
        clockwork_unet = ClockworkWrapper(unet, clock=2)

        # full UNet pass
        assert clockwork_unet.is_full_unet_graph
        assert clockwork_unet._time == 0
        clockwork_unet(sample, timestep, encoder_hidden_states)

        # adaptor pass
        assert clockwork_unet._time == 1
        clockwork_unet(sample, timestep, encoder_hidden_states)
        assert clockwork_unet.is_adaptor_graph

        # full UNet pass
        assert clockwork_unet._time == 2
        clockwork_unet(sample, timestep, encoder_hidden_states)
        assert clockwork_unet.is_full_unet_graph
