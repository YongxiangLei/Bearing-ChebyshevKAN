import math
import unittest
import torch
from core.model import FAConfig
from models import SpectralOrders, make_model


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.config = FAConfig(harmonics=1, sidebands=1, hidden=6)

    def test_common_contract_and_gradients(self):
        x = torch.randn(2, 2048)
        for name in ['fa_cheby', 'anchored_mlp', 'spectrum_mlp', 'spectrum_cheby']:
            model = make_model(name, self.config)
            out = model(x, 2048., torch.tensor([10., 11.]), [8., 8., 40., 0.])
            self.assertEqual(tuple(out['logits'].shape), (2, 4))
            self.assertEqual(tuple(out['features'].shape), (2, 7))
            self.assertEqual(tuple(out['amplitude'].shape), (2,))
            loss = out['logits'].square().mean() + out['amplitude'].mean() + out['decay'].mean()
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_spectral_representation_resolves_orders_and_ignores_geometry(self):
        t = torch.arange(4096, dtype=torch.float64)/4096
        # Exact 4th shaft order at shaft speed 16 Hz, amplitude invariant.
        x = torch.stack([torch.cos(2*math.pi*64*t), 2*torch.cos(2*math.pi*64*t+1.9)])
        model = SpectralOrders(8)
        output = model(x, 4096, 16, [8., 8., 40., 0.])
        self.assertTrue(torch.allclose(output[0], output[1], atol=1e-10))
        self.assertEqual(output[0].argmax().item(), 3)
        self.assertAlmostEqual(torch.expm1(output[0,3]).item(), 1., places=10)
        other_geometry = model(x, 4096, 16, [9., 7., 42., 0.])
        self.assertTrue(torch.equal(output, other_geometry))

    def test_spectral_silence_and_nyquist(self):
        layer = SpectralOrders(8)
        output = layer(torch.zeros(1, 128), 1024, 10)
        self.assertEqual(output.abs().sum().item(), 0.)
        with self.assertRaises(ValueError):
            layer(torch.zeros(1, 128), 100, 10)


if __name__ == '__main__':
    unittest.main()
