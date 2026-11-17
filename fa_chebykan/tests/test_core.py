"""Focused mathematical and input-contract tests for the prototype."""
import math
import subprocess
import sys
from pathlib import Path
import unittest
import torch
from core.model import FAConfig, FaultAnchoredProjection, FaultAnchoredChebyKAN, cheby_trig, fault_orders


class TestFA(unittest.TestCase):
    def test_orders(self):
        o=fault_orders(torch.tensor([8.,8.,40.,0.]))
        self.assertTrue(torch.allclose(o,torch.tensor([3.2,4.8,2.4,.4]),atol=1e-6))
        with self.assertRaises(ValueError): fault_orders(torch.tensor([8.,41.,40.,0.]))

    def test_chebyshev_trig(self):
        t=torch.linspace(-7,7,53,dtype=torch.float64)
        c,s=cheby_trig(t,5)
        for n in range(1,6):
            self.assertTrue(torch.allclose(c[n],torch.cos(n*t),atol=1e-12))
            self.assertTrue(torch.allclose(s[n],torch.sin(n*t),atol=1e-12))

    def test_phase_invariance_and_single_tone_response(self):
        cfg=FAConfig(harmonics=1,sidebands=0,hidden=4)
        module=FaultAnchoredProjection(cfg).double()
        t=torch.arange(4096,dtype=torch.float64)/4096
        # 3.2 * 20 Hz = 64 Hz; integer-period tone and window.
        x=torch.stack([torch.cos(2*math.pi*64*t),torch.cos(2*math.pi*64*t+1.27)])
        f,r,s=module(x,4096,torch.tensor([20.,20.]),[8.,8.,40.,0.])
        self.assertAlmostEqual(float(torch.expm1(f[0,0]).detach()),1.,places=8)
        self.assertTrue(torch.allclose(f[0],f[1],atol=1e-8))
        self.assertAlmostEqual(float(r[0,0].detach()),1.,places=8)
        self.assertEqual(f.shape,(2,3))

    def test_gradients_and_batch_interface(self):
        cfg=FAConfig(harmonics=2,sidebands=1,hidden=6)
        model=FaultAnchoredChebyKAN(cfg)
        torch.manual_seed(1)
        x=torch.randn(2,1024)
        g=torch.tensor([[8.,8.,40.,0.],[8.,8.,40.,0.]])
        out=model(x,torch.tensor([1024.,1024.]),torch.tensor([10.,11.]),g)
        self.assertEqual(out['logits'].shape,(2,4))
        self.assertEqual(out['features'].shape,(2,14))
        self.assertEqual(len(model.anchor.feature_names),14)
        self.assertEqual(model.anchor.feature_names[:2],['OR:h1:m0','OR:h2:m0'])
        (out['logits'].square().mean()+out['amplitude'].mean()+out['decay'].mean()).backward()
        self.assertTrue(torch.isfinite(model.anchor.eta.grad).all())
        self.assertGreater(float(model.anchor.eta.grad.abs().sum()),0)
        silent=model(torch.zeros(1,1024),1024.,10.,g[0])
        self.assertTrue(torch.isfinite(silent['features']).all())
        self.assertEqual(float(silent['features'].abs().sum().detach()),0.)
        with self.assertRaises(ValueError): model(x,50.,torch.tensor([10.,11.]),g)

    def test_envelope_frontend(self):
        # On the supplied Windows venv, Torch and SciPy's FFT load different
        # Intel OpenMP runtimes. Test the front end in a clean process, as in
        # the recommended preprocessing/training split.
        script=Path(__file__).with_name('signal_check.py')
        result=subprocess.run([sys.executable,str(script)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)


if __name__=='__main__': unittest.main()
