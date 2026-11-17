"""Verify each recording contributes equally to the training objective."""
import unittest
import numpy as np
import torch
from torch import nn
from experiment import fit


class ConstantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor([1.,0.,0.,0.]))
    def forward(self,x,fs,speed,geom):
        return {'logits': self.logit.expand(x.shape[0],-1), 'features':x[:,:1]}


class RecordingWeightTest(unittest.TestCase):
    def test_unequal_segment_counts(self):
        def row(identifier,label):
            return {'dataset':'CWRU','recording_id':identifier,'label':label,
                    'envelope':np.zeros(8,dtype=np.float32),'fs':12000.,'shaft_speed':30.,
                    'z':9,'d_mm':8.,'D_mm':40.,'alpha_deg':0.}
        train=[row('a','normal')]+[row('b','outer') for _ in range(3)]
        model=ConstantModel()
        h=fit(model,train,[row('c','normal'),row('d','outer')],1,4,0.,42,False)
        logits=model.logit.detach().reshape(1,4)
        expected=(torch.nn.functional.cross_entropy(logits,torch.tensor([0]))+
                  torch.nn.functional.cross_entropy(logits,torch.tensor([1])))/2
        self.assertAlmostEqual(h[0]['train_loss'],float(expected),places=6)


if __name__=='__main__': unittest.main()
