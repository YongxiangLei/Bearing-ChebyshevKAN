"""Safety-critical data and split checks; synthetic data only."""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import save_csv,load_csv,validate_splits,validate_record
from experiment import few_shot,metrics


def sample(recording='r1',split='train',label='normal'):
    return {'recording_id':recording,'dataset':'synthetic_test','split':split,
      'x':np.zeros(12000,dtype=np.float32),'x_unit':'arbitrary','fs':12000,
      'shaft_speed':30.,'z':9,'d_mm':8.,'D_mm':40.,'alpha_deg':0.,'label':label,
      'condition':'test','fault_size_mm':'','A0':0.,'decay':500.,'snr_db':0.,'provenance':'fixture'}


class Checks(unittest.TestCase):
    def test_split_is_by_recording_not_segment(self):
        with self.assertRaises(ValueError):
            validate_splits([sample(),sample(split='test')])
        with self.assertRaises(ValueError):
            validate_splits([sample(),sample(label='outer')])

    def test_speed_vector_and_csv_round_trip(self):
        r=sample();r['shaft_speed']=np.full(12000,30.)
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'dataset.csv';save_csv(path,[r]); restored=load_csv(path)[0]
            self.assertEqual(len(restored['shaft_speed']),len(restored['x']))
            self.assertEqual(restored['x_unit'],'arbitrary')
        r['shaft_speed'][0]=np.nan
        with self.assertRaises(ValueError): validate_record(r)

    def test_duration_geometry_and_units_are_checked(self):
        r=sample();r['x']=r['x'][:512]
        with self.assertRaises(ValueError): validate_record(r)
        r=sample();r['d_mm']=np.nan
        with self.assertRaises(ValueError): validate_record(r)
        r=sample();r['dataset']='real';r['x_unit']='arbitrary'
        with self.assertRaises(ValueError): validate_record(r)

    def test_k_counts_independent_recordings(self):
        rows=[sample('a'),sample('a'),sample('b')]
        chosen,ids=few_shot(rows,['normal'],1,42)
        self.assertEqual(len(ids),1)
        self.assertEqual(len({r['recording_id'] for r in chosen}),1)
        with self.assertRaises(ValueError): few_shot(rows,['normal'],3,42)

    def test_metrics_weight_recordings_not_windows(self):
        entries=[{'dataset':'fixture','recording_id':'a','label':0,'logits':np.array([1,0,0,0])}]*4
        entries += [{'dataset':'fixture','recording_id':'b','label':1,'logits':np.array([1,0,0,0])}]
        self.assertEqual(metrics(entries)['recordings'],2)
        self.assertEqual(metrics(entries)['accuracy'],.5)


if __name__=='__main__': unittest.main()
