"""Verify explicit speed provenance for MAT records lacking an RPM field."""
import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.io import savemat
from data import import_cwru


class CWRUSpeedTest(unittest.TestCase):
    def test_nominal_is_explicit_and_mat_speed_takes_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mat_path = tmp / 'example.mat'
            manifest = tmp / 'manifest.csv'
            savemat(mat_path, {'signal': np.zeros(14400)})
            row = dict(path=str(mat_path),signal_key='signal',rpm_key='RPM',x_unit='g',
                       fs='12000',z='9',d_mm='7.94',D_mm='39.04',alpha_deg='0',
                       label='normal',split='test',condition='1 hp',provenance='official nominal speed',
                       nominal_rpm='1772')
            def write():
                with manifest.open('w',newline='',encoding='utf8') as f:
                    w=csv.DictWriter(f,fieldnames=row.keys());w.writeheader();w.writerow(row)
            write()
            sample = import_cwru(manifest,1.2)[0]
            self.assertAlmostEqual(sample['shaft_speed'],1772/60)
            self.assertIn('explicit manifest nominal_rpm=1772',sample['provenance'])
            row['nominal_rpm']='';write()
            with self.assertRaises(ValueError): import_cwru(manifest,1.2)
            row['nominal_rpm']='1772';write()
            savemat(mat_path, {'signal':np.zeros(14400), 'RPM':np.array([1780.])})
            sample = import_cwru(manifest,1.2)[0]
            self.assertAlmostEqual(sample['shaft_speed'],1780/60)
            self.assertIn('MAT variable RPM',sample['provenance'])

    def test_antialiased_common_fs_before_segmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); mat_path=tmp/'example.mat'; manifest=tmp/'manifest.csv'
            t=np.arange(57600)/48000
            savemat(mat_path, {'signal':np.sin(2*np.pi*100*t)+np.sin(2*np.pi*8000*t)})
            row=dict(path=str(mat_path),signal_key='signal',rpm_key='',rpm='1772',
                     rpm_source='official approximate',x_unit='unknown',fs='48000',z='9',
                     d_mm='7.94',D_mm='39.04',alpha_deg='0',label='normal',split='test',
                     condition='1 hp',provenance='48k hypothesis')
            with manifest.open('w',newline='',encoding='utf8') as f:
                w=csv.DictWriter(f,fieldnames=row.keys());w.writeheader();w.writerow(row)
            sample=import_cwru(manifest,1.2,common_fs=12000)[0]
            self.assertEqual(sample['fs'],12000)
            self.assertEqual(len(sample['x']),14400)
            self.assertIn('original fs=48000 Hz; output fs=12000 Hz',sample['provenance'])
            expected=np.sin(2*np.pi*100*np.arange(14400)/12000)
            self.assertLess(float(np.sqrt(np.mean((sample['x'][100:-100]-expected[100:-100])**2))),0.02)


if __name__=='__main__': unittest.main()
