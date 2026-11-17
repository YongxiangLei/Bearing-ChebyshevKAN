import csv
from pathlib import Path
import tempfile
import unittest
import numpy as np
from import_paderborn import _channel, load_manifest


class Checks(unittest.TestCase):
    def test_channel_identifies_expected_raster(self):
        t = np.arange(401)/4000
        doc = {'X':[{'Data':t}], 'Y':[{'Name':'speed', 'XIndex':1,
                'Data':np.full(len(t),1500.)}]}
        values, time, fs = _channel(doc, 'speed')
        self.assertEqual(len(values),len(time))
        self.assertAlmostEqual(fs, 4000.)
        with self.assertRaises(ValueError):
            _channel(doc,'vibration_1')

    def test_bearing_disjoint_and_duplicate_paths_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'one.mat').touch()
            (root/'two.mat').touch()
            path=root/'manifest.csv'
            columns=['bearing_id','label','split','condition','recording_id','relative_path','sha256','label_source']
            rows=[{'bearing_id':'K001','label':'normal','split':'train','condition':'N15',
                   'recording_id':'1','relative_path':'one.mat','label_source':'official dataset'},
                  {'bearing_id':'K001','label':'normal','split':'test','condition':'N15',
                   'recording_id':'2','relative_path':'two.mat','label_source':'official dataset'}]
            with path.open('w',encoding='utf8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
            with self.assertRaisesRegex(ValueError,'crosses splits'):
                load_manifest(path,root)
            rows[1]['split']='train'
            with path.open('w',encoding='utf8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
            self.assertEqual(len(load_manifest(path,root)),2)


if __name__=='__main__': unittest.main()
