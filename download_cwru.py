from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import requests
from scipy.io import loadmat

root = Path(__file__).parent / 'data' / 'CWRU'
root.mkdir(parents=True, exist_ok=True)
ids = [97,98,99,100,105,106,107,108,130,131,132,133,118,119,120,121]
def get(i):
    url = f'https://engineering.case.edu/sites/default/files/{i}.mat'
    p = root / f'{i}.mat'
    if not p.exists():
        resp = requests.get(url, timeout=180)
        resp.raise_for_status()
        p.write_bytes(resp.content)
    data = p.read_bytes()
    variables = [k for k in loadmat(p) if k.endswith('_DE_time')]
    if not variables:
        raise ValueError(f'No drive-end signal in {p}')
    return {'file':p.name,'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'drive_end_variables':variables}
with ThreadPoolExecutor(max_workers=4) as pool:
    result = list(pool.map(get, ids))
(root/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
