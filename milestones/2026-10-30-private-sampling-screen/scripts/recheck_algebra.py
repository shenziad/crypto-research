"""Run the frozen model-free check in a fresh directory and compare results."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
parent = root/'local-runs'
parent.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=parent) as folder:
    work = Path(folder)
    (work/'experiments').mkdir()
    (work/'results').mkdir()
    name = 'fixed_proposal_sampling_algebra_check'
    shutil.copyfile(root/'experiments'/(name+'.py'), work/'experiments'/(name+'.py'))
    subprocess.run([sys.executable, str(work/'experiments'/(name+'.py'))], check=True,
                   stdout=subprocess.DEVNULL)
    actual = (work/'results'/(name+'.json')).read_bytes()
    expected = (root/'results'/(name+'.json')).read_bytes()
    assert actual == expected
    print('98304 exact enumeration cases; archived JSON matches byte-for-byte',
          hashlib.sha256(actual).hexdigest())
