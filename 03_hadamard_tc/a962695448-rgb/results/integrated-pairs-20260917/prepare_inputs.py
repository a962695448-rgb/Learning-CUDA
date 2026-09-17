"""Reconstruct all frozen inputs from baseline 4516fd4 and the source overlay."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline',type=Path,required=True)
parser.add_argument('--work',type=Path,required=True)
args=parser.parse_args();archive=Path(__file__).resolve().parent
args.work.mkdir(exist_ok=False)
control=args.work/'control/cuda';candidate=args.work/'candidate/cuda'
tree=json.loads((archive/'cuda-base-tree.json').read_text())['tree']
for entry in tree:
    if entry['type']!='blob':continue
    name=PurePosixPath(entry['path']);assert not name.is_absolute() and '..' not in name.parts
    data=(args.baseline/entry['path']).read_bytes()
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==entry['sha'],entry['path']
    target=control/entry['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    target.chmod(int(entry['mode'],8)&0o777)
shutil.copytree(control,candidate)
project=candidate/'03_hadamard_tc/a962695448-rgb'
for path in (archive/'source').rglob('*'):
    if path.is_file():
        assert not path.is_symlink()
        target=project/path.relative_to(archive/'source');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
manifest=json.loads((archive/'INPUT_MANIFEST.json').read_text())
for name in manifest:
    relative=PurePosixPath(name);assert not relative.is_absolute() and '..' not in relative.parts
    if name.startswith(('control/cuda/','candidate/cuda/')):continue
    source=archive/('INPUT_README.zh-CN.md' if name=='README.zh-CN.md' else name)
    target=args.work/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
for name,sha in manifest.items():assert hashlib.sha256((args.work/name).read_bytes()).hexdigest()==sha,name
shutil.copy2(archive/'INPUT_MANIFEST.json',args.work/'INPUT_MANIFEST.json')
print('VERIFIED FROZEN INPUTS',len(manifest))
