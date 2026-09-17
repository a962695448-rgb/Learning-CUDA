"""Verify text transport, gzip and every result file before recovering a run."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('receipt',type=Path)
parser.add_argument('parts',type=Path)
parser.add_argument('output',type=Path)
args=parser.parse_args()
receipt=json.loads(args.receipt.read_text())
encoded=[]
for part in receipt['parts']:
    name=PurePosixPath(part['name'])
    assert not name.is_absolute() and '..' not in name.parts
    value=(args.parts/part['name']).read_bytes()
    if len(value)==part['characters']+1 and value.endswith(b'\n'):
        value=value[:-1]
    assert len(value)==part['characters']
    assert hashlib.sha256(value).hexdigest()==part['sha256'],part['name']
    encoded.append(value)
encoded=b''.join(encoded)
assert len(encoded)==receipt['base64_characters']
compressed=base64.b64decode(encoded,validate=True)
assert len(compressed)==receipt['gzip_bytes']
assert hashlib.sha256(compressed).hexdigest()==receipt['gzip_sha256']
raw=gzip.decompress(compressed)
assert len(raw)==receipt['bytes'] and hashlib.sha256(raw).hexdigest()==receipt['sha256']
data=json.loads(raw)
args.output.mkdir(exist_ok=False)
for name,item in data.items():
    relative=PurePosixPath(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    value=item['text'].encode('utf-8')
    assert hashlib.sha256(value).hexdigest()==item['sha256'],name
    path=args.output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
(args.output/'RECOVERY_CHECK.json').write_text(json.dumps({'files_verified':len(data),'raw_bytes':len(raw),
    'raw_sha256':hashlib.sha256(raw).hexdigest(),'gzip_sha256':hashlib.sha256(compressed).hexdigest()},indent=2)+'\n')
print('VERIFIED RECOVERY',len(data),'files',len(raw),'raw bytes')
