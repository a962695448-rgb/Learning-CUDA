import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent
m=json.loads((r/'archive_manifest.json').read_text(encoding='utf-8'))
for n,v in m['files'].items():
 p=(r/n).resolve()
 if not p.is_relative_to(r.resolve()):raise RuntimeError(n)
 b=p.read_bytes()
 if len(b)!=v['size'] or hashlib.sha256(b).hexdigest()!=v['sha256']:raise RuntimeError(n)
print(json.dumps({'status':'PASS','files':len(m['files']),'bytes':sum(x['size'] for x in m['files'].values())}))
