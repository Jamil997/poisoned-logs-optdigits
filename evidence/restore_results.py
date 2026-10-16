"""Restore the exact published result bytes, checking every member and final log."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parent

def digest(b):return hashlib.sha256(b).hexdigest()
def restore():
 index=json.loads((ROOT/'RESULTS_INDEX.json').read_text())
 for archive,members in index['archives'].items():
  with zipfile.ZipFile(ROOT/archive) as z:
   assert set(z.namelist())==set(members)
   for name,h in members.items():
    target=ROOT/name
    assert target.resolve().is_relative_to(ROOT.resolve())
    b=z.read(name);assert digest(b)==h,(archive,name)
    if target.exists():assert target.read_bytes()==b,str(target)
    else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
 b=b''.join((ROOT/name).read_bytes() for name in index['parts'])
 assert digest(b)==index['runs_sha256']
 target=ROOT/'results/runs.csv.gz'
 if target.exists():assert target.read_bytes()==b
 else:target.write_bytes(b)
 print('Restored and verified all archives and the original compressed outcome file.')
if __name__=='__main__':restore()
