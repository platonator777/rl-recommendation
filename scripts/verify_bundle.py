"""Check local links, syntax, imported module origins and included resource hashes."""
import argparse,ast,hashlib,importlib,json,pkgutil,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'checks/bundle.json');a=p.parse_args()
    errors=[];links=0;imports=0
    for f in ROOT.rglob('*.py'):
        if any(x in f.parts for x in ['work','__pycache__']):continue
        try:ast.parse(f.read_text(encoding='utf-8-sig'))
        except Exception as e:errors.append(f'{f.relative_to(ROOT)}: {e}')
    for f in ROOT.rglob('*.md'):
        if 'work' in f.relative_to(ROOT).parts:continue
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',f.read_text(encoding='utf-8')):
            if re.match(r'\w+://',target):continue
            path,_,anchor=target.partition('#');dst=(f.parent/path).resolve()
            if not dst.is_relative_to(ROOT) or not dst.exists():errors.append(f'{f.name}: missing/outside {target}')
            elif anchor and dst.suffix=='.md' and '<a id="'+anchor+'"' not in dst.read_text(encoding='utf-8'):
                # Explicit anchors are used for source-map navigation; other Markdown anchors are implicit.
                if '<a id=' in dst.read_text(encoding='utf-8'):errors.append('missing anchor '+target)
            links+=1
    import tlab
    for m in pkgutil.walk_packages(tlab.__path__,tlab.__name__+'.'):
        try:
            mod=importlib.import_module(m.name);imports+=1
            if not Path(mod.__file__).resolve().is_relative_to(ROOT/'src'):errors.append('outside import '+m.name)
        except Exception as e:errors.append(f'import {m.name}: {type(e).__name__}: {e}')
    manifest=json.loads((ROOT/'assets_manifest.json').read_text())
    for asset in manifest['assets']:
        if asset['status']!='included':continue
        f=ROOT/asset['destination']
        if not f.is_file():errors.append('missing asset '+asset['destination']);continue
        if hashlib.sha256(f.read_bytes()).hexdigest()!=asset['sha256']:errors.append('hash '+asset['destination'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'passed':not errors,'links':links,'modules':imports,'errors':errors},indent=2),encoding='utf-8')
    print(json.dumps({'passed':not errors,'links':links,'modules':imports,'errors':errors}))
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
