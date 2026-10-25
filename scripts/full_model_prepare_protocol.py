"""Record actual inputs for a new reproduction, without historical plan files."""
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone
from tlab.config import load_config
from tlab.data.manifest import sha256_file
from tlab.train.checkpoint import source_tree_sha256
from tlab.experiments.full_model_matrix import create_panels

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
    c=load_config(a.config);root=c.path.parent.parent;out=root/c.project['artifacts_dir'];target=out/'protocol.json'
    if target.exists():raise FileExistsError(target)
    panels=create_panels(c.path)
    inputs=['data/raw/listens.parquet','data/processed/listens.parquet','data/processed/protocol.json','data/processed/catalog.parquet',
            'artifacts/reference/runs/sft_strong_seed17/best.pt','artifacts/reference/runs/rm_reference_fit_seed17/best.pt','artifacts/reference/runs/rm_reference_audit_seed17/best.pt']
    data={'protocol_id':'tlab_submission_reproduction_normalized_reward_v1','objective_version':'normalized_reward_v1',
          'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'new_reproduction_inputs_recorded',
          'note':'Reproduction of the disclosed recipe; not historical preregistration or independent confirmation.',
          'source_tree_sha256':source_tree_sha256(root/'src/tlab'),
          'inputs':{s:{'sha256':sha256_file(root/s),'bytes':(root/s).stat().st_size} for s in inputs},
          'panels':{k:panels[k] for k in ['actual_sizes','panel_ids_sha256']}}
    out.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(data,indent=2),encoding='utf-8');print(target)
if __name__=='__main__':main()
