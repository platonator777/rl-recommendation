"""Retain H5 snapshots without changing the existing training algorithm."""
from pathlib import Path
import shutil
import tlab.train.policy as policy

save=policy.save_training_checkpoint
def save_and_keep(path,model,optimizer,step,metadata,*args,**kwargs):
    save(path,model,optimizer,step,metadata,*args,**kwargs)
    if int(metadata.get('update',-1)) in (500,1000):
        dst=Path(path).parent/'validation_checkpoints'/f'update_{step:05d}.pt'
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dst)
policy.save_training_checkpoint=save_and_keep
if __name__=='__main__': policy.main()
