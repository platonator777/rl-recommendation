"""Create review tables and figures from bundled numerical evidence, without editing README."""
import argparse, csv, json, shutil, sys, os
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from audit_statistics import read,user_means

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'work/report');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    checks=[];sections=[]
    for prefix,title in [('results/policy/reward_scaling/','Основная панель'),('results/temporal/','Поздняя панель')]:
        users=read(prefix+'per_user_metrics.csv');rows=read(prefix+'method_metrics.csv')
        sections.extend(['## '+title,'','| Метод | Next-item NDCG@10 | Пользователи |','|---|---:|---:|'])
        for r in rows:
            means=user_means(users,r['method'],'next_ndcg10')
            actual=float(np.mean(list(means.values())))
            checks.append({'table':title,'method':r['method'],'metric':'next_ndcg10','difference':actual-float(r['next_ndcg10'])})
            assert abs(actual-float(r['next_ndcg10']))<1e-12
            sections.append(f'| {r["method"]} | {actual:.15g} | {len(means)} |')
            for col in ['next_hr10','positive_next_new_ndcg10','positive_next_repeat_ndcg10']:
                if r.get(col):
                    m=user_means(users,r['method'],col)
                    diff=float(np.mean(list(m.values())))-float(r[col]);assert abs(diff)<1e-12
                    checks.append({'table':title,'method':r['method'],'metric':col,'difference':diff})
        sections.append('')
        daily=read(prefix+'24h_metrics.csv') if 'temporal' not in prefix else rows
        for r in daily:
            mapping={'24h_positive_ndcg10':'overall_ndcg10','24h_positive_recall10':'overall_recall10',
                     '24h_new_ndcg10':'new_ndcg10','24h_new_recall10':'new_recall10',
                     '24h_repeat_ndcg10':'repeat_ndcg10','24h_repeat_recall10':'repeat_recall10'}
            if 'temporal' in prefix:mapping={k.replace('24h_positive','24h'):v.replace('overall','24h') if v.startswith('overall') else '24h_'+v for k,v in mapping.items()}
            for col,summary_col in mapping.items():
                means=user_means(users,r['method'],col)
                actual=float(np.mean(list(means.values())))
                diff=actual-float(r[summary_col]);assert abs(diff)<1e-12,(r['method'],col,diff)
                checks.append({'table':title,'method':r['method'],'metric':col,'difference':diff,'n_users':len(means)})
        for name in ['24h_metrics.csv','behavior_metrics.csv','seed_metrics.csv','registered_contrasts.csv']:
            path=ROOT/prefix/name
            if path.exists():shutil.copy2(path,a.output/(('late_' if 'temporal' in prefix else '')+name))
    # Preserve historical tables whose primary evidence is JSON rather than a per-user CSV.
    link=Path(os.path.relpath(ROOT/'results/full_model/tables.md',a.output)).as_posix()
    sections.extend(['## Полная модель','',f'[Сохранённые таблицы]({link}). '
      'Для H5 и основных сравнений доступен отдельный пересчёт из user aggregates. '
      'Не все исторические таблицы имеют per-user источник.'])
    (a.output/'tables.md').write_text('\n'.join(sections)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=read('results/diagnostics/gradient_parameter_noise.csv')
    levels=[4,16,64]
    groups=[[r for r in rows if int(r['samples_per_context'])==n] for n in levels]
    means=[np.mean([float(r['relative_error_of_mean_sampled']) for r in g]) for g in groups]
    sds=[np.std([float(r['relative_error_of_mean_sampled']) for r in g],ddof=1) for g in groups]
    cosines=[np.mean([float(r['cosine_of_mean_sampled_to_exact']) for r in g]) for g in groups]
    fig,ax=plt.subplots(figsize=(7.8,4.8))
    ax.errorbar(levels,means,yerr=sds,marker='o',capsize=4,label='relative error (mean ± batch SD)')
    ax.plot(levels,cosines,marker='s',label='cosine with exact gradient')
    ax.set_xscale('log',base=2);ax.set_xticks(levels,[str(x) for x in levels])
    ax.set(xlabel='Reward samples per context',ylabel='Error / cosine',title='Parameter-gradient sampling noise (8 fixed minibatches)')
    ax.grid(alpha=.25);ax.legend(frameon=False);fig.tight_layout()
    fig.savefig(a.output/'gradient_error_by_sample_count.png',dpi=150);plt.close(fig)
    (a.output/'verification.json').write_text(json.dumps({'passed':True,'checks':checks,
      'limits':'Behavior and external-audit summaries copied, not independently rederived from omitted full top-10 logs. README remains unchanged.'},indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'numeric_checks':len(checks),'output':str(a.output)}))
if __name__=='__main__':main()
