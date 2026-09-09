"""Export the eight requested evidence points. Reads saved evidence; never fits models."""
from pathlib import Path
from collections import Counter
import hashlib, json, math, re, zipfile, sys
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, precision_score, recall_score, f1_score

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'files'
BACK = Path('D:/python projects/Research_pre_cleanup_backup_20260704')
P16 = ROOT/'results/prompt_16_homecredit_model_stability_2024'
A16 = ROOT/'cleanup/audits/prompt_16_final_amended_oot'
O16 = P16/'oot_final_amended_v1'
P14 = ROOT/'cleanup/audits/prompt_14_two_dataset_oot_review_v3'
LEG = BACK/'results/finalized_research/final_report_inputs'
OUT.mkdir(parents=True, exist_ok=True)
sources, results, selections, checks, rowmeta = {}, [], [], [], {}

def ref(p):
    p=Path(p).resolve()
    for root, prefix in [(ROOT,'workspace:'),(BACK,'backup:')]:
        if p.is_relative_to(root): return prefix+p.relative_to(root).as_posix()
    return str(p)

def source(p):
    p=Path(p)
    if not p.is_file(): return ''
    key=ref(p)
    if key not in sources:
        sources[key]={'source':key,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'bytes':p.stat().st_size}
    return key

def csv(p,**kw):
    source(p); return pd.read_csv(p,**kw)

def js(p):
    source(p); return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(name, rows):
    d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    d.to_csv(OUT/name,index=False,encoding='utf-8-sig',lineterminator='\n',float_format='%.15g')
    print(name,len(d),flush=True)
    return d

def base(cohort,dataset,model,selector,run_id,version):
    return dict(evidence_cohort=cohort,dataset=dataset,model=model,selector=selector,run_id=run_id,protocol_version=version,
        dev_oof_auc=np.nan,dev_oof_rows=np.nan,dev_oof_available_folds=0,dev_fold_auc_mean=np.nan,
        oot_auc=np.nan,ks=np.nan,brier=np.nan,capture_at_10=np.nan,score_psi=np.nan,
        actual_feature_count=np.nan,nogueira=np.nan,jaccard=np.nan,stability_folds=0,
        stability_universe_count=np.nan,oot_rows=np.nan,oot_status='completed',
        metric_evidence_status='saved_run_evidence',dev_oof_status='not_saved',score_psi_reference='',
        metric_source='',dev_source='',selection_source='',notes='')

def addsel(row,names,scope,fold,p):
    for rank,name in enumerate(names,1):
        selections.append({k:row[k] for k in ['evidence_cohort','dataset','model','selector','run_id','protocol_version']}|
            dict(selection_scope=scope,fold_id=fold,selection_order=rank,feature=str(name),source=source(p)))

def names_from(p):
    d=csv(p)
    col=next((c for c in ['feature','feature_name','selected_feature','selected_features'] if c in d),d.columns[0])
    return d[col].dropna().astype(str).tolist()

def stability(row,sets,P,p,valid_nogueira=True):
    sets=[set(x) for x in sets if x]
    row['stability_folds']=len(sets);row['stability_universe_count']=P
    if len(sets)<2:return
    row['jaccard']=float(np.mean([len(a&b)/len(a|b) for i,a in enumerate(sets) for b in sets[i+1:] if a|b]))
    if valid_nogueira:
        m=len(sets); count=Counter(x for s in sets for x in s);k=np.mean([len(s) for s in sets])
        if 0<k<P:
            row['nogueira']=1-sum((v/m)*(1-v/m)*m/(m-1) for v in count.values())/(k*(1-k/P))
        else:row['notes']+=' Nogueira undefined for selecting the entire universe.'
    else:row['notes']+=' Nogueira withheld: audited pool/universe mismatch in historical source.'
    source(p)

def fold_selections(row,p,P,valid_nogueira=True):
    if not Path(p).is_file():return
    d=csv(p);fc=next((c for c in ['fold_id','fold'] if c in d),None)
    nc=next((c for c in ['feature','feature_name'] if c in d),None)
    if not fc or not nc:return
    sets=[]
    for f,g in d.groupby(fc):
        names=g[nc].dropna().astype(str).tolist();sets.append(names);addsel(row,names,'DEV_fold_training',int(f),p)
    stability(row,sets,P,p,valid_nogueira)

def read_preds(p):
    p=Path(p);source(p)
    if p.suffix=='.parquet':d=pd.read_parquet(p)
    else:
        cols=pd.read_csv(p,nrows=0).columns
        take=[c for c in cols if c in ['target','y_true','score','prediction','prediction_probability','y_pred_proba','fold_id','stable_row_id','case_id','decision_threshold','predicted_class']]
        d=pd.read_csv(p,usecols=take)
    yc=next(c for c in ['target','y_true'] if c in d)
    sc=next(c for c in ['score','prediction_probability','y_pred_proba','prediction'] if c in d)
    return d,yc,sc

def oof(row,paths,require_fold_column=False):
    if not paths:return
    parts=[]
    for p in paths:
        d,yc,sc=read_preds(p)
        if require_fold_column and ('fold_id' not in d or d.fold_id.isna().any()):
            row['dev_oof_status']='saved_DEV_file_is_in_sample_not_OOF';row['dev_source']=source(p);return
        parts.append(d.rename(columns={yc:'y',sc:'s'}))
    d=pd.concat(parts,ignore_index=True)
    idc=next((c for c in ['case_id','stable_row_id'] if c in d),None)
    if idc: assert not d[idc].duplicated().any(), (row['run_id'],'duplicate OOF IDs')
    assert d.y.isin([0,1]).all() and np.isfinite(d.s).all()
    folds=d.fold_id.nunique() if 'fold_id' in d else len(paths)
    row.update(dev_oof_auc=roc_auc_score(d.y,d.s),dev_oof_rows=len(d),dev_oof_available_folds=folds,
        dev_oof_status='complete_five_fold_pooled_OOF' if folds==5 else 'pooled_available_folds_only',dev_source=';'.join(source(p) for p in paths))

def archive_run(dataset,model,method,run,cohort='canonical_matrix',version='canonical_llm_matrix_v2'):
    p=run/'results/experiment_summary.csv';s=csv(p).iloc[0]
    row=base(cohort,dataset,model,method,run.name,version)
    for key,skey in [('oot_auc','oot_auc'),('ks','oot_ks'),('brier','oot_brier'),('capture_at_10','oot_bad_rate_capture_at_10'),('score_psi','oot_model_score_psi'),('actual_feature_count','oot_selected_feature_count'),('dev_fold_auc_mean','cv_auc_mean')]:row[key]=s.get(skey,np.nan)
    row.update(metric_source=source(p),oot_rows=120053 if dataset=='homecredit' else 293105,score_psi_reference='full_DEV_final_model_in_sample_scores')
    P=529 if dataset=='homecredit' else 675
    fp=run/'features/final_selected_features.csv'
    if fp.exists():
        names=names_from(fp);addsel(row,names,'full_DEV_for_OOT','',fp);row['selection_source']=source(fp)
    fold_selections(row,run/'features/fold_selected_features.csv',P,valid_nogueira=not (dataset=='homecredit' and method in ['llm_then_mrmr','llm_corrected_clip_then_mrmr','corrected_clip_then_mrmr']))
    if method=='pca':
        row['nogueira']=np.nan;row['jaccard']=np.nan;row['stability_universe_count']=np.nan
        row['notes']+=' PCA component labels PC1..PCk recur across folds but do not identify the same fitted components; original-feature Nogueira/Jaccard are not applicable.'
    for q in [run/'results/dev_oof_predictions.csv',run/'results/dev_predictions.csv']:
        if q.exists():oof(row,[q],True);break
    rowmeta[row['run_id']]={'threshold':s.get('oot_decision_threshold'),'precision':s.get('oot_precision'),'recall':s.get('oot_recall'),'f1':s.get('oot_f1'),'prediction':run/'results/oot_predictions.csv','summary':p}
    results.append(row);return row

# 32 original final matrix identities, retaining their protocol labels.
for dataset in ['homecredit','lendingclub_v2']:
    for s in csv(BACK/f'results/{dataset}/matrix_runs.csv').itertuples():
        archive_run(dataset,s.model,s.selector,BACK/s.output_folder)

# 64 classical extension identities. DEV OOF must be actual held-out predictions.
p14=csv(P14/'two_dataset_results_long.csv')
devcombo=csv(ROOT/'cleanup/audits/prompt_13_combination_dev_review/dev_fold_results.csv')
for s in p14.itertuples():
    row=base('classical_extension',s.dataset,s.model,s.method,s.result_id,'prompt_14_two_dataset_oot_review_v3')
    row.update(configuration=s.configuration,actual_feature_count=s.realized_k,oot_auc=s.oot_auc,ks=s.oot_ks,brier=s.oot_brier,
        capture_at_10=s.oot_bad_rate_capture_at_10,score_psi=s.score_psi,dev_fold_auc_mean=s.dev_auc_mean,
        metric_source=source(P14/'two_dataset_results_long.csv'),oot_rows=s.oot_rows)
    pp=ROOT/s.prediction_authentication_reference
    P=529 if s.dataset=='homecredit' else 675
    if 'full_baseline_v1' in str(pp) or '/runs/' in pp.as_posix():
        run=pp.parent
        if run.name=='results':run=run.parent
        row['protocol_version']='full_baseline_v1' if 'full_baseline_v1' in str(pp) else 'cross_dataset_rank_voting_v1'
        row['score_psi_reference']='saved_DEV_probabilities; in_sample for full_baseline_v1; pooled OOF for cdv1'
        fp=run/'features/final_selected_features.csv'
        if not fp.exists():fp=run/'selected_features.csv'
        if fp.exists():addsel(row,names_from(fp),'full_DEV_for_OOT','',fp);row['selection_source']=source(fp)
        foldpath=run/'features/fold_selected_features.csv'
        if not foldpath.exists():foldpath=run/'fold_selections.csv'
        fold_selections(row,foldpath,P)
        dp=run/'results/dev_predictions.csv'
        if dp.exists():oof(row,[dp],True)
        sp=run/'results/experiment_summary.csv'
        if sp.exists():
            su=csv(sp).iloc[0];rowmeta[row['run_id']]={'threshold':su.get('oot_decision_threshold'),'precision':su.get('oot_precision'),'recall':su.get('oot_recall'),'f1':su.get('oot_f1'),'prediction':pp,'summary':sp}
    else:
        ep=Path(str(pp).replace('.oot_predictions.csv','.json'));ej=js(ep)
        sel=ROOT/'results/selector_combinations_v1/oot/selections'/f"{ej['selection_id']}.combination_result.json"
        sj=js(sel);addsel(row,sj['selected_features'],'full_DEV_for_OOT','',sel);row['selection_source']=source(sel)
        d=devcombo[(devcombo.dataset==s.dataset)&(devcombo.final_model==s.model)&(devcombo.method==s.method)]
        if s.method=='iv_then_boruta':d=d[d.iv_pool==int(s.configuration.replace('pool',''))]
        paths=[];sets=[]
        for v in d.itertuples():
            dp=ROOT/'results/selector_combinations_v1/dev/evaluations'/f'{v.cell_id}.dev_predictions.csv'
            if dp.exists():paths.append(dp)
            fp=ROOT/'results/selector_combinations_v1/dev/selections'/f'{v.selection_id}.combination_result.json'
            sj=js(fp);sets.append(sj['selected_features']);addsel(row,sj['selected_features'],'DEV_fold_training',v.fold_id,fp)
        oof(row,paths);stability(row,sets,P,sel)
        row['score_psi_reference']='pooled_DEV_OOF_scores'
        met=ej['worker_result'].get('metrics',{})
        rowmeta[row['run_id']]={'threshold':met.get('decision_threshold'),'precision':met.get('precision'),'recall':met.get('recall'),'f1':met.get('f1'),'prediction':pp,'summary':ep}
    results.append(row)

# Final amended Stability registry: include unavailable cells explicitly.
ac=csv(A16/'complete_amended_dev_accounting.csv')
oot16=csv(O16/'analysis/oot_metrics.csv');psi16=csv(O16/'analysis/score_psi.csv').set_index('configuration_order')
for s in oot16.itertuples():
    n=s.configuration_order;phase='supplemental' if n>=31 else 'classical'
    row=base('stability_final_amended','homecredit_model_stability_2024',s.model,s.method_id,s.configuration_id,'prompt_16_final_amended_oot_v1')
    row.update(configuration_order=n,oot_auc=s.auc,ks=s.ks,brier=s.brier,capture_at_10=s.bad_rate_capture_at_10,
        actual_feature_count=s.realized_support,oot_status=s.status,oot_rows=304916,notes='' if s.status=='complete' else s.reason,
        metric_source=source(O16/'analysis/oot_metrics.csv'),score_psi_reference='available_DEV_OOF_folds; frozen_quantile_bins')
    if n in psi16.index:row['score_psi']=psi16.loc[n,'score_psi']
    candidates=[]
    for sp in (O16/phase/'selection_fits').glob('*/selection.json'):
        sj=js(sp)
        if n in sj.get('fit_spec',{}).get('dependent_configuration_orders',[]) or (sj.get('method_id')==s.method_id and sj.get('model')==s.model):candidates.append((sp,sj))
    if candidates:
        sp,sj=candidates[0];addsel(row,sj.get('selected_features',[]),'full_DEV_for_OOT','',sp);row['selection_source']=source(sp)
        if sj.get('realized_support') is None and not sj.get('selected_features'):
            row['actual_feature_count']=np.nan
            row['notes']+=' No completed final selection; upstream zero support sentinel is exported as missing.'
    rec=ac[ac.configuration_order==n];paths=[];sets=[];aucs=[]
    for v in rec.itertuples():
        dp=P16/('dev_llm_supplement_v3' if v.source=='llm_supplement_v3' else 'dev_v1')/f'fold_{v.fold_id}'
        ep=dp/'evaluations'/f'cell_{n:03d}'
        if v.status=='complete' and (ep/'predictions.parquet').exists():paths.append(ep/'predictions.parquet');aucs.append(js(ep/'metrics.json')['auc'])
        if s.method_id=='llm' and candidates:
            sp,sj=candidates[0];names=sj.get('selected_features',[])
        else:
            fitdir=f'{s.method_id}_{s.model}' if v.source=='llm_supplement_v3' else str(v.fit_id)
            sp=dp/'selection_fits'/fitdir/'selection.json'
            sj=js(sp) if sp.exists() else {};names=sj.get('selected_features',[])
        if names:sets.append(names);addsel(row,names,'DEV_fold_training',v.fold_id,sp)
    oof(row,paths);row['dev_fold_auc_mean']=np.mean(aucs) if aucs else np.nan
    stability(row,sets,1068 if n>=31 else 1959,O16/'analysis/oot_metrics.csv')
    rowmeta[row['run_id']]={'threshold':s.decision_threshold,'precision':s.precision,'recall':s.recall,'f1':s.f1,'prediction':O16/phase/'evaluations'/f'cell_{n:03d}'/'predictions.parquet','summary':O16/'analysis/oot_metrics.csv'}
    results.append(row)

# Corrected historical CLIP pipelines and final Stability CLIP directions.
for model in ['lr','catboost']:
    for name in ['corrected_clip_then_mrmr','llm_then_corrected_clip_then_mrmr']:
        run=BACK/f'results/corrected_homecredit_clip/combined_pipeline/runs/homecredit_{model}_{name}'
        archive_run('homecredit',model,'llm_corrected_clip_then_mrmr' if name.startswith('llm') else name,run,'historical_corrected_clip','corrected_homecredit_clip / identity_equivalence_v2')
clip=csv(P16/'clip_experiment_v1/analysis/final_clip_results.csv');cliprows=[]
for s in clip.itertuples():
    row=base('stability_clip','homecredit_model_stability_2024',s.classifier,s.direction,f'clip_stability_v1_{s.direction}_{s.classifier}','stability_clip_experiment_v1')
    row.update(oot_auc=s.oot_auc,ks=s.oot_ks,brier=s.oot_brier,capture_at_10=s.oot_capture_at_10pct,score_psi=s.oot_score_psi,
        actual_feature_count=s.final_k,dev_oof_auc=s.dev_pooled_oof_auc,dev_fold_auc_mean=s.dev_fold_auc_mean,dev_oof_available_folds=5,
        dev_oof_status='complete_five_fold_pooled_OOF',oot_rows=304916,metric_source=source(P16/'clip_experiment_v1/analysis/final_clip_results.csv'),score_psi_reference='full_DEV_final_model_in_sample_scores')
    run=P16/'clip_experiment_v1/downstream'/s.direction/s.classifier
    oof(row,[run/'dev_oof_predictions.parquet'],True)
    assert abs(row['dev_oof_auc']-s.dev_pooled_oof_auc)<1e-12
    fp=run/'full_dev_selected_features.csv';addsel(row,names_from(fp),'full_DEV_for_OOT','',fp);row['selection_source']=source(fp)
    fold_selections(row,run/'fold_selected_features.csv',s.candidate_pool_size)
    results.append(row);cliprows.append(row.copy()|{'candidate_pool_P':s.candidate_pool_size,'global_feature_universe':1959})

# The screenshot's eight Home Credit voting runs; no DEV folds were executed here.
V=BACK/'results/final_experiments/voting_pipeline'
for s in csv(V/'voting_run_inventory.csv').itertuples():
    row=base('homecredit_voting_screenshot','homecredit',s.model,s.pipeline,s.run_id,'prompt4b_voting_pipeline')
    row.update(oot_auc=s.auc,ks=s.ks,actual_feature_count=s.selected_feature_count,oot_rows=s.actual_oot_rows,
        metric_source=source(V/'voting_run_inventory.csv'),dev_oof_status='not_executed_for_this_voting_protocol',notes='mRMR implementation is RF relevance / Pearson correlation, per saved ranking provenance.')
    pp=BACK/s.prediction_file;d,yc,sc=read_preds(pp)
    assert abs(roc_auc_score(d[yc],d[sc])-s.auc)<1e-12
    row['brier']=brier_score_loss(d[yc],d[sc]);top=np.argsort(-d[sc].to_numpy(),kind='stable')[:math.ceil(.1*len(d))]
    row['capture_at_10']=float(d[yc].to_numpy()[top].sum()/d[yc].sum());row['notes']+=' Brier and Capture@10 recomputed from saved OOT probabilities.'
    results.append(row)
vsel=csv(V/'voting_selected_features.csv')
for row in [r for r in results if r['evidence_cohort']=='homecredit_voting_screenshot']:
    d=vsel
    for c in ['run_id','pipeline','model']:
        if c in d:d=d[d[c]==(row['selector'] if c=='pipeline' else row[c])]
    nc=next(c for c in ['feature','feature_name','selected_feature'] if c in d)
    addsel(row,d[nc].tolist(),'full_DEV_for_OOT','',V/'voting_selected_features.csv')

# Additional frozen cross-dataset voting sensitivity runs. Primary P=200 already in P14.
cd=csv(ROOT/'results/final_experiments/cross_dataset_voting_inference_v1/voting_budget_results.csv')
for s in cd.itertuples():
    if s.configuration in ['reference','voting_k200']:continue
    run=ROOT/f'results/runs/{s.dataset}/{s.run_id}'
    row=base('cross_dataset_voting_sensitivity',s.dataset,s.model,'cdv1_'+s.configuration,s.run_id,'cross_dataset_rank_voting_v1')
    met=csv(run/'metrics.csv');mt=met[met.split=='OOT'].iloc[0]
    row.update(oot_auc=s.oot_auc,ks=s.oot_ks,brier=mt.brier,score_psi=s.score_psi,actual_feature_count=s.selected_feature_count,
        oot_rows=s.oot_row_count,score_psi_reference='pooled_DEV_OOF_scores',metric_source=source(run/'metrics.csv'))
    d,yc,sc=read_preds(run/'results/oot_predictions.csv');top=np.argsort(-d[sc].to_numpy(),kind='stable')[:math.ceil(.1*len(d))]
    row['capture_at_10']=float(d[yc].to_numpy()[top].sum()/d[yc].sum())
    oof(row,[run/'results/dev_predictions.csv'],True)
    fp=run/'selected_features.csv';addsel(row,names_from(fp),'full_DEV_for_OOT','',fp);row['selection_source']=source(fp)
    fold_selections(row,run/'fold_selections.csv',529 if s.dataset=='homecredit' else 675)
    results.append(row)

# A screenshot-only Stability LLM->mRMR row is not in the 34-cell executed registry.
r=base('screenshot_only','homecredit_model_stability_2024','catboost','llm_then_mrmr','screenshot_stability_llm_then_mrmr','not_supplied')
r.update(ks=.593430,metric_evidence_status='user_screenshot_aggregate_only',metric_source='user screenshot Section IV',oot_status='execution_not_identified',notes='No matching method in final 34-cell Stability registry.');results.append(r)

# Controlling screenshot values. Never attach unrelated historical metrics to revised AUCs.
updates={('homecredit','catboost','llm'):{'oot_auc':.793450,'ks':.450543,'brier':.153354},
 ('lendingclub_v2','lr','llm'):{'oot_auc':.740234},
 ('lendingclub_v2','catboost','llm_then_mrmr'):{'oot_auc':.770664,'ks':.394310,'brier':.202423},
 ('homecredit_model_stability_2024','lr','llm'):{'oot_auc':.834400},
 ('homecredit_model_stability_2024','catboost','llm'):{'oot_auc':.878400,'brier':.179236},
 ('homecredit','lr','mrmr'):{'oot_auc':.769890}}
masked_runs=set()
for row in results:
    key=(row['dataset'],row['model'],row['selector'])
    if key in updates and row['evidence_cohort'] in ['canonical_matrix','stability_final_amended']:
        masked_runs.add(row['run_id'])
        for col in ['dev_oof_auc','dev_oof_rows','dev_fold_auc_mean','oot_auc','ks','brier','capture_at_10','score_psi','actual_feature_count','nogueira','jaccard','stability_universe_count']:
            row[col]=np.nan
        row.update(updates[key]);row['metric_evidence_status']='user_screenshot_aggregate_only_unmatched_saved_prediction'
        row['oot_status']='historical_run_complete_revision_unverified'
        row['historical_protocol_version']=row['protocol_version']
        row['protocol_version']='screenshot_revision_protocol_unverified'
        row['dev_oof_available_folds']=np.nan;row['stability_folds']=np.nan;row['oot_rows']=np.nan
        row['score_psi_reference']='unavailable_for_screenshot_revision'
        row['metric_source']='user screenshots Sections III/IV';row['dev_oof_status']='unavailable_for_screenshot_revision'
        row['notes']='Historical run identity retained for disclosure only; its other metrics and selected features are not authenticated to this revised score.'
for s in selections:s['screenshot_revision_link']='unverified_historical_selection_not_linked_to_revised_AUC' if s['run_id'] in masked_runs else 'saved_run_selection'
write('1_Complete_results.csv',results)
write('6_CLIP_Stability_2024_results.csv',cliprows)
write('7_Selected_feature_lists.csv',selections)
source(ROOT/'configs/protocols/homecredit_model_stability_2024_v1/third_dataset_protocol_lock.json')
split=js(ROOT/'cleanup/audits/third_dataset_protocol_freeze/proposed_split_and_fold_boundaries.json')
foldrows=[]
for f in split['folds']:
    v=f['validation'];observed=ac.loc[ac.fold_id==f['fold_id'],'rows'].dropna().unique();assert list(observed)==[v['rows']]
    foldrows.append(dict(fold_id=f['fold_id'],validation_rows=v['rows'],validation_start=v['date_min'],validation_end=v['date_max'],training_rows=f['train']['rows'],training_start=f['train']['date_min'],training_end=f['train']['date_max'],validation_events=v['target_1'],validation_row_id_sha256=v['ordered_case_id_sha256'],verified_against='final complete_amended_dev_accounting.csv',source=source(ROOT/'cleanup/audits/third_dataset_protocol_freeze/proposed_split_and_fold_boundaries.json')))
write('1_Stability_validation_fold_sizes.csv',foldrows)

# Point 2: screenshot headline choices, without pretending a DEV selection audit exists.
headline_specs=[
 ('homecredit','lr','stable_core_llm_fill',.748857,'mrmr',.769890,-.021033,-.031907,-.010159,.0003001),
 ('homecredit','catboost','llm',.793450,'rfe_catboost',.781426,.012024,.001532,.022516,.0247008),
 ('lendingclub_v2','lr','llm',.740234,'iv_then_boruta',.709803,.030431,.025766,.035096,1e-36),
 ('lendingclub_v2','catboost','llm_then_mrmr',.770664,'iv_then_boruta',.720147,.050517,.045959,.055075,7.20e-104),
 ('homecredit_model_stability_2024','lr','llm',.834400,'iv_then_boruta',.802956,.031444,.020319,.042569,1.08e-7),
 ('homecredit_model_stability_2024','catboost','llm',.878400,'rfe_catboost',.849969,.028431,.018409,.038453,1.08e-7)]
headlines=[];hpaired=[];thresholds=[]
for ds,model,llm,la,cls,ca,delta,lo,hi,hp in headline_specs:
    for family,method,auc in [('best_LLM_assisted',llm,la),('best_classical',cls,ca)]:
        matches=[r for r in results if r['dataset']==ds and r['model']==model and r['selector']==method and math.isfinite(r['oot_auc']) and abs(r['oot_auc']-auc)<=.00000051]
        assert matches,(ds,model,method,auc)
        r=matches[0]
        headlines.append(dict(dataset=ds,model=model,headline_family=family,selector=method,displayed_oot_auc=auc,
            score_partition_shown='OOT',dev_oof_auc_for_same_score_revision=r['dev_oof_auc'],
            selection_rule_status='Original DEV-versus-OOT selection decision not authenticated; screenshot displays OOT score',
            matched_run_id=r['run_id'],protocol_version=r['protocol_version'],evidence_status=r['metric_evidence_status'],source='user screenshot Section III'))
        t=dict(dataset=ds,model=model,headline_family=family,selector=method,headline_oot_auc=auc,
            dev_selected_threshold=np.nan,holdout_precision=np.nan,holdout_recall=np.nan,holdout_f1=np.nan,
            threshold_selection_partition='full_DEV_training_predictions',threshold_rule='argmax TPR-FPR (KS/Youden)',
            holdout_rows=r['oot_rows'],run_id=r['run_id'],status='not_available_for_screenshot_score_revision',source='')
        meta=rowmeta.get(r['run_id'],{})
        if r['run_id'] not in masked_runs and meta and meta.get('threshold') is None:
            t.update(status='final_threshold_not_selected_or_saved_by_combination_OOT_worker',threshold_selection_partition='not_implemented_in_this_OOT_worker',threshold_rule='not_implemented_in_this_OOT_worker',source=source(ROOT/'src/credit_risk_fs/experiments/selector_combinations.py'))
        if r['run_id'] not in masked_runs and meta and meta.get('threshold') is not None and pd.notna(meta.get('threshold')):
            d,yc,sc=read_preds(meta['prediction']);tau=float(meta['threshold']);pred=(d[sc]>=tau).astype(int)
            calc={'precision':precision_score(d[yc],pred,zero_division=0),'recall':recall_score(d[yc],pred,zero_division=0),'f1':f1_score(d[yc],pred,zero_division=0)}
            assert abs(roc_auc_score(d[yc],d[sc])-auc)<.00000051
            for k,val in calc.items():
                if meta.get(k) is not None and pd.notna(meta[k]):assert abs(val-meta[k])<1e-12,(r['run_id'],k)
                t['holdout_'+k]=val
            t.update(dev_selected_threshold=tau,status='verified_from_saved_holdout_predictions',source=source(meta['summary'])+';'+source(meta['prediction']))
        thresholds.append(t)
    assert abs((la-ca)-delta)<.0000011
    hpaired.append(dict(dataset=ds,model=model,llm_selector=llm,llm_oot_auc=la,classical_selector=cls,classical_oot_auc=ca,
        reported_delta_auc=delta,reported_ci95_lower=lo,reported_ci95_upper=hi,reported_holm_p=hp,raw_p=np.nan,
        evidence_status='screenshot_transcription_only_inference_not_reproduced',test_method='not_supplied',holm_family='not_supplied',source='user screenshot Section III'))
write('2_Headline_selection.csv',headlines)
write('2_Headline_paired_inference_as_supplied.csv',hpaired)
write('8_Frozen_threshold_metrics.csv',thresholds)

# Points 3 and 4: explicit scopes and availability.
elig=[]
for ds in ['homecredit','lendingclub_v2']:
    elig.append(dict(dataset=ds,protocol_version='canonical_llm_matrix_v2 / stability_expert_v3',pure_llm_iv_screened=True,
        iv_partition='current DEV training fold; full DEV for final OOT refit',iv_settings='min_iv=0.01; max_iv_for_leakage=0.5; encode=true',
        prompt_statistics_partition='same training slice AFTER IV/WOE transformation',missingness_filter='training-slice missing rate <=0.95',
        ranking_reuse='shared between models/hybrids for identical training metadata/cache key; distinct folds and full DEV have distinct keys',
        target_used_in_eligibility=True,oot_used_in_eligibility=False,
        source=source(ROOT/'src/credit_risk_fs/selectors/llm_screening.py')+';'+source(ROOT/'src/credit_risk_fs/selectors/registry.py')))
elig.append(dict(dataset='homecredit_model_stability_2024',protocol_version='dev_llm_supplement_v3 / stability_expert_v3_target_free',pure_llm_iv_screened=False,
    iv_partition='not calculated',iv_settings='iv_filter_kwargs={}',prompt_statistics_partition='none; names/descriptions only; no row statistics supplied',
    missingness_filter='fold-1 DEV training only: 200661 rows; missing rate <=0.90; 1959 to 1068 features',
    ranking_reuse='single sealed order reused for all five folds and OOT; K=20/40 truncation',target_used_in_eligibility=False,oot_used_in_eligibility=False,
    source=source(ROOT/'src/credit_risk_fs/experiments/prompt_16_llm_supplement.py')+';'+source(ROOT/'cleanup/audits/prompt_16_llm_scope_correction/maximum_performance_prefilter_v3.json')))
write('3_LLM_eligibility.csv',elig)
full=[]
for r in results:
    if r['selector']=='full_features':
        st=r['dataset']=='homecredit_model_stability_2024'
        full.append(dict(dataset=r['dataset'],model=r['model'],run_id=r['run_id'],protocol_version=r['protocol_version'],
            oot_auc=r['oot_auc'],actual_feature_count=r['actual_feature_count'],final_split_match='registered_same_split',
            preprocessing_match='configured frozen semantics; no completed full-feature evaluation' if st else 'same frozen baseline preprocessing',
            feature_universe_match='1959 classical predictors; differs from 1068-feature LLM v3 eligibility' if st else '529 Home Credit / 675 LendingClub original predictors',
            model_settings_match='configured same LR/CatBoost settings; no completed full-feature evaluation' if st else 'same frozen final-model settings',
            inherited_result='unavailable status inherited from all five DEV resource stops; no inherited numerical AUC' if st else 'authenticated full_baseline_v1 result reused in prompt14 review',
            status=r['oot_status'],source=r['metric_source']))
write('4_Full_feature_references.csv',full)

# Point 5: all 24 actual voting tests, plus the four changed screenshot LR references.
vp=BACK/'results/final_experiments/powered_comparisons/voting_24_powered_results.csv'
vtests=csv(vp);vout=[]
for s in vtests.itertuples():
    r=dict(comparison_id=s.comparison_id,dataset=s.dataset,model=s.model,voting_selector=s.pipeline_a,
        comparator=s.pipeline_b,voting_run_id=s.run_id_a,comparator_run_id=s.run_id_b,
        voting_protocol_version='prompt4b_voting_pipeline; RF/correlation mRMR',
        comparator_protocol_version='corrected_homecredit_clip / identity_equivalence_v2' if s.pipeline_b=='llm_corrected_clip_then_mrmr' else 'canonical_llm_matrix_v2; RF/correlation mRMR',
        inference_protocol_version='prompt5_powered_comparisons_v1',actual_test_performed=True,
        voting_auc=s.auc_a,comparator_auc=s.auc_b,paired_delta_auc=s.delta_auc,
        ci95_lower=s.bootstrap_auc_ci_lower,ci95_upper=s.bootstrap_auc_ci_upper,raw_p=s.delong_p,holm_adjusted_p=s.holm_adjusted_p,
        p_test='two_sided_paired_DeLong',ci_method='paired_stratified_bootstrap_percentile',bootstrap_iterations=s.bootstrap_iterations,bootstrap_seed=s.bootstrap_seed,
        holm_family='voting_24',holm_family_size=24,paired_holdout_rows=s.matched_borrowers,
        delta_definition='voting minus comparator',status='saved_completed_test',source=source(vp))
    if s.model=='lr' and s.pipeline_b=='statistical_mrmr':
        for k in ['comparator_auc','paired_delta_auc','ci95_lower','ci95_upper','raw_p','holm_adjusted_p']:r[k]=np.nan
        r['status']='executed_historical_comparator_conflicts_with_screenshot_numerics_withheld'
        r2=r.copy();r2.update(comparison_id=s.comparison_id+'_screenshot_reference',comparator='full-pool mRMR as labelled in screenshot',
            comparator_run_id='',comparator_protocol_version='screenshot calls it MI-based; matching run not identified',actual_test_performed=False,
            comparator_auc=.769890,paired_delta_auc=s.auc_a-.769890,holm_family='',holm_family_size=np.nan,
            status='point_estimate_difference_only_no_matching_paired_test',source='user screenshot Section VI;'+source(vp))
        vout.append(r2)
    vout.append(r)
    if not (s.model=='lr' and s.pipeline_b=='statistical_mrmr'):assert abs(s.delta_auc-(s.auc_a-s.auc_b))<1e-12
# Independently verify the preserved Holm family including the four suppressed tests.
order=np.argsort(vtests.delong_p.to_numpy());adjust=np.minimum(1,np.maximum.accumulate((24-np.arange(24))*vtests.delong_p.to_numpy()[order]))
assert np.allclose(adjust,vtests.holm_adjusted_p.to_numpy()[order],rtol=1e-11,atol=1e-14)
write('5_Voting_comparisons.csv',vout)
write('5_Actual_test_inventory.csv',[dict(comparison_id=s.comparison_id,model=s.model,voting_selector=s.pipeline_a,comparator=s.pipeline_b,
    run_id_a=s.run_id_a,run_id_b=s.run_id_b,AUC_p_test='two-sided paired DeLong',AUC_CI='paired stratified bootstrap; 2000 resamples',
    KS_CI='paired stratified bootstrap; 2000 resamples',KS_p_test='none',holm_family='voting_24',family_size=24,
    paired_rows=s.matched_borrowers,unmatched_a=s.unmatched_a,unmatched_b=s.unmatched_b,target_mismatches=s.target_mismatches,
    performed=True,source=source(vp)) for s in vtests.itertuples()])

# Point 6 historical before/after, retaining the documented Nogueira exclusion.
beforeafter=[]
hist=csv(LEG/'tables/table_03_corrected_clip_incremental.csv')
for s in hist.itertuples():
    beforeafter.append(dict(dataset='homecredit',model='lr' if s.model=='Logistic Regression' else 'catboost',
        before_pipeline='LLM then RF/correlation mRMR',before_protocol_version='canonical_llm_matrix_v2',
        after_pipeline='LLM then corrected CLIP then RF/correlation mRMR',after_protocol_version='corrected_homecredit_clip / identity_equivalence_v2; reporting package 2.0',
        before_auc=s.reference_oot_auc,after_auc=s.clip_oot_auc,delta_auc=s.delta_oot_auc,
        before_jaccard=s.reference_selection_jaccard,after_jaccard=s.clip_selection_jaccard,delta_jaccard=s.delta_selection_jaccard,
        before_nogueira=np.nan,after_nogueira=np.nan,delta_nogueira=np.nan,
        nogueira_status='excluded_by_source_audit: saved N=529 rather than authenticated P=60/100; no validated replacement',
        before_score_psi=s.reference_score_psi,after_score_psi=s.clip_score_psi,delta_score_psi=s.delta_score_psi,
        score_psi_reference='full_DEV_final_model_in_sample_scores',source=source(LEG/'tables/table_03_corrected_clip_incremental.csv')+';'+source(BACK/'results/final_research_package_v2/reproducibility_summary.md')))
write('6_CLIP_improvement_before_after.csv',beforeafter)

# Point 7 model/selector settings and application-level call accounting.
settingrows=[]
cfgpath=ROOT/'configs/experiments/full_baseline_v1.yaml';source(cfgpath);cfg=yaml.safe_load(cfgpath.read_text())
def flatten_settings(scope,value,p,path=''):
    if isinstance(value,dict):
        for k,v in value.items():flatten_settings(scope,v,p,(path+'.' if path else '')+k)
    else:settingrows.append(dict(scope=scope,parameter=path,value=json.dumps(value),source=source(p)))
for section in ['final_model_settings','selector_settings','feature_budgets','provenance','pilot_review_decisions']:
    flatten_settings('full_baseline_v1 / '+section,cfg[section],cfgpath)
for p in [ROOT/'configs/selectors/boruta.yaml',ROOT/'configs/selectors/llm.yaml']:
    source(p);flatten_settings('legacy defaults (see run-specific provenance)',yaml.safe_load(p.read_text()),p)
flatten_settings('Stability LLM v3 amendment',js(ROOT/'cleanup/audits/prompt_16_llm_scope_correction/maximum_performance_prefilter_v3.json')['method_preservation'],ROOT/'cleanup/audits/prompt_16_llm_scope_correction/maximum_performance_prefilter_v3.json')
write('7_Implementation_settings.csv',settingrows)
cost=csv(LEG/'evidence/llm_cost_results.csv');callrows=[]
for s in cost[cost.record_type.isin(['usage_taxonomy','runtime'])].to_dict('records'):
    if s.get('record_type')=='runtime' and not s.get('logical_requests',0):continue
    callrows.append(dict(scope=s.get('scenario'),dataset=s.get('dataset'),model=s.get('model'),pipeline=s.get('pipeline'),run_id=s.get('run_id'),
        logical_requests=s.get('logical_requests'),canonical_recorded_generation_calls=s.get('canonical_physical_calls'),source_generation_calls=s.get('source_generation_calls'),
        recorded_total_generation_calls=s.get('total_physical_calls'),local_reuses=s.get('local_reuse'),
        exact_application_attempts=np.nan,exact_application_retries=np.nan,transport_retries=np.nan,
        status='historical cost audit counts generations/reuse; exact per-request retry attempts not preserved',source=source(LEG/'evidence/llm_cost_results.csv')))
callrows.extend([
 dict(scope='accepted Stability v3 ranking',dataset='homecredit_model_stability_2024',logical_requests=1,recorded_total_generation_calls=1,exact_application_attempts=2,exact_application_retries=1,transport_retries=np.nan,status='attempt 1 rejected unknown feature; attempt 2 accepted; no fallback',source=source(P16/'dev_llm_supplement_v3/llm_ranking/ranking_payload.json')),
 dict(scope='final Stability OOT',dataset='homecredit_model_stability_2024',logical_requests=0,recorded_total_generation_calls=0,exact_application_attempts=0,exact_application_retries=0,transport_retries=0,status='sealed DEV ranking reused; zero regeneration',source=source(O16/'final_evidence_manifest.json')),
 dict(scope='this export',dataset='all',logical_requests=0,recorded_total_generation_calls=0,exact_application_attempts=0,exact_application_retries=0,transport_retries=0,status='no LLM or model calls made to build this export',source='build_evidence.py')])
for r in results:
    if r['evidence_cohort']=='canonical_matrix' and r['selector']=='llm_then_boruta':
        callrows.append(dict(scope='legacy run outside detailed cost-audit table',dataset=r['dataset'],model=r['model'],pipeline=r['selector'],run_id=r['run_id'],logical_requests=np.nan,recorded_total_generation_calls=np.nan,exact_application_attempts=np.nan,exact_application_retries=np.nan,transport_retries=np.nan,status='Executed run; call/retry counts not established by the cited cost-audit table',source=r['metric_source']))
    if r['run_id'] in masked_runs and r['selector']!='mrmr':
        callrows.append(dict(scope='screenshot score revision',dataset=r['dataset'],model=r['model'],pipeline=r['selector'],run_id='',logical_requests=np.nan,recorded_total_generation_calls=np.nan,exact_application_attempts=np.nan,exact_application_retries=np.nan,transport_retries=np.nan,status='Revised AUC has no authenticated execution/call-log linkage',source='user screenshots Sections III/IV'))
callrows.append(dict(scope='external full-529 voting LLM ranking generation',dataset='homecredit',model='shared',pipeline='prompt4b_voting_pipeline',logical_requests=np.nan,recorded_total_generation_calls=np.nan,exact_application_attempts=np.nan,exact_application_retries=np.nan,transport_retries=np.nan,status='Full ranking externally generated; exact provider attempt log not identified',source=source(V/'voting_manifest.json')))
for r in callrows:
    r['screenshot_revision_link']='unverified_historical_calls_not_linked_to_revised_AUC' if r.get('run_id') in masked_runs else 'scope_as_labelled'
write('7_LLM_call_and_retry_counts.csv',callrows)

# Additional implementation provenance, kept factual and scoped to saved protocols.
for scope,value,p in [
 ('Legacy registry Boruta',{'engine':'BorutaPy','RF_n_estimators':500,'RF_max_depth':6,'n_estimators':'auto','max_iter':15,'random_state':42,'n_jobs':1,'support':'confirmed only; budget truncation; no padding'},ROOT/'src/credit_risk_fs/selectors/boruta.py'),
 ('Legacy integer-step RFE',{'engine':'sklearn RFE','estimator':'CatBoost','iterations':500,'depth':6,'learning_rate':.05,'step':10,'random_state':42},ROOT/'src/credit_risk_fs/selectors/rfe.py'),
 ('Canonical fractional RFE',{'remove_each_iteration':'max(1, floor(0.20 * surviving_count)), capped at surviving_count-K','stop':'exact K','importance':'CatBoost feature importance'},ROOT/'src/credit_risk_fs/selectors/heavy/rfe_catboost.py')]:flatten_settings(scope,value,p)
mp=ROOT/'cleanup/audits/prompt_16_llm_scope_correction/maximum_performance_prefilter_v3.json'
flatten_settings('Stability LLM v3 runtime controls',js(mp)['maximum_performance_controls'],mp)
write('7_Implementation_settings.csv',settingrows)
fe=[dict(dataset='homecredit',original_feature_universe=529,summary='Application fields plus client-level bureau, previous-application, POS-CASH, installment and credit-card aggregates; ratios, counts and repayment summaries.',time_and_target_exclusions='TARGET; recent_decision; PREV_recent_decision_MAX; DAYS_DECISION; application_time_proxy',source=source(ROOT/'src/credit_risk_fs/feature_engineering/homecredit/assemble.py')),
 dict(dataset='lendingclub_v2',original_feature_universe=675,summary='Origination-safe FICO, income/affordability, loan exposure, utilization, history, inquiries, account mix, delinquency, joint-applicant, missingness and interpretable interaction features; ratios and fixed transforms.',time_and_target_exclusions='TARGET; loan_status; payment/recovery; hardship/settlement; future performance/date fields',source=source(ROOT/'docs/lendingclub_v2_feature_engineering_spec.md')+';'+source(ROOT/'src/credit_risk_fs/feature_engineering/lendingclub/application.py')),
 dict(dataset='homecredit_model_stability_2024',original_feature_universe=1959,summary='Approved depth-0 fields and deterministic depth-1 per-case aggregates; depth-2 excluded. Counts, nonmissing/missing counts, sum/mean/min/max/variance and ordered first/last summaries. LLM supplement v3 retains 1068 by earliest-training-fold missingness.',time_and_target_exclusions='target; case_id; date_decision; availability-reviewed exclusions',source=source(ROOT/'src/credit_risk_fs/data/homecredit_model_stability_2024/adapter.py'))]
write('7_Feature_engineering_summary.csv',fe)
tuning=[dict(scope='LR/CatBoost final models',provenance='Frozen values in full_baseline_v1; reused by final amended Stability and combination protocols',selection_data='No originating hyperparameter-search study or trial table identified in inspected evidence',status='settings_known_original_tuning_provenance_unavailable',source=source(cfgpath)),
 dict(scope='Final feature budgets',provenance='K=20 LR and K=40 CatBoost fixed in configs; IV->Boruta uses natural confirmed support rather than mandatory K',selection_data='No budget optimization study identified',status='fixed_protocol_budgets_not_established_as_DEV_optimized',source=source(cfgpath)),
 dict(scope='Heavy selectors',provenance='Six authenticated DEV first-fold pilot cells; no OOT used; retain confirmed Boruta support without padding',selection_data='DEV feasibility/runtime pilot',status='pilot_authenticated_then_frozen',source=source(cfgpath)),
 dict(scope='IV->Boruta pools and voting pools',provenance='P=100,200,300 preregistered; corrected cross-dataset voting P=200 primary and 100/300 sensitivity',selection_data='DEV review did not select/remove/tune/reorder configurations',status='registered_comparisons',source=source(ROOT/'src/credit_risk_fs/experiments/prompt_13_dev_audit.py')),
 dict(scope='Stability CLIP checkpoint',provenance='Seeds 11,22,33,44,55; choose minimum source-validation-loss checkpoint separately per seed; five-seed Procrustes consensus',selection_data='source representation validation; not downstream OOT AUC',status='frozen_protocol',source=source(ROOT/'docs/clip/STABILITY_CLIP_EXPERIMENT_V1.md'))]
write('7_Tuning_provenance.csv',tuning)
source(V/'ranking_source_audit.csv');source(V/'voting_manifest.json')
source(BACK/'results/final_experiments/powered_comparisons/powered_comparison_manifest.json')
source(BACK/'results/corrected_homecredit_clip/combined_pipeline/combined_pipeline_manifest.json')
source(ROOT/'src/credit_risk_fs/clip/stability_experiment.py')

# All additional registration states and missing metrics remain inspectable.
df=pd.DataFrame(results);sel=pd.DataFrame(selections)
availability=[]
for r in results:
    for metric in ['dev_oof_auc','oot_auc','ks','brier','capture_at_10','score_psi','actual_feature_count','nogueira','jaccard']:
        if pd.isna(r[metric]):
            reason=r['notes'] or r['dev_oof_status']
            if metric=='dev_oof_auc':reason=r['dev_oof_status']
            if r['evidence_cohort']=='homecredit_voting_screenshot' and metric in ['score_psi','nogueira','jaccard']:reason='No matching DEV score reference or five-fold selections saved for this voting protocol.'
            availability.append(dict(point=1,run_id=r['run_id'],dataset=r['dataset'],model=r['model'],selector=r['selector'],metric=metric,status='unavailable_not_zero',reason=reason))
write('1_Missing_metric_evidence.csv',availability)

# Validate the actual final feature-list counts and all six screenshot CLIP pairs.
assert len(df)==157 and not df.duplicated(['evidence_cohort','run_id']).any()
assert not sel.duplicated(['evidence_cohort','run_id','selection_scope','fold_id','feature']).any()
counts=sel[sel.selection_scope=='full_DEV_for_OOT'].groupby('run_id').feature.nunique()
for r in results:
    if pd.notna(r['actual_feature_count']):assert r['actual_feature_count']==counts.get(r['run_id'],0),(r['run_id'],'feature count')
expected_clip={('stability_to_stability','lr'):(.722500,.697266),('stability_to_stability','catboost'):(.683333,.688861),
 ('homecredit_to_stability','lr'):(.752500,.722518),('homecredit_to_stability','catboost'):(.712500,.710146),
 ('lendingclub_to_stability','lr'):(.685000,.656678),('lendingclub_to_stability','catboost'):(.720833,.717478)}
for r in cliprows:
    no,ja=expected_clip[(r['selector'],r['model'])]
    assert abs(r['nogueira']-no)<.00000051 and abs(r['jaccard']-ja)<.00000051
for col in ['dev_oof_auc','oot_auc','ks','brier','capture_at_10','jaccard']:
    v=df[col].dropna();assert v.between(0,1).all(),col
assert len(thresholds)==12 and sum(t['status']=='verified_from_saved_holdout_predictions' for t in thresholds)==4
assert [f['validation_rows'] for f in foldrows]==[204567,203798,205980,201466,202820]
validation=[dict(check='Result identity uniqueness',status='PASS',detail='157 versioned records; no duplicate cohort/run keys'),
 dict(check='Selected-feature uniqueness and final counts',status='PASS',detail=f'{len(sel)} feature-membership rows; all nonmissing final counts match lists'),
 dict(check='Stability chronological folds',status='PASS',detail='Five sizes match frozen split and final execution accounting'),
 dict(check='Stability CLIP screenshot metrics',status='PASS',detail='All six Nogueira/Jaccard pairs recomputed from saved fold selections and match at displayed precision'),
 dict(check='Pooled OOF provenance',status='PASS',detail='Only validation predictions pooled; 36 in-sample DEV files rejected as OOF; partial folds labelled'),
 dict(check='Voting AUC and deltas',status='PASS',detail='Eight voting AUCs recomputed from saved probabilities; all retained test deltas reconcile'),
 dict(check='Voting Holm correction',status='PASS',detail='All 24 saved p-values used to independently verify family adjustment; four conflicting test rows are suppressed only on export'),
 dict(check='Headline frozen-threshold metrics',status='PASS_WITH_GAPS',detail='4 of 12 recomputed; 6 revised scores have no matching predictions; 2 combination OOT workers did not select/save thresholds'),
 dict(check='Headline selection provenance',status='UNRESOLVED',detail='Displayed scores are OOT; original DEV-versus-OOT winner-selection decision not authenticated'),
 dict(check='Headline screenshot inference',status='UNVERIFIED',detail='Six supplied CIs/Holm p-values transcribed; no matching authenticated complete comparison family'),
 dict(check='Historical CLIP Nogueira',status='UNAVAILABLE',detail='Prior audit rejected denominator; no validated replacement copied into this export')]
write('1-8_Validation_checks.csv',validation)
write('1-8_Source_provenance.csv',list(sources.values()))

readme=f'''# Evidence for points 1–8

Prepared 2026-09-09 from the saved workspace, the July 4 backup and the four screenshots supplied in this request. This package contains factual exports and evidence limitations. No models, selectors, thresholds or LLM requests were rerun.

## Files and source precedence

CSV names start with the requested point. All written explanations are in this file. `1-8_Source_provenance.csv` records source paths and SHA-256 hashes. `workspace:` resolves under `D:/python projects/Research`; `backup:` resolves under `D:/python projects/Research_pre_cleanup_backup_20260704`. Source files are not bundled. The validation notebook checks the delivered tables. `1-8_Build_evidence.py` preserves the extraction and calculation code. With the source workspace and backup available, run `python 1-8_Build_evidence.py "D:/python projects/Research"`; it writes a `files` subdirectory beside the script. The original build command is `.venv/Scripts/python.exe .research_evidence_2026-09-09/build_evidence.py` from the workspace root.

Your screenshots control the explicitly supplied values. Six revised pipeline rows do not match saved prediction results; their conflicting historical numbers and unsupported companion metrics are omitted. Their saved run IDs are retained only to identify the earlier evidence. The earlier selected features are labelled `unverified_historical_selection_not_linked_to_revised_AUC`. A row containing supplied aggregate values does not establish that its earlier model produced those values. Later workbook/scorecard overrides that conflict with your screenshots were not used.

Blank means unavailable, unverified or mathematically undefined, never zero. `1_Missing_metric_evidence.csv` explains each blank requested metric. Some available OOF values use fewer than five completed folds; the exact number is reported in `dev_oof_available_folds`. Decimal precision from the original saved tables is retained; screenshots are preserved to their displayed precision.

## 1. Complete results

`1_Complete_results.csv` contains 157 versioned records: 32 original final matrix identities, 64 classical-extension identities, 34 final Stability registered identities, four historical corrected CLIP pipelines, six Stability CLIP pipelines, eight screenshot voting pipelines, eight additional cross-dataset voting sensitivity pipelines, and one screenshot-only Stability LLM-then-mRMR entry. Registered but unavailable cells remain visible. Pilots, interrupted attempts and superseded Stability v2 folds are excluded. The separately trained cross-dataset voting protocol is distinguished from the screenshot voting protocol. Classical run names with different implementations are also distinguished.

DEV OOF AUC is calculated by pooling saved validation predictions. `dev_fold_auc_mean` is separately labelled and never substituted for pooled OOF AUC. The original legacy matrix does not preserve pooled OOF probabilities. All 36 `full_baseline_v1` saved DEV prediction files inspected contain final-model in-sample probabilities, so their values are not reported as OOF. The final Stability registry includes incomplete fold coverage; those pooled values explicitly describe only the available folds.

The final Stability validation-fold sizes, chronologically, are **204,567; 203,798; 205,980; 201,466; 202,820**. Their date ranges, training sizes and identity hashes are in `1_Stability_validation_fold_sizes.csv`. These total {sum(f['validation_rows'] for f in foldrows):,} OOF validation rows; the initial training period is not an OOF validation period.

KS, Brier and Capture@10 refer to OOT. Capture@10 is the fraction of positive cases in the highest-scored decile. Newly calculated voting capture uses the highest `ceil(0.10*n)` probabilities with stable row-order tie handling. Original saved capture values are copied under their source protocol. Score PSI uses different saved references across protocols: legacy matrix and both historical/Stability CLIP use full-DEV final-model in-sample scores; combination and cross-dataset voting analysis use pooled DEV OOF; final Stability uses available DEV OOF folds and frozen bins. The CSV identifies the reference. PSI values with different references are not interchangeable.

Feature counts are actual final selected columns, not the requested K; PCA counts refer to components. PCA's recurring labels PC1 through PCk do not establish identical fitted components across folds, so original-feature Nogueira/Jaccard are not applicable and are blank. A full-DEV resource stop that supplies no completed selection is blank rather than the upstream zero sentinel. Nogueira is recomputed from fold-selection indicators with the finite-sample variance correction: `1 - sum_j Var_sample(Z_j) / (mean_k*(1-mean_k/P))`. Jaccard is the mean over unordered pairs of available fold sets. `stability_universe_count` supplies P. Selecting the entire universe makes that Nogueira denominator zero; the conventional saved value of 1 is not treated as a defined Nogueira estimate here. The fixed CLIP pool denominator is 60 for LR and 100 for CatBoost. Historical Home Credit hybrid Nogueira values covered by the prior pool-denominator exclusion are blank.

## 2. Headline selection

`2_Headline_selection.csv` lists both headline families for all six dataset/model cases and the precise screenshot scores. The scores shown are **OOT AUC**. The saved evidence does **not authenticate whether the final cross-protocol headline winners were chosen using DEV or OOT AUC**. The combination DEV review explicitly records no DEV-based configuration selection, removal, tuning or reordering. No blanket claim that these twelve headline choices were DEV-selected can be supported.

The Home Credit LR mRMR score in your screenshot has no matching saved prediction identity in the inspected evidence. The revised Pure LLM and LendingClub LLM-then-mRMR scores likewise lack matching probability-level evidence. `2_Headline_paired_inference_as_supplied.csv` preserves the screenshot deltas, confidence limits and Holm p-values as supplied values only. Their raw p-values, exact testing method and full multiplicity family were not supplied, and they are not authenticated inferential results in this export.

## 3. Pure LLM eligibility

For the documented original Home Credit and LendingClub pipelines, **yes: Pure LLM candidates were IV-screened**. The registry specifies `min_iv=0.01`, `max_iv_for_leakage=0.5`, `encode=true`, after a training-slice missingness filter at 0.95. `LLMSelector.fit` uses that fold's training X and y for IV/WOE, then builds prompt statistics from the resulting training candidate frame. Thus statistics describe the post-IV/WOE representation. Final OOT selection uses full DEV for that operation. Validation and OOT do not supply those IV or prompt calculations. Shared cache reuse is tied to the training metadata identity; it does not mean one target-free global order was used across all folds.

For final Stability supplement v3, **no IV screening was used**: `iv_filter_kwargs={{}}`. A target-free availability filter retained 1,068 of 1,959 predictors using only missing rates in the earliest DEV training fold (200,661 rows, 2019-01-01 to 2019-03-28), retaining missing rate <=0.90. The prompt contains names/descriptions and excludes row statistics, targets, IV, performance and OOT information. One sealed ranking is reused across the five folds and OOT at K=20/40. These are verified implementation distinctions; the execution provenance of the revised screenshot AUCs is not available.

## 4. Full-feature references

The final Stability full-feature LR and CatBoost identities register the final split, all 1,959 classical predictors and the frozen model settings. **Neither has a numerical final OOT result**. Their unavailable status is inherited from all five authenticated DEV resource stops. No inherited successful full-feature score was substituted. Accordingly, numerical like-for-like baseline agreement cannot be confirmed.

The classical full universe is 1,959; the LLM supplement v3 eligibility universe is 1,068. Final Stability OOT preprocessing uses the sparse amendment preserving numeric mean imputation, centered scaling, categorical missing token and one-hot semantics. The baseline's registered settings can be identified, but a completed baseline execution under those settings does not exist. `4_Full_feature_references.csv` also identifies the successful Home Credit/LendingClub `full_baseline_v1` references inherited by the later classical review.

## 5. Voting comparisons and actual tests

The screenshot voting execution is `prompt4b_voting_pipeline`, on Home Credit's 529-feature universe, with inference version `prompt5_powered_comparisons_v1`. The saved implementation/provenance identifies **RF relevance and absolute-correlation redundancy**, not the MI-based mRMR label in the screenshot. The externally supplied complete LLM ranking and repaired corrected CLIP ranking were used for the normalized-rank voting. The historical mRMR comparator and sequential refinements carry their saved RF/correlation identity.

The actual family contains **24 comparisons**: two models × four voting variants × three comparators. The comparators are full-pool statistical mRMR, LLM-then-mRMR, and LLM-then-corrected-CLIP-then-mRMR. Every pair has 120,053 aligned OOT borrowers, no unmatched rows and no target mismatch. Each comparison ran two-sided paired DeLong for AUC plus paired stratified bootstrap confidence intervals for AUC and KS, with 2,000 resamples and seed 20260706. Holm adjustment uses all 24 DeLong p-values together. There is no KS p-value test in this family. The separate original 12-comparison family is not silently combined with voting; no new inferential tests were run for this export.

`5_Actual_test_inventory.csv` lists every actual comparison. `5_Voting_comparisons.csv` provides numeric inference for the 20 comparisons whose comparator values do not conflict with the screenshots. The four historical LR full-pool comparisons remain listed, but their conflicting comparator AUC, delta and inference are blank. Four additional rows show the arithmetic difference against the screenshot LR reference AUC 0.769890, with `actual_test_performed=false` and no CI or p-value. The old test results cannot be attached to this changed comparator. Existing Holm values for the other 20 rows retain the original 24-test family.

## 6. CLIP improvement

`6_CLIP_improvement_before_after.csv` identifies the historical Home Credit comparison: LLM-then-mRMR versus LLM-then-corrected-CLIP-then-mRMR, with both versions. It supplies the saved before/after OOT AUC, Jaccard and score PSI and their changes. These are the source of the screenshot's approximate Jaccard changes +0.325 (LR) and +0.499 (CatBoost).

Historical before/after **Nogueira is unavailable as a validated result**. The earlier reproducibility audit excluded the saved Nogueira/Kuncheva scalars because they used N=529 rather than the authenticated 60/100 candidate pools; no validated replacement was found. Those excluded numbers are not copied.

`6_CLIP_Stability_2024_results.csv` describes the separate `stability_clip_experiment_v1`: all three source-to-Stability directions × both models, including AUC, PSI and the six screenshot stability pairs. Their Jaccard and Nogueira values match independent calculations from the saved five fold-feature sets. These six final pipelines are not the historical Home Credit before/after pair.

## 7. Implementation, tuning, features and calls

`7_Implementation_settings.csv` contains the frozen model/selector parameters and distinguishes legacy Boruta/RFE from canonical versions. Canonical Boruta uses BorutaPy, a random forest configured with 500 trees and depth 6, automatic Boruta tree count, 10 iterations, percentile 100, alpha 0.05, two-step correction, seed 42 and confirmed-only selection without padding. The legacy Boruta registry uses 15 iterations. Canonical CatBoost RFE uses 500 iterations, depth 6, learning rate 0.05 and repeatedly removes floor(20% of the surviving features), at least one and capped to stop at K. Legacy RFE uses the integer step 10. These internal estimators differ from the final 1,500-iteration CatBoost classifier.

Final LR uses liblinear, 1,000 iterations, balanced class weights and seed 42. Final CatBoost settings include depth 10, learning rate 0.01, L2 95, minimum leaf data 290, column sampling 0.9, random strength 0.125, Depthwise growth, Newton leaf estimation, Bernoulli bootstrap, subsample 0.55 and balanced class weights. Early stopping 150 is configured, but the documented frozen final-model paths fit without an evaluation set. K=20/40 and pools P=100/200/300 are protocol settings. IV-then-Boruta retains natural confirmed support, so its actual feature count can exceed those fixed K values. Six early DEV pilot cells established heavy-selector feasibility. An originating model/budget hyperparameter-search history was not identified; no optimality or DEV-tuning provenance is inferred. `7_Tuning_provenance.csv` records those boundaries.

Preprocessing is fit at the training boundary. Selection encoders use one column per original variable, numeric median imputation and deterministic categorical codes with unseen values -1. Final models use numeric mean imputation/scaling and categorical missing-token plus one-hot encoding; CatBoost receives the resulting encoded matrix. The final sparse Stability amendment preserves these semantics. `7_Feature_engineering_summary.csv` describes each feature universe. `7_Selected_feature_lists.csv` contains complete saved final and fold selections where available, including protocol, run identity and the revised-score linkage warning.

`7_LLM_call_and_retry_counts.csv` distinguishes historical generation/reuse accounting from actual attempts. The historical cost-audit scope reports 72 logical requests, 24 canonical recorded generations, six source generations, 30 recorded generations in total, and 48 local reuses. Those figures are not a complete attempt-by-attempt provider log or a count of every experiment in this repository; exact older retries cannot be recovered. The accepted Stability v3 ranking has two application attempts, one rejected response and one successful retry, without fallback; final Stability OOT makes zero LLM calls/regenerations. Provider transport retries and unlogged external ranking-generation attempts remain unknown. An extant cache-file count is not an API-call count.

## 8. Frozen-threshold metrics

`8_Frozen_threshold_metrics.csv` has all twelve headline pipelines. Four are verified from saved holdout probabilities at the saved DEV-selected threshold: Home Credit LR stable-core + LLM-fill, Home Credit CatBoost RFE, Stability LR IV-then-Boruta, and Stability CatBoost RFE. In those paths the final threshold maximizes KS/Youden on full-DEV in-sample predictions and is applied unchanged to OOT; it is not selected from pooled OOF predictions.

Six revised headline rows lack matching saved predictions, so their thresholds and threshold metrics are blank. The two LendingClub classical IV-then-Boruta headline rows are a distinct implementation gap: `combination_oot_evaluation_worker` fits the final model, then saves OOT probabilities and threshold-free metrics; it does not select/save a final DEV threshold or precision/recall/F1. The earlier repository-wide methodology statement about every final pipeline having a DEV threshold therefore does not cover this worker. No new threshold was invented for this export.

## Correctness and unresolved items

`1-8_Validation_checks.csv` records the performed checks. The six revised AUC identities, six screenshot inference rows, original headline selection criterion, historical CLIP Nogueira, exact older retry counts and eight missing headline threshold rows are unresolved as described above. The MI-based voting label conflicts with the saved RF/correlation implementation. The supplied Stability LLM-then-mRMR secondary KS row has no matching entry in the final 34-cell Stability registry. No conflicting historical values were substituted into those gaps. These limitations mean the package does not claim a fully verified, gap-free experimental result table.
'''
(OUT/'README_points_1_to_8.md').write_text(readme,encoding='utf-8')

# Portable, executed checks on the delivered tables (no source data/model access).
notebook_code='''from pathlib import Path
import pandas as pd
p = Path('.')
r = pd.read_csv(p/'1_Complete_results.csv')
f = pd.read_csv(p/'1_Stability_validation_fold_sizes.csv')
t = pd.read_csv(p/'8_Frozen_threshold_metrics.csv')
v = pd.read_csv(p/'5_Voting_comparisons.csv')
assert len(r)==157
assert not r.duplicated(['evidence_cohort','run_id']).any()
assert f.validation_rows.tolist()==[204567,203798,205980,201466,202820]
assert len(t)==12 and t.dev_selected_threshold.notna().sum()==4
z=v.dropna(subset=['voting_auc','comparator_auc','paired_delta_auc'])
assert ((z.voting_auc-z.comparator_auc-z.paired_delta_auc).abs()<1e-12).all()
assert v.actual_test_performed.sum()==24
print('PASS: identity, fold-size, threshold-coverage and voting-delta checks.')
'''
import contextlib,io,os
capture=io.StringIO();oldcwd=Path.cwd()
try:
    os.chdir(OUT)
    with contextlib.redirect_stdout(capture):exec(notebook_code,{})
finally:os.chdir(oldcwd)
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[
 {'cell_type':'markdown','id':'purpose','metadata':{},'source':['# Delivered evidence validation\n','Run beside the exported CSVs. This notebook validates the delivered tables; source-backed calculations are in 1-8_Build_evidence.py.']},
 {'cell_type':'code','id':'checks','metadata':{},'execution_count':1,'source':notebook_code.splitlines(True),'outputs':[{'output_type':'stream','name':'stdout','text':capture.getvalue().splitlines(True)}]}]}
(OUT/'1-8_Validation_notebook.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
(OUT/'1-8_Build_evidence.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
manifest=[]
for p in sorted(OUT.iterdir()):
    if p.is_file() and p.name!='1-8_Package_manifest.csv':manifest.append(dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()))
write('1-8_Package_manifest.csv',manifest)
zip_path=OUT.parent/'evidence_points_1_to_8.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file():z.write(p,p.name)
with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None
print('ZIP',zip_path,zip_path.stat().st_size,flush=True)
