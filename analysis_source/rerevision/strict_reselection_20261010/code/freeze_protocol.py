from engine import *
import platform,sklearn,scipy,psutil

def main():
 if (OUT/'protocol_freeze.json').exists():raise FileExistsError('Protocol already frozen')
 grids={f:C.GRIDS[f] for f in FAMILIES};assert sum(map(len,grids.values()))==59
 protocol=dict(created_utc=now(),stage='strict-clean full pre-2025 reselection',conditions=CONDITIONS,collections=['BAYC','MAYC'],groups=['metadata','image','early','late'],encoders=ENCODERS,families=FAMILIES,grids=grids,development_folds=FOLDS,training_start='2022-01-01',training_end='2025-01-01',evaluation_start='2025-01-01',evaluation_end_exclusive='2026-04-14',time_zone='UTC',target='Unchanged archived LRP; no rebuilding market reference after cleaning',filters=dict(strict='Existing base_clean AND NOT observed_multi_nft_tx AND NOT reverse_pair_24h',strict_no_wei='strict AND price_eth > 1e-18*(1+1e-9)'),apply_filters='Both training and development validation in each fold',representation='All feature groups use token-level transformed-target means and sale-count weights for squared-loss models. PLS uses signed weighted centering; SVR retains all transaction responses and the regularized intercept.',tree_limit='All trees consistently use counted training across metadata/image/early. This is an explicitly different algorithm from transaction-row tree fitting, NOT a claim of implementation equivalence. Base condition is rerun in the same runtime; compare cleaning conditions within this implementation.',choice_rationale='Uniform representation across feature groups, retained image/early implementation, bounded memory. Not selected by evaluation performance.',preprocessing='Training-only NFT-document TF-IDF and weighted StandardScaler; metadata trees use raw TF-IDF, other feature blocks standardized as before. Constant dimensions removed. PCA 1024 on unique retained training tokens for SDXL_VAE and DreamSim, refit per condition and origin, shared between image and early.',pca_settings=I.pr.PCA_SETTINGS,selection_metric='Pooled original-LRP transaction RMSE over 2024 Q2-Q4',hyperparameter_ties='Reuse common.ordered_choices: existing family-specific complexity rule at 1e-12 MSE tolerance, boosted-library rounded MSE rule.',group_ties='Exact pooled MSE then existing encoder order and family order.',late='Select TF-IDF metadata across 8 learners; for all 7 encoders x 8 image learners select own hyperparameters, then evaluate image weight 0..1 by 0.1 on aligned development predictions; select across 616 branch/weight combinations per collection/condition.',final_failure='Try next development-ranked valid configuration within the chosen learner; record attempts before opening evaluation labels; if no within-family configuration succeeds, stop and amend rather than silently change cohort.',evaluation_access='Development code never opens evaluation labels. Record complete candidate rankings and selection freeze before final fits; record final fit resolutions before loading evaluation labels.',evaluation_sets=['common base evaluation rows','common strict evaluation rows'],metrics=['transaction RMSE','transaction MAE','transaction R2','equal-token RMSE'],inference=dict(replicates=2000,seed=SEED,token_cluster=True,calendar_block_days=14,calendar_anchor='2025-01-01',confidence=.95,exploratory=True,model_selection_uncertainty_included=False,multiplicity_adjusted=False),prior_evaluation_inspection=True,expected_development_fits=15930,expected_late_weight_scores=3696,threads=3,workers=2,source_plan_sha256=sha(AUDIT/'Analysis_Plan.md'),python=sys.version,versions=dict(sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,xgboost=C.xgboost.__version__,lightgbm=C.lightgbm.__version__),historical_environment_limit='Current sklearn 1.8.0 versus archived 1.9.0. All three cleaning conditions rerun under current environment; differences from archived baseline can include implementation/version effects.')
 inputs=[E.REGISTRY_PATH,E.cr.base.OLD/'metadata_normalized.jsonl',ROOT/'rerevision/Revised_Manuscript.docx',ROOT/'rerevision/Comments.txt']
 for c in ['BAYC','MAYC']:
  inputs += [E.cr.base.TARGET/f'{c.lower()}_development_targets.jsonl',E.cr.base.TARGET/f'{c.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl',E.cr.base.OLD/f'{c.lower()}_trade_eligibility.jsonl']
  for enc in ENCODERS:inputs += [ROOT/REG[enc][c]['matrix'],ROOT/REG[enc][c]['manifest']]
 originals={str(Path(r['path'])).replace('\\','/'):r for r in readj(AUDIT/'baseline_manifest.json')['files'] if r.get('exists')}
 records=[]
 for p in dict.fromkeys(inputs):
  rel=p.relative_to(ROOT).as_posix();digest=sha(p);old=originals.get(rel)
  if old:assert old['sha256']==digest,rel
  records.append(dict(path=rel,sha256=digest,bytes=p.stat().st_size,baseline_verified=bool(old)))
 sources=[]
 for module in list(sys.modules.values()):
  p=getattr(module,'__file__',None)
  if p:
   p=Path(p)
   if p.suffix=='.py' and p.is_relative_to(ROOT) and '.venv' not in str(p):sources.append(p)
 sources+=list((OUT/'code').glob('*.py'))
 protocol['inputs']=records;protocol['sources']=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in dict.fromkeys(sources)]
 cohorts=[]
 for c in ['BAYC','MAYC']:
  base=cohort(c,'base');base_map=dict(zip(base['rowids'],base['y']))
  for cond in CONDITIONS:
   d=cohort(c,cond);assert all(base_map[r]==y for r,y in zip(d['rowids'],d['y']))
   for origin,end in ORIGINS:
    ti=np.flatnonzero(d['times']<origin);vi=np.flatnonzero((d['times']>=origin)&(d['times']<end))
    cohorts.append(dict(collection=c,condition=cond,origin=origin,end=end,train_n=len(ti),train_tokens=len(np.unique(d['tokens'][ti])),validation_n=len(vi),validation_tokens=len(np.unique(d['tokens'][vi])),training_row_sha256=ash(d['rowids'][ti]),validation_row_sha256=ash(d['rowids'][vi]),nominal_wei_training=int(np.sum(d['prices'][ti]<=1e-18*(1+1e-9))),target_parameters=json.dumps(C.fit_target_transform(d['y'][ti]))))
 writecsv(OUT/'development_cohorts.csv',cohorts)
 savej(OUT/'protocol_freeze.json',protocol)
 savej(OUT/'amendment_01_uniform_implementation.json',dict(created_utc=now(),reason='Step 4 showed tree training implementation dependence. Step 5 requires uniform training representation across feature groups.',decision=protocol['representation'],tree_limit=protocol['tree_limit'],runtime=protocol['versions'],new_evaluation_results_loaded=False,original_analysis_plan_sha256=protocol['source_plan_sha256'],protocol_sha256=sha(OUT/'protocol_freeze.json')))
 print(json.dumps(dict(inputs=len(records),source_modules=len(protocol['sources']),expected_fits=15930,cohorts=cohorts),indent=2))
if __name__=='__main__':main()
