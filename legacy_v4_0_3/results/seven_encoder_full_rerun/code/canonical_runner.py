"""Fresh, resumable canonical regression/fusion rerun. Never writes revision/.

No previously fitted results are inputs. Runtime checkpoints belong to this run.
Evaluation is a fixed retrospective out-of-time evaluation window, not selection data.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import psutil
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from audit_inputs import ROOT, REV, OUT, ENCODERS, COLLECTIONS, read, lines, sha, now, save, csvsave

sys.path.insert(0, str(REV / "code"))
import metadata_baseline_pipeline as base
import full_metadata_comparison as meta
import encoder_development_v4 as vision


def compatible_onehot(*args, **kwargs):
    if "sparse" in kwargs:
        kwargs["sparse_output"] = kwargs.pop("sparse")
    return OneHotEncoder(*args, **kwargs)


base.OneHotEncoder = compatible_onehot
FAMILIES = ["Ridge", "ElasticNet", "PLS", "LinearSVR"]
MFAMILIES = FAMILIES + ["HistGradientBoosting"]
CROSS_ORDER = ["SigLIP2", "CLIP", "DINOv2", "SAM", "SDXL_VAE", "DreamSim", "AIM"]
GRIDS = meta.candidates(base.effective_spec())
WEIGHTS = [i / 10 for i in range(11)]
FINAL_Q = [("2024-04-01", "2024-07-01"), ("2024-07-01", "2024-10-01"), ("2024-10-01", "2025-01-01")]
SAMPLE_DIRS = {"original": "target_pipeline_20260909", "one_wei": "one_wei_sensitivity_targets_20260909"}


class ResourceBlocked(RuntimeError):
    pass


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def check_training(data, train, valid, origin):
    assert len(train) and np.intersect1d(train, valid).size == 0
    assert all(t < "2025-01-01" and t < origin for t in data["times"][train])
    if len(valid):
        assert data["times"][train].max() < data["times"][valid].min()
        assert all(t < "2025-01-01" for t in data["times"][valid])
    assert np.isfinite(data["y"][train]).all()
    assert np.isfinite(data["y"][valid]).all()


def choose_metrics(metrics, order):
    eligible = [k for k in order if metrics.get(k, {}).get("available", False)]
    if not eligible:
        raise RuntimeError("No valid candidate: never silently substitute another cohort/model.")
    best = min(metrics[k]["sse"] / metrics[k]["n"] for k in eligible)
    return next(k for k in eligible if metrics[k]["sse"] / metrics[k]["n"] <= best + 1e-12)


def augmented_order():
    return ["late_"+m+"_"+v for m in MFAMILIES for v in FAMILIES] + ["early_"+f for f in FAMILIES]


def phase_model(family, config, phase):
    model = base.model_pipeline(family, config).named_steps["model"]
    if family == "ElasticNet" and phase == "final":
        model.set_params(max_iter=5000)
    return model


def load_development(collection, encoder=None, sample="original"):
    rows = lines(REV / SAMPLE_DIRS[sample] / f"{collection.lower()}_development_targets.jsonl")
    assert all(r["time"] < "2025-01-01" and r["split"] == "development" for r in rows)
    metadata = {(r["collection"], int(r["token_id"])): r for r in lines(base.OLD / "metadata_normalized.jsonl")}
    cols = base.FEATURES + (["generation"] if collection == "MAYC" else [])
    split_dir = "image_validation_20260909" if sample == "original" else SAMPLE_DIRS[sample]
    data = dict(y=np.asarray([r["y_log_relative_price"] for r in rows], dtype=np.float64),
                tokens=np.asarray([r["token_id"] for r in rows],dtype=np.int64),
                rowids=np.asarray([r["source_row"] for r in rows],dtype=np.int64),
                times=np.asarray([r["time"] for r in rows],dtype=object),columns=cols,
                split=read(REV / split_dir / "image_ready_split_manifest_v1.json")[collection])
    data["rowmap"] = {int(v):i for i,v in enumerate(data["rowids"])}
    assert len(data["rowmap"]) == len(rows)
    assert set(data["rowmap"]) == set(data["split"]["development_source_rows"])
    if encoder:
        info = read(OUT / "embedding_registry.json")[encoder][collection]
        ids = np.asarray(info["token_ids"],dtype=np.int64)
        mapping = {int(t):i for i,t in enumerate(ids)}
        if set(data["tokens"]) - set(mapping):
            raise RuntimeError("Embedding cannot reproduce audited cohort.")
        raw_image=np.load(ROOT/info["matrix"],mmap_mode="r",allow_pickle=False)
        # Native representation is not native arithmetic dtype: the canonical loader computes in float64.
        if raw_image.size*8 + 1.5*2**30 > psutil.virtual_memory().available:
            raise ResourceBlocked("Insufficient memory for canonical float64 feature loader.")
        data.update(image=raw_image.astype(np.float64),feature_tokens=ids,
                    X=np.asarray([[metadata[collection,int(t)][c] for c in cols] for t in ids],dtype=object),
                    index=np.asarray([mapping[int(t)] for t in data["tokens"]],dtype=np.int64))
        assert np.array_equal(ids[data["index"]], data["tokens"])
        assert data["image"].shape[1] == info["dimensions"]
    else:
        data["X"] = np.asarray([[metadata[collection,int(t)][c] for c in cols] for t in data["tokens"]],dtype=object)
    return data


class FreshRunner(meta.Runner):
    def __init__(self):
        super().__init__(OUT, 3)
        self.registry = read(OUT / "embedding_registry.json")
        self.completed_new_fits = 0
        self.resource_blocks = []

    def memory_gate(self):
        if psutil.virtual_memory().available < 1.5 * 2**30:
            raise ResourceBlocked("Less than canonical minimum 1.5 GiB available memory.")

    def prepare_checked(self, data, train, valid, mode, origin):
        check_training(data, train, valid, origin)
        if mode != "metadata":
            d = data["image"].shape[1]
            unique = len(np.unique(data["index"][train]))
            # Conservative simultaneous arrays in unchanged prepare + EN full/Gram paths.
            needed = ((4*len(data["image"])+2*unique+2*len(train))*d*8 + 2*d*d*8)
            available = psutil.virtual_memory().available
            if needed + 1.5*2**30 > available:
                raise ResourceBlocked(f"Native {d}-column {mode} working-set bound {needed/2**30:.2f} GiB exceeds available {available/2**30:.2f} GiB.")
            p = vision.prepare(data, train, valid, mode)
            if not len(valid):
                p["vi"]=data["index"][train[:20]]
            assert np.isfinite(p["z"]).all()
        else:
            p = self.prepare(data["X"],train,valid,data["columns"])
            assert np.isfinite(p["raw_train"]).all() and np.isfinite(p["ztrain"]).all()
        self.log(dict(stage="preprocessing_fit",mode=mode,origin=origin,
                      train_max_time=str(data["times"][train].max()),training_rows=len(train),
                      training_row_sha256=array_sha(data["rowids"][train]),validation_rows=len(valid)))
        return p

    def fit_checked(self, data, train, valid, origin, p, mode, family, config, phase, folder, context):
        check_training(data, train, valid, origin)
        folder.mkdir(parents=True,exist_ok=True)
        number = GRIDS[family].index(config)
        stem = f"{family}_{number}"
        result_path = folder/(stem+".json")
        if result_path.exists():
            result = read(result_path)
            assert result["config"] == config and result["mode"] == mode and result["phase"] == phase
            assert result["training_row_sha256"] == array_sha(data["rowids"][train])
            assert result["validation_row_sha256"] == array_sha(data["rowids"][valid])
            pred = np.load(folder/(stem+".npy"),allow_pickle=False) if result["valid"] else None
            if result["valid"]:
                assert sha(folder/(stem+".npy")) == result["prediction_sha256"]
                assert np.isfinite(pred).all()
            return pred,result
        started = time.perf_counter()
        model = offset = pred = None
        info = dict(valid=False)
        try:
            if mode == "metadata":
                if family == "PLS" and p["rank"] < config["components"]:
                    raise ValueError("PLS components exceed conservative verified training rank.")
                model = phase_model(family,config,phase)
                tr = p["raw_train"] if family == "HistGradientBoosting" else p["ztrain"]
                va = p["raw_valid"] if family == "HistGradientBoosting" else p["zvalid"]
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(tr,data["y"][train])
                    pred = np.asarray(model.predict(va if len(va) else tr[:20])).reshape(-1)
                valid_fit = np.isfinite(pred).all() and not any(issubclass(w.category,ConvergenceWarning) for w in caught)
                if family == "PLS":
                    valid_fit &= len(model.n_iter_) == config["components"] and np.isfinite(model.coef_).all()
                info = dict(valid=bool(valid_fit),warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught],
                            n_iter=np.asarray(getattr(model,"n_iter_",[])).tolist())
            else:
                # Preserve the canonical source's fitting code, replacing only the final EN iteration limit.
                original_builder = base.model_pipeline
                def builder(f,c):
                    pipe = original_builder(f,c)
                    if f == "ElasticNet" and phase == "final":
                        pipe.named_steps["model"].set_params(max_iter=5000)
                    return pipe
                base.model_pipeline = builder
                try:
                    model,offset,pred,info = vision.fit_prepared(p,family,config)
                finally:
                    base.model_pipeline = original_builder
            if info["valid"]:
                assert pred is not None and np.isfinite(pred).all()
                if len(valid):
                    info.update(sse=float(np.sum((data["y"][valid]-pred)**2)),n=len(valid))
                # Persist fitted states for outer refit and final full-development refit only.
                if context["role"] in ("outer","refit"):
                    if mode == "metadata":
                        from sklearn.pipeline import Pipeline
                        bundle = Pipeline(p["encoding"].steps + [("scale","passthrough" if family == "HistGradientBoosting" else p["scaler"]),("model",model)])
                    else:
                        bundle = dict(state=p["state"],model=model,offset=offset,mode=mode)
                    model_path = folder/(stem+".joblib")
                    if model_path.exists():
                        raise RuntimeError("Refusing to overwrite fitted model.")
                    joblib.dump(bundle,model_path,compress=3)
                    info.update(model_path=str(model_path.relative_to(OUT)),model_sha256=sha(model_path))
                with (folder/(stem+".npy")).open("xb") as f:
                    np.save(f,pred,allow_pickle=False)
                info["prediction_sha256"]=sha(folder/(stem+".npy"))
        except (ResourceBlocked, MemoryError):
            raise
        except Exception as error:
            info = dict(valid=False,error=repr(error))
        info.update(family=family,config=config,mode=mode,phase=phase,seconds=time.perf_counter()-started,
                    training_row_sha256=array_sha(data["rowids"][train]),validation_row_sha256=array_sha(data["rowids"][valid]),
                    training_max_time=str(data["times"][train].max()),origin=origin,training_rank_lower_bound=p["rank"])
        save(result_path,info)
        self.completed_new_fits += 1
        self.log(dict(stage="fresh_fit_completed",**context,**info))
        del model
        return pred,info

    def quarter(self,data,collection,encoder,begin,end,mode,sample="original",phase="development"):
        train = np.flatnonzero(data["times"] < begin)
        valid = np.flatnonzero((data["times"] >= begin)&(data["times"] < end))
        assert len(valid)>=200 and len(np.unique(data["tokens"][valid]))>=100
        key = encoder or "metadata"
        folder = OUT/"checkpoints"/sample/collection/key/phase/("quarter_"+begin)/mode
        families = MFAMILIES if mode == "metadata" else FAMILIES
        all_paths = [folder/f"{f}_{j}.json" for f in families for j in range(len(GRIDS[f]))]
        complete = all(p.exists() for p in all_paths)
        p = None if complete else self.prepare_checked(data,train,valid,mode,begin)
        results = {}
        for family in families:
            for config in GRIDS[family]:
                pred,info = self.fit_checked(data,train,valid,begin,p,mode,family,config,phase,folder,
                                            dict(collection=collection,encoder=encoder,role="inner",quarter=begin,sample=sample))
                results[(family,meta.config_id(config))]=(pred,info)
        del p
        gc.collect()
        return dict(results=results,valid=valid,begin=begin,end=end)

    def refit_from_scores(self,data,collection,encoder,mode,family,quarters,train,valid,origin,role,sample="original",phase="development"):
        scores = []
        for config in GRIDS[family]:
            parts = [q["results"][(family,meta.config_id(config))][1] for q in quarters]
            ok = all(p["valid"] for p in parts)
            scores.append(dict(config=config,valid=ok,mse=sum(p["sse"] for p in parts)/sum(p["n"] for p in parts) if ok else None))
        candidates = meta.ordered_valid(family,scores)
        if not candidates:
            return None,dict(available=False,inner_scores=scores)
        folder = OUT/"checkpoints"/sample/collection/(encoder or "metadata")/phase/(role+"_"+origin)/mode
        p = None
        attempts = []
        for choice in candidates:
            result_path = folder/f"{family}_{GRIDS[family].index(choice['config'])}.json"
            if not result_path.exists() and p is None:
                p = self.prepare_checked(data,train,valid,mode,origin)
            pred,info = self.fit_checked(data,train,valid,origin,p,mode,family,choice["config"],phase,folder,
                                        dict(collection=collection,encoder=encoder,role=role,sample=sample))
            attempts.append(info)
            if info["valid"]:
                result = dict(available=True,selected_config=choice["config"],inner_scores=scores,attempts=attempts)
                if len(valid):
                    result["metrics"] = base.metrics(data["y"][valid],pred,data["tokens"][valid])
                del p
                gc.collect()
                return pred,result
        del p
        gc.collect()
        return None,dict(available=False,inner_scores=scores,attempts=attempts)

    def metadata_development(self,collection,sample="original"):
        output = OUT/"development"/sample/collection/"metadata"
        summary_path = output/"summary.json"
        if summary_path.exists():
            return read(summary_path)
        data = load_development(collection,sample=sample)
        predictions = {f:[] for f in MFAMILIES}
        ys,tokens,rows=[],[],[]
        reports=[]
        for fold in data["split"]["folds"]:
            train=np.asarray([data["rowmap"][int(r)] for r in fold["train_source_rows"]])
            valid=np.asarray([data["rowmap"][int(r)] for r in fold["validation_source_rows"]])
            origin=data["times"][valid].min()[:10]
            # Use specified fold boundary, not first observed trade time.
            origin={"D1":"2023-01-01","D2":"2023-07-01","D3":"2024-01-01"}[fold["name"]]
            qs=[self.quarter(data,collection,None,q["train_end_exclusive"][:10],q["validation_end_exclusive"][:10],"metadata",sample)
                for q in fold["inner_quarter_schedule"]]
            report=dict(fold=fold["name"],families={})
            fold_arrays=dict(y=data["y"][valid],tokens=data["tokens"][valid],source_rows=data["rowids"][valid])
            for family in MFAMILIES:
                pred,result=self.refit_from_scores(data,collection,None,"metadata",family,qs,train,valid,origin,"outer",sample)
                report["families"][family]=result
                if pred is not None:
                    predictions[family].append(pred)
                    fold_arrays[family]=pred
            ys.append(data["y"][valid]);tokens.append(data["tokens"][valid]);rows.append(data["rowids"][valid])
            reports.append(report)
            output.mkdir(parents=True,exist_ok=True)
            path=output/(fold["name"]+".npz")
            if not path.exists():
                np.savez_compressed(path,**fold_arrays)
            path=output/(fold["name"]+".json")
            if not path.exists():
                save(path,report)
        y,tok=np.concatenate(ys),np.concatenate(tokens)
        pooled={f:dict(available=len(predictions[f])==3,**(base.metrics(y,np.concatenate(predictions[f]),tok) if len(predictions[f])==3 else {})) for f in MFAMILIES}
        summary=dict(collection=collection,sample=sample,pooled=pooled,selected=choose_metrics(pooled,MFAMILIES),folds=reports)
        save(summary_path,summary)
        return summary

    def encoder_development(self,collection,encoder,metadata_summary,sample="original"):
        output=OUT/"development"/sample/collection/encoder
        if (output/"summary.json").exists():
            return read(output/"summary.json")
        data=load_development(collection,encoder,sample)
        md=load_development(collection,sample=sample)
        all_predictions={k:[] for k in ["image_"+f for f in FAMILIES]+augmented_order()}
        ys,tokens=[],[]
        reports=[]
        for fold,meta_report in zip(data["split"]["folds"],metadata_summary["folds"]):
            train=np.asarray([data["rowmap"][int(r)] for r in fold["train_source_rows"]])
            valid=np.asarray([data["rowmap"][int(r)] for r in fold["validation_source_rows"]])
            origin={"D1":"2023-01-01","D2":"2023-07-01","D3":"2024-01-01"}[fold["name"]]
            qs={mode:[self.quarter(data,collection,encoder,q["train_end_exclusive"][:10],q["validation_end_exclusive"][:10],mode,sample)
                      for q in fold["inner_quarter_schedule"]] for mode in ("image","early")}
            mq=[self.quarter(md,collection,None,q["train_end_exclusive"][:10],q["validation_end_exclusive"][:10],"metadata",sample)
                for q in fold["inner_quarter_schedule"]]
            meta_pred=np.load(OUT/"development"/sample/collection/"metadata"/(fold["name"]+".npz"))
            assert np.array_equal(meta_pred["source_rows"],data["rowids"][valid])
            report=dict(fold=fold["name"],candidates={})
            arrays=dict(y=data["y"][valid],tokens=data["tokens"][valid],source_rows=data["rowids"][valid])
            image_selected={}
            for mode in ("image","early"):
                for family in FAMILIES:
                    pred,result=self.refit_from_scores(data,collection,encoder,mode,family,qs[mode],train,valid,origin,"outer",sample)
                    name=mode+"_"+family
                    report["candidates"][name]=result
                    if pred is not None:
                        arrays[name]=pred
                    if mode=="image" and result["available"]:
                        image_selected[family]=result["selected_config"]
            for mf in MFAMILIES:
                mr=meta_report["families"][mf]
                for vf in FAMILIES:
                    name="late_"+mf+"_"+vf
                    if not mr["available"] or vf not in image_selected:
                        report["candidates"][name]=dict(available=False)
                        continue
                    mc,vc=mr["selected_config"],image_selected[vf]
                    parts=[(data["y"][iq["valid"]], m["results"][(mf,meta.config_id(mc))][0],iq["results"][(vf,meta.config_id(vc))][0])
                           for m,iq in zip(mq,qs["image"])]
                    weight,scores=vision.late_weights(parts,WEIGHTS)
                    pred=(1-weight)*meta_pred[mf]+weight*arrays["image_"+vf]
                    arrays[name]=pred
                    report["candidates"][name]=dict(available=True,metadata_config=mc,image_config=vc,image_weight=weight,
                                                     weight_scores=scores,metrics=base.metrics(data["y"][valid],pred,data["tokens"][valid]))
            for name in all_predictions:
                if name in arrays:
                    all_predictions[name].append(arrays[name])
            ys.append(data["y"][valid]);tokens.append(data["tokens"][valid]);reports.append(report)
            output.mkdir(parents=True,exist_ok=True)
            if not (output/(fold["name"]+".json")).exists():
                save(output/(fold["name"]+".json"),report)
            if not (output/(fold["name"]+".npz")).exists():
                np.savez_compressed(output/(fold["name"]+".npz"),**arrays)
            meta_pred.close()
        y,tok=np.concatenate(ys),np.concatenate(tokens)
        pooled={k:dict(available=len(parts)==3,**(base.metrics(y,np.concatenate(parts),tok) if len(parts)==3 else {})) for k,parts in all_predictions.items()}
        summary=dict(collection=collection,encoder=encoder,sample=sample,native_dimension=self.registry[encoder][collection]["dimensions"],
                     pooled=pooled,selected_image=choose_metrics(pooled,["image_"+f for f in FAMILIES]),
                     selected_augmented=choose_metrics(pooled,augmented_order()),folds=reports)
        save(output/"summary.json",summary)
        return summary

    def final_spec(self,collection,encoder,candidate,sample="original"):
        md=load_development(collection,sample=sample)
        data=load_development(collection,encoder,sample)
        all_rows=np.arange(len(data["y"]))
        empty=np.asarray([],dtype=int)
        parts=candidate.split("_")
        specs={}
        needed=[("metadata",parts[1]),("image",parts[2])] if parts[0]=="late" else [("early",parts[1])]
        for mode,family in needed:
            d=md if mode=="metadata" else data
            qs=[self.quarter(d,collection,None if mode=="metadata" else encoder,b,e,mode,sample,phase="final") for b,e in FINAL_Q]
            _,result=self.refit_from_scores(d,collection,None if mode=="metadata" else encoder,mode,family,qs,all_rows,empty,"2025-01-01","refit",sample,"final")
            if not result["available"]:
                raise RuntimeError(f"Frozen family {family} cannot refit: no evaluation-driven reselection permitted.")
            specs[mode]=result
        if parts[0]=="late":
            mc,vc=specs["metadata"]["selected_config"],specs["image"]["selected_config"]
            mq=[self.quarter(md,collection,None,b,e,"metadata",sample,"final") for b,e in FINAL_Q]
            iq=[self.quarter(data,collection,encoder,b,e,"image",sample,"final") for b,e in FINAL_Q]
            weight,scores=vision.late_weights([(data["y"][i["valid"]],m["results"][(parts[1],meta.config_id(mc))][0],i["results"][(parts[2],meta.config_id(vc))][0]) for m,i in zip(mq,iq)],WEIGHTS)
            specs.update(image_weight=weight,weight_scores=scores)
        return dict(encoder=encoder,candidate=candidate,components=specs,native_dimension=data["image"].shape[1])

    def freeze_all(self,summaries,metadata):
        specs=dict(created_utc=now(),selection_data_end_exclusive="2025-01-01",collections={})
        for collection in COLLECTIONS:
            ranked=sorted([(encoder,summaries[collection][encoder]["selected_augmented"],
                            summaries[collection][encoder]["pooled"][summaries[collection][encoder]["selected_augmented"]])
                           for encoder in CROSS_ORDER],key=lambda r:r[2]["sse"]/r[2]["n"])
            md=load_development(collection)
            family=metadata[collection]["selected"]
            mq=[self.quarter(md,collection,None,b,e,"metadata",phase="final") for b,e in FINAL_Q]
            _,mresult=self.refit_from_scores(md,collection,None,"metadata",family,mq,np.arange(len(md["y"])),np.asarray([],dtype=int),"2025-01-01","refit",phase="final")
            if not mresult["available"]:
                raise RuntimeError("Development-selected metadata family cannot refit.")
            encoder_specs={}
            for encoder in CROSS_ORDER:
                s=summaries[collection][encoder]
                encoder_specs[encoder]={
                    "augmented":self.final_spec(collection,encoder,s["selected_augmented"]),
                    "image":self.final_image(collection,encoder,s["selected_image"]),
                    "mandatory_early_ElasticNet":self.final_spec(collection,encoder,"early_ElasticNet")}
            specs["collections"][collection]=dict(metadata_family=family,metadata=mresult,selected_encoder=ranked[0][0],
                                                 selected_candidate=ranked[0][1],development_ranking=[dict(encoder=e,candidate=c,**m) for e,c,m in ranked],encoders=encoder_specs)
        path=OUT/"selected_primary_models.json"
        save(path,specs)
        save(OUT/"selection_freeze.json",dict(frozen_utc=now(),specification_sha256=sha(path),
                                             evaluation_predictions_generated=False))
        return specs

    def final_image(self,collection,encoder,candidate):
        data=load_development(collection,encoder)
        family=candidate.split("_",1)[1]
        qs=[self.quarter(data,collection,encoder,b,e,"image",phase="final") for b,e in FINAL_Q]
        _,result=self.refit_from_scores(data,collection,encoder,"image",family,qs,np.arange(len(data["y"])),np.asarray([],dtype=int),"2025-01-01","refit",phase="final")
        if not result["available"]:
            raise RuntimeError("Development-selected image family cannot refit.")
        return dict(encoder=encoder,candidate=candidate,native_dimension=data["image"].shape[1],components={"image":result})


def development_tables(metadata,summaries):
    tables={"metadata_model_selection.csv":[],"image_model_selection_all_encoders.csv":[],"early_fusion_all_encoders.csv":[],"late_fusion_all_encoders.csv":[]}
    for collection,m in metadata.items():
        for f,metric in m["pooled"].items():
            tables["metadata_model_selection.csv"].append(dict(collection=collection,model_family=f,selected=f==m["selected"],development_RMSE=metric.get("rmse"),
                available=metric["available"],outer_fold_configs=[r["families"][f].get("selected_config") for r in m["folds"]]))
        if len(summaries.get(collection,{}))!=len(ENCODERS):
            continue
        ranking=[]
        for encoder,s in summaries[collection].items():
            candidate=s["selected_augmented"]
            ranking.append(dict(collection=collection,encoder=encoder,native_dimension=s["native_dimension"],candidate=candidate,
                                development_RMSE=s["pooled"][candidate]["rmse"],development_MSE=s["pooled"][candidate]["sse"]/s["pooled"][candidate]["n"]))
            for name,metric in s["pooled"].items():
                mode=name.split("_",1)[0]
                row=dict(collection=collection,encoder=encoder,native_dimension=s["native_dimension"],candidate=name,
                         available=metric["available"],development_RMSE=metric.get("rmse"),selected_within_encoder=name==candidate,
                         selected_image=name==s["selected_image"],outer_fold_specs=[r["candidates"][name] for r in s["folds"]])
                dest={"image":"image_model_selection_all_encoders.csv","early":"early_fusion_all_encoders.csv","late":"late_fusion_all_encoders.csv"}[mode]
                tables[dest].append(row)
        ranking.sort(key=lambda r:(r["development_MSE"],CROSS_ORDER.index(r["encoder"])))
        for rank,row in enumerate(ranking,1):
            row.update(rank=rank,development_selected_Astar=rank==1)
        path=OUT/f"development_encoder_ranking_{collection}.csv"
        if not path.exists():
            csvsave(path,ranking)
    for name,rows in tables.items():
        if rows and not (OUT/name).exists():
            csvsave(OUT/name,rows)


def run_development(runner):
    metadata={}
    summaries={c:{} for c in COLLECTIONS}
    for collection in COLLECTIONS:
        metadata[collection]=runner.metadata_development(collection)
    for collection in COLLECTIONS:
        for encoder in ENCODERS:
            try:
                summaries[collection][encoder]=runner.encoder_development(collection,encoder,metadata[collection])
            except ResourceBlocked as error:
                runner.resource_blocks.append(dict(collection=collection,encoder=encoder,error=str(error)))
                runner.log(dict(stage="resource_blocked_not_excluded",collection=collection,encoder=encoder,error=str(error)))
    complete=all(len(summaries[c])==7 for c in COLLECTIONS)
    if complete:
        development_tables(metadata,summaries)
    if not complete:
        save(OUT/("resource_status_"+str(time.time_ns())+".json"),dict(utc=now(),complete=False,blocks=runner.resource_blocks,
                                                                    primary_selection_permitted=False))
        raise ResourceBlocked("All seven development candidates must complete before A* selection or evaluation.")
    return metadata,summaries


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["development","all"],default="development")
    args=parser.parse_args()
    manifest=read(OUT/"run_manifest.json")
    assert manifest["embedding_integrity_pass"] and manifest["cohort_integrity_pass"]
    for record in manifest["input_files"]+manifest["canonical_scripts"]:
        assert sha(ROOT/record["path"])==record["sha256"],f"Input changed: {record['path']}"
    config=dict(script_sha256=sha(Path(__file__)),grids=GRIDS,weights=WEIGHTS,
                source_files=manifest["canonical_scripts"],input_files=manifest["input_files"],threads=3,
                analysis_scripts=[dict(path=str(p.relative_to(OUT)),sha256=sha(p)) for p in sorted((OUT/"code").glob("*.py"))],
                EN_development_max_iter=20000,EN_final_max_iter=5000,
                solver_vision_and_early="counted_primal_svr_v3.CountedPrimalSVR",seed=20260908)
    path=OUT/"run_configuration.json"
    if path.exists():
        assert read(path)==config,"Cannot resume checkpoints with changed code/inputs/settings."
    else:
        save(path,config)
    runner=FreshRunner()
    runner.log(dict(stage="fresh_run_started",utc=now(),evaluation_window="fixed retrospective out-of-time evaluation window"))
    try:
        with threadpool_limits(limits=3):
            metadata,summaries=run_development(runner)
            if args.stage=="all":
                if not (OUT/"selection_freeze.json").exists():
                    runner.freeze_all(summaries,metadata)
                from evaluate_frozen import evaluate_all
                evaluate_all()
                runner.log(dict(stage="main_evaluation_completed_sensitivities_pending",utc=now()))
        runner.log(dict(stage="development_completed",utc=now()))
    except BaseException as error:
        runner.log(dict(stage="run_incomplete",error=repr(error),utc=now(),primary_results_certified=False))
        raise


if __name__=="__main__":
    main()
