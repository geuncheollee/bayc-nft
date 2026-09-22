"""Evaluation and inference; cannot select or fit any model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from audit_inputs import OUT, ROOT, REV, COLLECTIONS, ENCODERS, read, lines, sha, save, csvsave, now


def require_freeze(out=OUT):
    marker=read(out/"selection_freeze.json")
    path=out/"selected_primary_models.json"
    if marker["specification_sha256"]!=sha(path):
        raise RuntimeError("Frozen specification changed: evaluation is prohibited.")
    specs=read(path)
    if specs["selection_data_end_exclusive"]!="2025-01-01":
        raise RuntimeError("Invalid fitting/selection cutoff.")
    for c in COLLECTIONS:
        if set(specs["collections"][c]["encoders"])!=set(ENCODERS):
            raise RuntimeError("Seven-encoder development selection is incomplete.")
    return specs


def empirical_p(values):
    n=len(values)
    return min(1.,2*(min(int(np.count_nonzero(values<=0)),int(np.count_nonzero(values>=0)))+1)/(n+1))


def summarize_bootstrap(delta,utility,primary):
    result=dict(delta_CI95_lower=float(np.percentile(delta,2.5)),delta_CI95_upper=float(np.percentile(delta,97.5)),
                utility_CI95_lower=float(np.percentile(utility,2.5)),utility_CI95_upper=float(np.percentile(utility,97.5)),
                empirical_P_value=empirical_p(delta),inference="primary two-collection adjusted" if primary else "exploratory nominal")
    if primary:
        result.update(delta_CI975_lower=float(np.percentile(delta,1.25)),delta_CI975_upper=float(np.percentile(delta,98.75)),
                      utility_CI975_lower=float(np.percentile(utility,1.25)),utility_CI975_upper=float(np.percentile(utility,98.75)),
                      Bonferroni_P_value=min(1.,2*result["empirical_P_value"]))
    return result


def paired_samples(y,metadata,predictions,groups,rng,n=2000):
    """Exact paired cluster bootstrap via per-cluster SSE/count sufficient sums.

    This is resampling all transactions of each sampled cluster, not averaging
    cluster RMSEs. Identical bootstrap multiplicities apply to every prediction.
    """
    labels,inverse=np.unique(groups,return_inverse=True)
    counts=np.bincount(inverse).astype(np.float64)
    squared=np.column_stack([(y-metadata)**2]+[(y-p)**2 for p in predictions.values()])
    sums=np.column_stack([np.bincount(inverse,weights=squared[:,j]) for j in range(squared.shape[1])])
    assert np.isfinite(sums).all()
    rmse=np.empty((n,len(predictions)+1))
    for b in range(n):
        chosen=rng.choice(len(labels),size=len(labels),replace=True)
        rmse[b]=np.sqrt(sums[chosen].sum(axis=0)/counts[chosen].sum())
    delta=rmse[:,1:]-rmse[:,:1]
    utility=100*(rmse[:,:1]-rmse[:,1:])/np.maximum(rmse[:,:1],1e-12)
    return dict(rmse=rmse,delta=delta,utility=utility,cluster_count=len(labels),prediction_order=list(predictions))


def component_model(result):
    info=result["attempts"][-1]
    path=OUT/info["model_path"]
    if sha(path)!=info["model_sha256"]:
        raise RuntimeError(f"Fitted model changed after specification freezing: {path}")
    return joblib.load(path)


def metadata_predict(result,X):
    return np.asarray(component_model(result).predict(X)).reshape(-1)


def image_predict(result,X,image):
    import canonical_runner as cr
    bundle=component_model(result)
    z=cr.vision.transform(bundle["state"],X,image)
    assert np.isfinite(z).all()
    return cr.vision.model_predict(bundle["model"],z,bundle["offset"])


def predict_spec(spec,X,image):
    comp=spec["components"]
    if spec["candidate"].startswith("late_"):
        m=metadata_predict(comp["metadata"],X)
        i=image_predict(comp["image"],X,image)
        w=comp["image_weight"]
        return (1-w)*m+w*i
    key="early" if spec["candidate"].startswith("early_") else "image"
    return image_predict(comp[key],X,image)


def evaluate_all():
    import canonical_runner as cr
    specs=require_freeze()
    tables={}
    primary=[]
    rng=np.random.default_rng(20260908)
    for collection in COLLECTIONS:
        selected=specs["collections"][collection]
        path=REV/"target_pipeline_20260909"/f"{collection.lower()}_temporal_test_targets.NOT_FOR_SELECTION.jsonl"
        rows=lines(path)
        assert all("2025-01-01"<=r["time"]<"2026-04-14" for r in rows)
        tok=np.asarray([r["token_id"] for r in rows],dtype=np.int64)
        y=np.asarray([r["y_log_relative_price"] for r in rows],dtype=np.float64)
        ids=np.asarray([r["source_row"] for r in rows],dtype=np.int64)
        times=np.asarray([r["time"] for r in rows])
        assert np.isfinite(y).all()
        metadata={(r["collection"],int(r["token_id"])):r for r in lines(cr.base.OLD/"metadata_normalized.jsonl")}
        cols=cr.base.FEATURES+(["generation"] if collection=="MAYC" else [])
        X=np.asarray([[metadata[collection,int(t)][c] for c in cols] for t in tok],dtype=object)
        pm=metadata_predict(selected["metadata"],X)
        mm=cr.base.metrics(y,pm,tok)
        predictions={}
        result_rows=[]
        for encoder in ENCODERS:
            info=read(OUT/"embedding_registry.json")[encoder][collection]
            mapping={int(t):i for i,t in enumerate(info["token_ids"])}
            # Missing rows are a failure, never a filtering operation.
            indices=np.asarray([mapping[int(t)] for t in tok],dtype=np.int64)
            assert np.array_equal(np.asarray(info["token_ids"])[indices],tok)
            image=np.load(ROOT/info["matrix"],mmap_mode="r",allow_pickle=False)[indices].astype(np.float64)
            assert np.isfinite(image).all()
            for role in ["augmented","image","mandatory_early_ElasticNet"]:
                spec=selected["encoders"][encoder][role]
                pred=predict_spec(spec,X,image)
                assert np.isfinite(pred).all() and len(pred)==len(y)
                name=encoder+"__"+role
                predictions[name]=pred
                metric=cr.base.metrics(y,pred,tok)
                is_primary=encoder==selected["selected_encoder"] and role=="augmented"
                result_rows.append(dict(collection=collection,encoder=encoder,role=role,native_embedding_dimension=info["dimensions"],
                                        candidate=spec["candidate"],hyperparameters={k:v["selected_config"] for k,v in spec["components"].items() if isinstance(v,dict) and "selected_config" in v},
                                        selected_late_fusion_weight=spec["components"].get("image_weight"),
                                        development_selected_Astar=is_primary,comparison_type="development-selected A*" if is_primary else "non-selected exploratory comparison",
                                        RMSE=metric["rmse"],MAE=metric["mae"],out_of_sample_R2=metric["r2"],
                                        equal_token_RMSE=metric["equal_token_rmse"],Delta_RMSE=metric["rmse"]-mm["rmse"],
                                        Utility_percent=100*(mm["rmse"]-metric["rmse"])/mm["rmse"],metadata_RMSE=mm["rmse"],
                                        observations=len(y),unique_tokens=len(np.unique(tok))))
        cluster=paired_samples(y,pm,predictions,tok,rng)
        # Retain canonical RNG consumption: collection token bootstrap, then block bootstrap.
        days=times.astype("datetime64[D]")
        blocks=((days-days.min()).astype(int)//14)
        name=selected["selected_encoder"]+"__augmented"
        block=paired_samples(y,pm,{name:predictions[name]},blocks,rng)
        for j,(key,row) in enumerate(zip(predictions,result_rows)):
            row.update(summarize_bootstrap(cluster["delta"][:,j],cluster["utility"][:,j],row["development_selected_Astar"]))
            assert np.isclose(row["Delta_RMSE"],row["RMSE"]-row["metadata_RMSE"],atol=1e-14)
            assert np.isclose(row["Utility_percent"],100*(row["metadata_RMSE"]-row["RMSE"])/row["metadata_RMSE"],atol=1e-12)
            if row["development_selected_Astar"]:
                primary.append(row.copy())
        result_rows.insert(0,dict(collection=collection,encoder="metadata",role="M*",candidate=selected["metadata_family"],
                                  development_selected_Astar=False,comparison_type="fresh development-selected metadata baseline",
                                  RMSE=mm["rmse"],MAE=mm["mae"],out_of_sample_R2=mm["r2"],equal_token_RMSE=mm["equal_token_rmse"],
                                  Delta_RMSE=0.,Utility_percent=0.,metadata_RMSE=mm["rmse"],observations=len(y),unique_tokens=len(np.unique(tok))))
        csvsave(OUT/f"final_evaluation_all_encoders_{collection}.csv",result_rows)
        prediction_path=OUT/f"final_predictions_{collection}.npz"
        with prediction_path.open("xb") as f:
            np.savez_compressed(f,y=y,tokens=tok,source_rows=ids,times=times,metadata=pm,**predictions)
        with (OUT/f"bootstrap_draws_{collection}.npz").open("xb") as f:
            np.savez_compressed(f,delta=cluster["delta"],utility=cluster["utility"],rmse=cluster["rmse"],
                                prediction_order=np.asarray(cluster["prediction_order"]),block_delta=block["delta"],block_utility=block["utility"])
        sensitivity=OUT/"sensitivity_results"
        sensitivity.mkdir(exist_ok=True)
        csvsave(sensitivity/f"calendar_block_bootstrap_{collection}.csv",[dict(collection=collection,encoder=selected["selected_encoder"],blocks=block["cluster_count"],
                                                                                   **summarize_bootstrap(block["delta"][:,0],block["utility"][:,0],False))])
        dev_tokens=set(cr.load_development(collection)["tokens"])
        subgroups=[]
        for subgroup,known in [("known",True),("unseen",False)]:
            mask=np.isin(tok,list(dev_tokens))
            mask=mask if known else ~mask
            count=int(mask.sum());unique=len(np.unique(tok[mask]))
            record=dict(collection=collection,subgroup=subgroup,trades=count,tokens=unique,meets_minimum_support=count>=50 and unique>=30)
            if count:
                sm=cr.base.metrics(y[mask],pm[mask],tok[mask]);sa=cr.base.metrics(y[mask],predictions[name][mask],tok[mask])
                record.update(metadata_RMSE=sm["rmse"],augmented_RMSE=sa["rmse"],Delta_RMSE=sa["rmse"]-sm["rmse"],Utility_percent=100*(sm["rmse"]-sa["rmse"])/sm["rmse"])
            subgroups.append(record)
        csvsave(sensitivity/f"known_unseen_{collection}.csv",subgroups)
        tables[collection]=result_rows
    csvsave(OUT/"bootstrap_primary_results.csv",primary)
    save(OUT/"evaluation_completed.json",dict(utc=now(),frozen_specification_sha256=sha(OUT/"selected_primary_models.json"),
                                              primary_results=primary,selection_changed_after_evaluation=False,
                                              sensitivities_pending=["exact 1-wei fixed-specification and independent refit","past-only residual-prediction diagnostics","past-only stacking diagnostics"]))
    return tables


if __name__=="__main__":
    evaluate_all()
