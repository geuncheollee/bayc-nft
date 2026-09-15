"""Construct v1 log-relative-price labels, with strict timestamp batching.

Uses original CSVs only as read-only inputs and matches frozen source record IDs.
The terminal evaluation labels are materialized separately, not scored/explored.
"""
import argparse
import bisect
import csv
import hashlib
import itertools
import json
import math
import statistics
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'revision/data_audit_20260908'
COHORT = ROOT/'revision/image_validation_20260909'
DEFAULT_OUT = ROOT/'revision/target_pipeline_20260909'
DAY = 86400


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(s) for s in stream if s.strip()]


def save(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def baseline_history(rows, window_days=20):
    """(source_row,timestamp,token,positive_price) -> prior median/counts.

    All same-timestamp labels read the same PRE-batch state; the lower boundary
    t-window is inclusive. No metadata/image filtering of the market history.
    """
    order = sorted(rows, key=lambda r:(r[1],r[0]))
    queue, tokens, prices = deque(), Counter(), []
    output = {}
    for stamp, batch_iter in itertools.groupby(order, key=lambda r:r[1]):
        while queue and queue[0][1] < stamp-window_days*DAY:
            _, _, token, price = queue.popleft()
            prices.pop(bisect.bisect_left(prices, price))
            tokens[token] -= 1
            if not tokens[token]:
                del tokens[token]
        n = len(prices)
        median = None if not n else (prices[n//2] if n%2 else (prices[n//2-1]+prices[n//2])/2)
        batch = list(batch_iter)
        for rowid, _, _, _ in batch:
            output[rowid] = dict(baseline_eth=median, prior_trades=n,
                prior_tokens=len(tokens), prior_latest_epoch=queue[-1][1] if queue else None)
        for row in batch:
            assert math.isfinite(row[3]) and row[3] > 0
            queue.append(row)
            tokens[row[2]] += 1
            bisect.insort(prices, row[3])
    return output


def tests():
    rows=[(1,0,1,2.),(2,10*DAY,2,4.),(3,20*DAY,3,100.),
          (4,20*DAY,4,10000.),(5,20*DAY+1,1,8.)]
    result=baseline_history(rows)
    assert result[1]['baseline_eth'] is None
    assert result[3]['baseline_eth']==3 and result[4]['baseline_eth']==3
    assert result[3]['prior_trades']==2 and result[3]['prior_tokens']==2
    assert result[5]['prior_trades']==3 and result[5]['baseline_eth']==100
    altered=baseline_history(rows[:2]+[(r[0],r[1],r[2],r[3]*7) for r in rows[2:]])
    assert all(result[i]==altered[i] for i in (1,2,3,4))
    assert baseline_history(list(reversed(rows)))==result
    assert math.log(.5)<0 and math.isfinite(math.log(.5))
    return dict(inclusive_lower_boundary=True, simultaneous_batch_excluded=True,
                future_and_current_price_invariance=True, order_invariance=True,
                negative_log_ratio_valid=True, missing_initial_history_handled=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    out=args.output
    start=time.perf_counter()
    unit=tests()
    spec=read(OLD/'analysis_specification_v1.json')
    release=read(COHORT/'cohort_release_v1.json')
    assert sha(OLD/'analysis_specification_v1.json')==release['analysis_specification_sha256']
    assert spec['target']['baseline_window_days']==20 and spec['target']['baseline_statistic']=='median'
    assert spec['target']['minimum_prior_trades']==20 and spec['target']['minimum_prior_distinct_tokens']==10
    for r in release['artifacts']:
        assert sha(ROOT/r['path'])==r['sha256'],r['path']
    originals=read(OLD/'source_manifest.json')
    for r in originals:
        assert sha(ROOT/r['path'])==r['sha256'],r['path']
    out.mkdir(exist_ok=False)
    save(out/'implementation_decisions_v1.json',dict(
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        target_policy_sha256=sha(OLD/'analysis_specification_v1.json'),
        cohort_release_sha256=sha(COHORT/'cohort_release_v1.json'),
        implementation='sorted rolling price multiset; simultaneous timestamps queried before insert',
        target_scaling='none beyond natural log(P/B); epsilon and alpha interpreted on this target scale',
        final_evaluation='separate label file; no final descriptive price summaries or predictions in this stage',
        no_target_winsorization=True, no_new_price_filters=True,
        controls=['training mean of y','fixed zero y, equivalent to prior collection median price'],
        control_role='diagnostic baselines, not additional primary hypothesis tests',
        planned_model_stage='pipeline smoke tests plus nested development Ridge only',
        no_final_model_selected=True))
    summaries={}
    real_checks=[]
    artifacts=[]
    for collection in ('BAYC','MAYC'):
        eligible=lines(OLD/(collection.lower()+'_trade_eligibility.jsonl'))
        eligible_map={r['source_row']:r for r in eligible}
        master=lines(COHORT/(collection.lower()+'_master_rows_v1.jsonl'))
        master_map={r['source_row']:r for r in master}
        market=[]
        csvpath=ROOT/'revision'/(collection.lower()+'_all_trades.csv')
        source_snapshot=(csvpath.stat().st_size,csvpath.stat().st_mtime_ns,sha(csvpath))
        with csvpath.open(encoding='utf-8-sig',newline='') as stream:
            seen=0
            for rowid,r in enumerate(csv.DictReader(stream),2):
                old=eligible_map[rowid]
                assert r['unique_trade_id']==old['unique_trade_id']
                assert int(r['token_id'])==old['token_id']
                stamp=datetime.fromisoformat(r['block_time'].removesuffix(' UTC')+'+00:00')
                assert stamp.isoformat(sep=' ')==old['time']
                if old['base_clean']:
                    assert r['currency_contract'].lower()==spec['currency']['allowed_contracts'][r['currency_symbol']]
                    price=float(r['amount_original'])
                    assert price>0 and math.isfinite(price)
                    market.append((rowid,stamp.timestamp(),old['token_id'],price))
                seen+=1
        assert seen==len(eligible)
        history=baseline_history(market)
        for r in market:
            h=history[r[0]]; old=eligible_map[r[0]]
            assert h['prior_trades']==old['prior_20d_trades']
            assert h['prior_tokens']==old['prior_20d_tokens']
            assert h['prior_latest_epoch'] is None or h['prior_latest_epoch']<r[1]
        order=sorted(market,key=lambda r:(r[1],r[0]))
        positions=sorted({round(i*(len(order)-1)/39) for i in range(40)})
        max_delta=0.
        for pos in positions:
            row=order[pos]
            past=[r for r in market if row[1]-20*DAY<=r[1]<row[1]]
            expected=statistics.median(r[3] for r in past) if past else None
            actual=history[row[0]]
            assert actual['baseline_eth']==expected
            assert actual['prior_trades']==len(past)
            assert actual['prior_tokens']==len({r[2] for r in past})
        # Production prefix invariance: recompute on pre-test data only, not merely a toy case.
        cutoff=datetime(2025,1,1,tzinfo=timezone.utc).timestamp()
        dev_history=baseline_history([r for r in market if r[1]<cutoff])
        assert all(history[k]==v for k,v in dev_history.items())
        # Change all final-period prices: every earlier baseline must remain identical.
        changed=baseline_history([(a,b,c,d if b<cutoff else d*3.7) for a,b,c,d in market])
        assert all(history[k]==changed[k] for k in dev_history)
        real_checks.append(dict(collection=collection, brute_force_windows=len(positions),
            all_prior_counts_match_previous_audit=True, all_development_prefixes_identical=True,
            changed_final_prices_do_not_change_development=True))
        records={'development':[],'temporal_test':[]}
        for rowid,stamp,tid,price in order:
            if rowid not in master_map:
                continue
            h=history[rowid]
            assert h['prior_trades']>=20 and h['prior_tokens']>=10 and h['baseline_eth']>0
            ratio=price/h['baseline_eth']; target=math.log(ratio)
            assert math.isfinite(target) and math.isclose(math.exp(target)*h['baseline_eth'],price,rel_tol=1e-12)
            old=master_map[rowid]
            result=dict(collection=collection,source_row=rowid,unique_trade_id=old['unique_trade_id'],
                token_id=tid,time=old['time'],split=old['split'],price_eth=price,
                baseline_eth=h['baseline_eth'],relative_price=ratio,y_log_relative_price=target,
                prior_20d_trades=h['prior_trades'],prior_20d_tokens=h['prior_tokens'],
                prior_latest_time_utc=datetime.fromtimestamp(h['prior_latest_epoch'],timezone.utc).isoformat(),
                known_training_token=old['known_training_token'])
            records[old['split']].append(result)
        assert sum(len(v) for v in records.values())==len(master)
        assert {r['source_row'] for v in records.values() for r in v}==set(master_map)
        for split,rows in records.items():
            name=collection.lower()+('_development_targets.jsonl' if split=='development' else '_temporal_test_targets.NOT_FOR_SELECTION.jsonl')
            with (out/name).open('x',encoding='utf-8') as stream:
                for r in rows:
                    stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
            artifacts.append(dict(path=str((out/name).relative_to(ROOT)),sha256=sha(out/name),rows=len(rows)))
        ys=sorted(r['y_log_relative_price'] for r in records['development'])
        summaries[collection]=dict(market_history_rows=len(market),
            development_rows=len(records['development']),development_tokens=len({r['token_id'] for r in records['development']}),
            final_labels_materialized=len(records['temporal_test']),final_label_values_summarized=False,
            development_y=dict(min=ys[0],q01=ys[round(.01*(len(ys)-1))],median=statistics.median(ys),
                               q99=ys[round(.99*(len(ys)-1))],max=ys[-1],negative=sum(y<0 for y in ys)),
            source_unchanged=(csvpath.stat().st_size,csvpath.stat().st_mtime_ns,sha(csvpath))==source_snapshot)
        assert summaries[collection]['source_unchanged']
        print(json.dumps(dict(collection=collection,status='targets_validated',
                             development_rows=len(records['development']),seconds=round(time.perf_counter()-start,2))),flush=True)
    qa=dict(unit_tests=unit,real_data_checks=real_checks,collections=summaries,
            seconds=round(time.perf_counter()-start,2),final_predictions_computed=False,
            final_metrics_computed=False,original_sources_modified=False)
    save(out/'target_validation_qa.json',qa)
    artifacts += [dict(path=str((out/n).relative_to(ROOT)),sha256=sha(out/n)) for n in
                  ('implementation_decisions_v1.json','target_validation_qa.json')]
    save(out/'target_release_v1.json',dict(status='TARGETS_VALIDATED; FINAL_LABELS_NOT_SCORED',
        artifacts=artifacts,source_manifest_sha256=sha(OLD/'source_manifest.json'),
        metadata_normalized_sha256=sha(OLD/'metadata_normalized.jsonl'),
        split_manifest_sha256=sha(COHORT/'image_ready_split_manifest_v1.json'),
        script_sha256=sha(Path(__file__)),python_version=__import__('sys').version))
    print(json.dumps(qa,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
