"""Exact-one-wei sensitivity: refilter market AND targets, preserve v1 releases.

Uses the frozen timestamps/cohort; terminal labels are written, never scored.
"""
import argparse
import copy
import csv
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import build_past_only_targets as base

ROOT = base.ROOT
ORIGINAL = ROOT/'revision/target_pipeline_20260909'
AMENDMENT = ROOT/'revision/Diagnostic Amendment 20260909.json'
DEFAULT = ROOT/'revision/one_wei_sensitivity_targets_20260909'


def exact_one_wei(row, spec):
    symbol = row['currency_symbol']
    allowed = spec['currency']['allowed_contracts']
    return (symbol in allowed and row['currency_contract'].lower() == allowed[symbol]
            and Decimal(row['amount_raw']) == 1)


def write_lines(path, rows):
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT)
    out = parser.parse_args().output.resolve()
    spec = base.read(base.OLD/'analysis_specification_v1.json')
    assert spec['currency']['raw_decimals'] == 18
    for manifest in (base.read(base.COHORT/'cohort_release_v1.json')['artifacts'],
                     base.read(ORIGINAL/'target_release_v1.json')['artifacts'],
                     base.read(base.OLD/'source_manifest.json')):
        for r in manifest:
            assert base.sha(ROOT/r['path']) == r['sha256'], r['path']
    splits = base.read(base.COHORT/'image_ready_split_manifest_v1.json')
    original_splits = copy.deepcopy(splits)
    out.mkdir(exist_ok=False)
    base.save(out/'implementation_decisions_v1.json', dict(
        version='exact-one-wei-sensitivity-v1', diagnostic_amendment_sha256=base.sha(AMENDMENT),
        original_preregistration=False, primary_sample_changed=False,
        rule='allowed symbol + matching contract + 18 decimals + exact Decimal(amount_raw)==1',
        scope='exclude from market history and target trades at all timestamps',
        no_other_price_thresholds=True, original_dates_and_grid_preserved=True,
        support='recheck >=20 prior trades and >=10 prior tokens; retain original cohort intersection',
        final_labels='materialize separately; no distribution, predictions or metrics',
        original_target_release_sha256=base.sha(ORIGINAL/'target_release_v1.json')))
    unit = base.tests()
    example = dict(currency_symbol='ETH', currency_contract=spec['currency']['allowed_contracts']['ETH'], amount_raw='1')
    assert exact_one_wei(example, spec)
    assert not exact_one_wei(dict(example, amount_raw='2'), spec)
    assert not exact_one_wei(dict(example, currency_contract='0xwrong'), spec)
    assert not exact_one_wei(dict(example, currency_symbol='OTHER'), spec)
    unit['exact_raw_and_allowed_currency_filter'] = True
    summaries, exclusions, audit_cases, fold_checks = {}, [], [], []
    for collection in ('BAYC', 'MAYC'):
        c = collection.lower()
        eligibility = {r['source_row']: r for r in base.lines(base.OLD/(c+'_trade_eligibility.jsonl'))}
        master = {r['source_row']: r for r in base.lines(base.COHORT/(c+'_master_rows_v1.jsonl'))}
        dev_old = {r['source_row']: r for r in base.lines(ORIGINAL/(c+'_development_targets.jsonl'))}
        csvpath = ROOT/'revision'/(c+'_all_trades.csv')
        source_hash = base.sha(csvpath)
        market, full_market, removed, raw_dev = [], [], set(), []
        with csvpath.open(encoding='utf-8-sig', newline='') as stream:
            for rowid, r in enumerate(csv.DictReader(stream), 2):
                e = eligibility[rowid]
                assert r['unique_trade_id'] == e['unique_trade_id']
                if e['time'] < '2025':
                    raw_dev.append((rowid, r))
                if not e['base_clean']:
                    continue
                stamp = datetime.fromisoformat(e['time']).timestamp()
                row = (rowid, stamp, e['token_id'], float(r['amount_original']))
                full_market.append(row)
                if exact_one_wei(r, spec):
                    assert Decimal(r['amount_original']) == Decimal('1e-18')
                    removed.add(rowid)
                    exclusions.append(dict(collection=collection, source_row=rowid, split=e['split'],
                        unique_trade_id=e['unique_trade_id'], in_original_master=rowid in master,
                        reason='exact_one_wei_allowed_18_decimal_currency'))
                else:
                    market.append(row)
        case_hashes = {r['tx_hash'] for i, r in raw_dev if i in removed and i in dev_old}
        tx_rows = {tx: [(i, r) for i, r in raw_dev if r['tx_hash'] == tx] for tx in case_hashes}
        for rowid, r in raw_dev:
            if rowid not in removed or rowid not in dev_old:
                continue
            e = eligibility[rowid]
            related = tx_rows[r['tx_hash']]
            audit_cases.append(dict(collection=collection, source_row=rowid, source=r,
                exact_raw_conversion=True, seller_differs_from_buyer=r['seller'].lower()!=r['buyer'].lower(),
                same_collection_tx_rows=len(related), same_collection_tx_tokens=len({v['token_id'] for _, v in related}),
                same_collection_other_rows=[dict(source_row=i,token_id=v['token_id'],amount_raw=v['amount_raw'],
                    currency_symbol=v['currency_symbol'],seller=v['seller'],buyer=v['buyer']) for i,v in related if i!=rowid],
                strict_clean=e['strict_clean'], observed_multi_nft_tx=e['observed_multi_nft_tx'],
                reverse_pair_24h=e['reverse_pair_24h'],
                interpretation='raw CSV internally consistent; economic meaning requires event/payment evidence'))
        history = base.baseline_history(market)
        original_history = base.baseline_history(full_market)
        # Exact full-production prior-history check against independent brute force.
        order = sorted(market, key=lambda r:(r[1],r[0]))
        positions = sorted({round(i*(len(order)-1)/39) for i in range(40)})
        for pos in positions:
            row = order[pos]
            past = [r for r in market if row[1]-20*base.DAY <= r[1] < row[1]]
            h = history[row[0]]
            assert h['baseline_eth'] == (statistics.median(r[3] for r in past) if past else None)
            assert h['prior_trades'] == len(past)
            assert h['prior_tokens'] == len({r[2] for r in past})
        cutoff = datetime(2025,1,1,tzinfo=timezone.utc).timestamp()
        prefix = base.baseline_history([r for r in market if r[1] < cutoff])
        altered = base.baseline_history([(a,b,c,d if b<cutoff else d*3.7) for a,b,c,d in market])
        assert all(history[k] == v == altered[k] for k,v in prefix.items())
        records = {'development': [], 'temporal_test': []}
        support_lost = []
        for rowid, stamp, token, price in order:
            if rowid not in master:
                continue
            h, old = history[rowid], master[rowid]
            assert h['prior_latest_epoch'] is None or h['prior_latest_epoch'] < stamp
            if h['prior_trades'] < 20 or h['prior_tokens'] < 10:
                support_lost.append(rowid)
                exclusions.append(dict(collection=collection,source_row=rowid,split=old['split'],
                    unique_trade_id=old['unique_trade_id'],in_original_master=True,reason='insufficient_recomputed_prior_support'))
                continue
            y = math.log(price/h['baseline_eth'])
            assert math.isfinite(y) and math.isclose(math.exp(y)*h['baseline_eth'],price,rel_tol=1e-12)
            records[old['split']].append(dict(collection=collection,source_row=rowid,unique_trade_id=old['unique_trade_id'],
                token_id=token,time=old['time'],split=old['split'],price_eth=price,
                baseline_eth=h['baseline_eth'],relative_price=price/h['baseline_eth'],y_log_relative_price=y,
                prior_20d_trades=h['prior_trades'],prior_20d_tokens=h['prior_tokens'],
                prior_latest_time_utc=datetime.fromtimestamp(h['prior_latest_epoch'],timezone.utc).isoformat()))
        dev_tokens = {r['token_id'] for r in records['development']}
        retained = {r['source_row'] for rows in records.values() for r in rows}
        assert retained == set(master)-removed-set(support_lost)
        for split, rows in records.items():
            for r in rows:
                r['known_training_token'] = r['token_id'] in dev_tokens
            suffix = '_development_targets.jsonl' if split=='development' else '_temporal_test_targets.NOT_FOR_SELECTION.jsonl'
            write_lines(out/(c+suffix), rows)
            key = 'development_source_rows' if split=='development' else 'temporal_test_source_rows'
            splits[collection][key] = [i for i in original_splits[collection][key] if i in retained]
            assert set(splits[collection][key]) == {r['source_row'] for r in rows}
        rowmap = {r['source_row']:r for r in records['development']}
        for fold, old_fold in zip(splits[collection]['folds'], original_splits[collection]['folds']):
            for key in ('train_source_rows','validation_source_rows'):
                fold[key] = [i for i in old_fold[key] if i in retained]
            train = [rowmap[i] for i in fold['train_source_rows']]
            valid = [rowmap[i] for i in fold['validation_source_rows']]
            def check(tr, va, stage):
                assert max(r['time'] for r in tr) < min(r['time'] for r in va)
                assert not {r['source_row'] for r in tr} & {r['source_row'] for r in va}
                sizes = dict(train_trades=len(tr),train_tokens=len({r['token_id'] for r in tr}),
                             validation_trades=len(va),validation_tokens=len({r['token_id'] for r in va}))
                assert all(sizes[k] >= v for k,v in spec['validation']['minimum_fold_support'].items())
                fold_checks.append(dict(collection=collection,fold=fold['name'],stage=stage,**sizes))
                return sizes
            check(train, valid, 'outer')
            for q in fold['inner_quarter_schedule']:
                begin,end = q['train_end_exclusive'],q['validation_end_exclusive']
                tr = [r for r in train if r['time'] < begin]
                va = [r for r in train if begin <= r['time'] < end]
                q.update(check(tr,va,begin))
        changes = [abs(r['y_log_relative_price']-dev_old[r['source_row']]['y_log_relative_price']) for r in records['development']]
        summaries[collection] = dict(original_market_rows=len(full_market),sensitivity_market_rows=len(market),
            market_excluded_by_split=dict(Counter(eligibility[i]['split'] for i in removed)),
            original_development_rows=len(dev_old),development_rows=len(records['development']),
            development_tokens=len(dev_tokens),final_labels_materialized=len(records['temporal_test']),
            target_excluded_by_split=dict(Counter(master[i]['split'] for i in removed if i in master)),
            additional_support_exclusions=len(support_lost),
            retained_development_baselines_changed=sum(d>0 for d in changes),max_development_target_change=max(changes),
            production_brute_force_checks=len(positions),development_prefix_and_future_mutation_passed=True,
            final_label_values_summarized=False,source_unchanged=base.sha(csvpath)==source_hash)
        assert summaries[collection]['source_unchanged']
        print(json.dumps(dict(collection=collection,**summaries[collection])),flush=True)
    assert len(audit_cases) == 21
    base.save(out/'image_ready_split_manifest_v1.json',splits)
    write_lines(out/'excluded_rows.jsonl',exclusions)
    write_lines(out/'development_one_wei_source_audit.jsonl',audit_cases)
    base.save(out/'target_validation_qa.json',dict(collections=summaries,unit_tests=unit,fold_checks=fold_checks,
        final_predictions_computed=False,final_metrics_computed=False,original_sources_modified=False))
    artifacts = [dict(path=str(p.relative_to(ROOT)),sha256=base.sha(p)) for p in sorted(out.iterdir())]
    base.save(out/'target_release_v1.json',dict(status='ONE_WEI_SENSITIVITY_TARGETS_VALIDATED; FINAL_NOT_SCORED',
        artifacts=artifacts,metadata_normalized_sha256=base.sha(base.OLD/'metadata_normalized.jsonl'),
        split_manifest_sha256=base.sha(out/'image_ready_split_manifest_v1.json'),
        script_sha256=base.sha(Path(__file__)),baseline_builder_sha256=base.sha(Path(base.__file__)),
        diagnostic_amendment_sha256=base.sha(AMENDMENT)))


if __name__ == '__main__':
    main()
