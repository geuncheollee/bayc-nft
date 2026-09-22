"""Freeze image-ready master rows without modifying v1 or opening price outcomes.

This is the data master, not a claim that every future encoder extraction works.
"""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'revision/data_audit_20260908'
OUT = ROOT/'revision/image_validation_20260909'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(s) for s in stream if s.strip()]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(name, value):
    with (OUT/name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def counts(rows):
    return dict(trades=len(rows), tokens=len({r['token_id'] for r in rows}))


def check_window(rows, start, end):
    train = [r for r in rows if r['time'] < start]
    valid = [r for r in rows if start <= r['time'] < end]
    for part in (train, valid):
        assert len(part) >= 200 and len({r['token_id'] for r in part}) >= 100
    assert max(r['time'] for r in train) < min(r['time'] for r in valid)
    assert not ({r['source_row'] for r in train} & {r['source_row'] for r in valid})
    return dict(train_end_exclusive=start, validation_end_exclusive=end,
                train=counts(train), validation=counts(valid))


def main():
    summary = read(OUT/'image_validation_summary.json')
    assert summary['inputs_unchanged_during_run'] and summary['images_unchanged_during_run']
    assert not any(summary['duplicates'].values()), 'Duplicate images require review'
    assert all(c['passed']==c['checked'] for c in summary['collections'].values())
    assert all(not c['missing_candidate_ids'] for c in summary['inventory'].values())
    spec = read(OLD/'analysis_specification_v1.json')
    old_qa = read(OLD/'protocol_qa.json')
    assert sha(OLD/'analysis_specification_v1.json') == old_qa['spec_sha256']
    for r in read(OUT/'input_snapshot.json'):
        assert sha(ROOT/r['path']) == r['sha256'], 'Prior audit input changed'
    for r in read(OUT/'raw_source_recheck.json'):
        assert sha(ROOT/r['path']) == r['sha256'], 'Source changed since image audit'
    images = lines(OUT/'image_integrity_manifest.jsonl')
    image_map = {(r['collection'],r['token_id']):r for r in images}
    for r in images:
        stat = (ROOT/r['path']).stat()
        assert (stat.st_size,stat.st_mtime_ns)==(r['size_bytes'],r['mtime_ns'])
    metadata = lines(OLD/'metadata_normalized.jsonl')
    meta_map = {(r['collection'],r['token_id']):r for r in metadata}
    remote = {}
    for name in ('source_link_reverification.json','source_link_reverification_retry.json'):
        if (OUT/name).exists():
            for r in read(OUT/name)['records']:
                if r['verified']:
                    remote[r['token_id']] = dict(evidence_file=name, **r)
    missing_cid = summary['collections']['MAYC']['source_reference_missing_ids']
    assert set(missing_cid) <= set(remote), 'Unresolved source link; do not freeze yet'
    for tid in missing_cid:
        assert remote[tid]['local_sha256'] == image_map['MAYC',tid]['sha256']
        assert remote[tid]['expected_uri'] == meta_map['MAYC',tid]['image_uri']
    original_download = lines(ROOT/'MAYC_Images/download_manifest.jsonl')
    original_map = {r['token_id']:r for r in original_download}
    provenance_rows = []
    for tid, original in sorted(original_map.items()):
        uri = meta_map['MAYC',tid]['image_uri']
        entry = dict(token_id=tid, expected_uri=uri,
                     local_sha256=image_map['MAYC',tid]['sha256'],
                     original_manifest_cid=original.get('cid'),
                     original_manifest_gateway=original.get('gateway'),
                     original_timestamp=original.get('timestamp'),
                     historical_download_log_preserved=True,
                     independent_cid_dag_verification=False)
        timestamp = original.get('timestamp','')
        normalized = timestamp[:-1] if timestamp.endswith('+00:00Z') else timestamp
        try:
            entry['timestamp_utc_normalized'] = datetime.fromisoformat(normalized.replace('Z','+00:00')).isoformat()
        except ValueError:
            entry['timestamp_utc_normalized'] = None
        entry['duplicate_timezone_suffix_corrected_in_derived_record'] = timestamp.endswith('+00:00Z')
        if tid in remote:
            entry.update(status='fresh_gateway_bytes_match_local',
                         reverification_evidence=remote[tid]['evidence_file'],
                         original_download_url_recovered=False)
        else:
            assert 'ipfs://'+original['cid'] == uri
            entry.update(status='original_manifest_cid_and_local_sha_match',
                         reconstructed_recorded_request_url=original['gateway']+original['cid'],
                         fresh_gateway_request_performed=False)
        provenance_rows.append(entry)
    with (OUT/'mayc_provenance_supplement.jsonl').open('x',encoding='utf-8') as stream:
        for r in provenance_rows:
            stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    candidates = read(OLD/'candidate_token_ids_v1.json')
    splits = read(OLD/'split_manifest_v1.json')
    trade_summary = read(OLD/'trade_audit.json')
    new_splits = {}
    stats = {}
    token_rows = []
    fold_checks = []
    for collection in ('BAYC','MAYC'):
        eligible = lines(OLD/(collection.lower()+'_trade_eligibility.jsonl'))
        rows = [r for r in eligible if r['candidate_primary']]
        ids = {r['token_id'] for r in rows}
        assert ids == set(candidates[collection])
        assert len(rows) == trade_summary[collection]['candidate_primary']['trades']
        assert len({r['source_row'] for r in rows}) == len(rows)
        assert len({r['unique_trade_id'] for r in rows}) == len(rows)
        assert all(r['prior_20d_trades']>=20 and r['prior_20d_tokens']>=10 for r in rows)
        for tid in sorted(ids):
            key = collection,tid
            assert image_map[key]['local_integrity_ok'] and meta_map[key]['valid_for_primary']
            token_rows.append(dict(collection=collection, token_id=tid,
                contract=meta_map[key]['contract'], generation=meta_map[key].get('generation'),
                image_path=image_map[key]['path'], image_sha256=image_map[key]['sha256'],
                metadata_source=meta_map[key]['metadata_source'],
                image_data_ready=True, encoder_features_verified=False))
        dev = [r for r in rows if r['time'] < '2025-01-01 00:00:00+00:00']
        test = [r for r in rows if '2025-01-01 00:00:00+00:00' <= r['time'] < '2026-04-14 00:00:00+00:00']
        assert len(dev)+len(test)==len(rows)
        dev_ids = {r['token_id'] for r in dev}
        assert all(r['split']=='development' for r in dev)
        assert all(r['split']=='temporal_test' for r in test)
        assert all(r['known_training_token']==(r['token_id'] in dev_ids) for r in test)
        entry = dict(development_source_rows=[r['source_row'] for r in dev],
                     temporal_test_source_rows=[r['source_row'] for r in test], folds=[])
        for old_fold, definition in zip(splits[collection]['folds'],trade_summary[collection]['folds']):
            start,end=definition['train_end_exclusive'],definition['validation_end_exclusive']
            outer=check_window(rows,start,end)
            inner=[]
            for quarter in old_fold['inner_quarter_schedule']:
                icheck=check_window([r for r in rows if r['time']<start],
                    quarter['train_end_exclusive'],quarter['validation_end_exclusive'])
                assert icheck['train']['trades']==quarter['train_trades']
                assert icheck['validation']['trades']==quarter['validation_trades']
                inner.append(icheck)
            entry['folds'].append(dict(name=old_fold['name'],
                train_source_rows=[r['source_row'] for r in rows if r['time']<start],
                validation_source_rows=[r['source_row'] for r in rows if start<=r['time']<end],
                inner_quarter_schedule=old_fold['inner_quarter_schedule']))
            fold_checks.append(dict(collection=collection,fold=old_fold['name'],outer=outer,inner=inner))
        final_tuning=[]
        for start,end in spec['validation']['final_hyperparameter_tuning_quarters']:
            final_tuning.append(check_window(dev,start+' 00:00:00+00:00',end+' 00:00:00+00:00'))
        assert entry==splits[collection], 'Membership changed from frozen candidates'
        new_splits[collection]=entry
        stats[collection]=dict(primary=counts(rows), development=counts(dev), temporal_test=counts(test),
            test_known=counts([r for r in test if r['token_id'] in dev_ids]),
            test_unseen=counts([r for r in test if r['token_id'] not in dev_ids]),
            image_excluded_trades=0,image_excluded_tokens=0,
            final_tuning_quarters=final_tuning)
        with (OUT/(collection.lower()+'_master_rows_v1.jsonl')).open('x',encoding='utf-8') as stream:
            for r in rows:
                # Preserve original eligibility audit separately; expose current readiness explicitly.
                record={k:r[k] for k in ('collection','source_row','unique_trade_id','token_id','time','split',
                    'known_training_token','observed_pretest_trade','strict_clean','observed_multi_nft_tx','reverse_pair_24h')}
                record.update(image_data_ready=True,encoder_features_verified=False)
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    with (OUT/'master_tokens_v1.jsonl').open('x',encoding='utf-8') as stream:
        for r in token_rows:
            stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    save('image_ready_split_manifest_v1.json',new_splits)
    save('image_ready_token_ids_v1.json',candidates)
    save('cohort_validation_qa.json',dict(collections=stats,fold_checks=fold_checks,
        same_candidate_ids=True,same_candidate_row_membership=True,all_fold_support_checks_passed=True,
        temporal_order_checked=True,source_hashes_rechecked=True,price_targets_generated=False,models_fit=False))
    stale=read(ROOT/'MAYC_Images/failed_tokens.json')
    progress=read(ROOT/'MAYC_Images/download_progress.json')
    bookkeeping=dict(original_failed_tokens=stale,
        original_failed_list_ids_now_locally_verified=[tid for tid in stale if image_map['MAYC',tid]['local_integrity_ok']],
        progress_reported_bytes=progress.get('total_bytes'),
        actual_verified_bytes=summary['collections']['MAYC']['total_bytes'],
        duplicate_timezone_suffix_rows=sum(r['duplicate_timezone_suffix_corrected_in_derived_record'] for r in provenance_rows),
        original_logs_edited=False)
    save('download_bookkeeping_reconciliation.json',bookkeeping)
    artifacts=[]
    for path in sorted(OUT.iterdir()):
        if path.suffix in ('.json','.jsonl'):
            artifacts.append(dict(path=str(path.relative_to(ROOT)),size=path.stat().st_size,sha256=sha(path)))
    release=dict(version='image-ready-master-v1',release_date_kst='2026-09-09',
        released_at_utc=datetime.now(timezone.utc).isoformat(),
        status='IMAGE_READY_DATA_MASTER_FROZEN; FINAL_ENCODER_COMMON_ROWS_PENDING_EXTRACTION_QA',
        analysis_specification_path=str((OLD/'analysis_specification_v1.json').relative_to(ROOT)),
        analysis_specification_sha256=sha(OLD/'analysis_specification_v1.json'),
        policy_changes=False,same_candidate_membership=True,collections=stats,
        provenance_missing_cid_resolved_by_new_requests=sorted(remote),
        original_four_download_urls_recovered=False,upstream_onchain_authentication=False,
        remaining_gates=['freeze encoder registry and preprocessing implementation',
                         'check extraction success for every primary encoder on common master',
                         'implement and test past-only price targets',
                         'document original trade query/export and upstream provenance'],
        models_fit=False,targets_generated=False,
        script_sha256=sha(Path(__file__)),artifacts=artifacts)
    save('cohort_release_v1.json',release)
    print(json.dumps(dict(status=release['status'],collections=stats,bookkeeping=bookkeeping),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
