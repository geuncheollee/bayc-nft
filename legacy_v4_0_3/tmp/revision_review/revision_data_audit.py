"""Local, provenance-preserving data audit. No model fit or network requests."""
from collections import Counter, defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'revision'/'data_audit_20260908'
TRAITS = ['background','fur','eyes','clothes','hat','mouth','earring']
CONTRACTS = {'BAYC':'0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d',
             'MAYC':'0x60e4d786628fea6478f785a6d7e704777c86a7c6'}
CURRENCIES = {'ETH':'0x0000000000000000000000000000000000000000',
              'WETH':'0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',
              'bpETH':'0x0000000000a39bb272e79075ade125fd351887ac'}
SPLIT = pd.Timestamp('2025-01-01',tz='UTC')
END = pd.Timestamp('2026-04-14',tz='UTC')
OUTERS = [('D1','2023-01-01','2023-07-01'),('D2','2023-07-01','2024-01-01'),
          ('D3','2024-01-01','2025-01-01')]


def emit(stage, **values):
    print(json.dumps(dict(stage=stage,**values), ensure_ascii=False, default=serialize),flush=True)


def serialize(value):
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,(Path,pd.Timestamp)): return str(value)
    if isinstance(value,set): return sorted(value)
    raise TypeError(type(value).__name__)


def save(name, values):
    (OUT/name).write_text(json.dumps(values,ensure_ascii=False,indent=2,default=serialize,allow_nan=False),encoding='utf-8')


def jsonl(name, rows):
    with (OUT/name).open('w',encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,default=serialize,allow_nan=False)+'\n')


def fingerprint(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): digest.update(block)
    return dict(path=str(path.relative_to(ROOT)),size=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns,sha256=digest.hexdigest())


def normalize(value, column):
    value=str(value).strip()
    return column+'_None' if value in ('','None',column+'_None') else value


def metadata_audit():
    summary={}
    records={}
    fields=['token_id','trait_count']+TRAITS
    bayc=pd.read_csv(ROOT/'tfidf20260103.csv',usecols=fields,dtype=str,keep_default_na=False)
    if bayc.token_id.duplicated().any(): raise ValueError('BAYC duplicate metadata IDs')
    bayc_rows={}
    for row in bayc.to_dict('records'):
        token=int(row['token_id'])
        row.update(collection='BAYC',contract=CONTRACTS['BAYC'],token_id=token,
                   trait_count=int(row['trait_count']),is_mega=False,generation='BAYC',
                   metadata_source='tfidf20260103.csv: raw categorical fields',image_uri=None,
                   official_provenance_verified=False)
        for c in TRAITS: row[c]=normalize(row[c],c)
        row['valid_for_primary']=all(row[c]!=c+'_None' for c in ['background','fur','eyes','mouth'])
        bayc_rows[token]=row
    historical=pd.read_csv(ROOT/'bayc_y_tfidf_20260103.csv',usecols=fields,dtype=str,keep_default_na=False)
    mismatches=[]
    for row in historical.to_dict('records'):
        token=int(row['token_id'])
        if token not in bayc_rows:
            mismatches.append([token,'missing_current_raw_traits'])
            continue
        for c in TRAITS:
            if normalize(row[c],c)!=bayc_rows[token][c]: mismatches.append([token,c])
        if int(row['trait_count'])!=bayc_rows[token]['trait_count']: mismatches.append([token,'trait_count'])
    cache=json.loads((ROOT/'bayc_metadata_cache.json').read_text(encoding='utf-8-sig'))
    cache_diff=[]
    for token,entry in cache.items():
        traits={a['trait_type'].lower():str(a['value']) for a in entry.get('traits',[])}
        if int(token) in bayc_rows:
            for c in TRAITS:
                if normalize(traits.get(c,''),c)!=bayc_rows[int(token)][c]: cache_diff.append([int(token),c])
    summary['BAYC']=dict(rows=len(bayc),unique_tokens=len(bayc_rows),
        historical_rows=len(historical),historical_duplicate_ids=int(historical.token_id.duplicated().sum()),
        historical_trait_mismatch_cells=len(mismatches),historical_mismatches=mismatches,
        trait_count_inconsistent_ids=[t for t,r in bayc_rows.items() if sum(r[c]!=c+'_None' for c in TRAITS)!=r['trait_count']],
        blanks_raw={c:int((bayc[c]=='').sum()) for c in TRAITS},
        none_normalized={c:sum(r[c]==c+'_None' for r in bayc_rows.values()) for c in TRAITS},
        categories={c:sorted({r[c] for r in bayc_rows.values()}) for c in TRAITS},
        cache_rows=len(cache),cache_trait_mismatch_cells=len(cache_diff),cache_mismatches=cache_diff,
        cache_image_base_path_count=sum('/base/' in v.get('image','') for v in cache.values()),
        cache_used=False,provenance='local raw categories; upstream acquisition not yet documented')
    records['BAYC']=bayc_rows
    mayc=pd.read_csv(ROOT/'revision'/'MAYC_속성.csv',dtype=str,keep_default_na=False)
    if mayc.token_id.duplicated().any(): raise ValueError('MAYC duplicate metadata IDs')
    csv_rows={int(r['token_id']):r for r in mayc.to_dict('records')}
    mayc_rows={}
    json_manifest=[]
    errors=[]
    mismatches=[]
    for path in sorted((ROOT/'revision'/'MAYC').glob('*.json')):
        try:
            token=int(re.search(r'#(\d+)',path.stem).group(1))
            raw=path.read_bytes()
            obj=json.loads(raw.decode('utf-8-sig'))
            meta=obj.get('metadata',obj)
            attrs=meta['attributes']
            attr={a['trait_type'].lower():str(a['value']) for a in attrs}
            if token in mayc_rows or len(attr)!=len(attrs): raise ValueError('Duplicate token or trait')
            mega=bool(attr.get('name'))
            generations={v[:2] for v in attr.values() if re.match(r'^M[123](?: |$)',v)}
            generation='Mega' if mega else (next(iter(generations)) if len(generations)==1 else 'unknown')
            row=dict(collection='MAYC',contract=CONTRACTS['MAYC'],token_id=token,
                     trait_count=len(attrs),generation=generation,is_mega=mega,mega_name=attr.get('name'),
                     metadata_source=str(path.relative_to(ROOT)),image_uri=meta.get('image'),
                     official_provenance_verified=False)
            for c in TRAITS: row[c]=normalize(attr.get(c,''),c)
            row['valid_for_primary']=(not mega and generation in ('M1','M2')
                                      and all(row[c]!=c+'_None' for c in ['background','fur','eyes','mouth']))
            if token not in csv_rows: mismatches.append([token,'missing_csv'])
            else:
                for c in TRAITS:
                    if row[c]!=normalize(csv_rows[token][c],c): mismatches.append([token,c])
                if len(attrs)!=int(csv_rows[token]['trait_count']): mismatches.append([token,'trait_count'])
            uri=obj.get('tokenUri','')
            if uri and uri.rstrip('/').split('/')[-1]!=str(token): errors.append([path.name,'tokenUri_ID_mismatch'])
            mayc_rows[token]=row
            json_manifest.append(dict(path=str(path.relative_to(ROOT)),token_id=token,size=len(raw),
                                      mtime_ns=path.stat().st_mtime_ns,sha256=hashlib.sha256(raw).hexdigest()))
        except Exception as exc: errors.append([path.name,str(exc)])
    image_counts=Counter(r['image_uri'] for r in mayc_rows.values())
    summary['MAYC']=dict(csv_rows=len(mayc),json_tokens=len(mayc_rows),csv_only_ids=sorted(set(csv_rows)-set(mayc_rows)),
        json_only_ids=sorted(set(mayc_rows)-set(csv_rows)),csv_json_mismatch_cells=len(mismatches),mismatches=mismatches,
        errors=errors,generations=Counter(r['generation'] for r in mayc_rows.values()),
        mega_ids=[t for t,r in mayc_rows.items() if r['is_mega']],
        blank_image_uris=sum(not r['image_uri'] for r in mayc_rows.values()),
        duplicated_image_uri_groups=sum(n>1 for n in image_counts.values()),
        categories={c:sorted({r[c] for r in mayc_rows.values()}) for c in TRAITS})
    records['MAYC']=mayc_rows
    jsonl('metadata_normalized.jsonl',(r for group in records.values() for r in group.values()))
    jsonl('mayc_json_manifest.jsonl',json_manifest)
    save('metadata_audit.json',summary)
    emit('metadata_complete',bayc_tokens=len(bayc_rows),mayc_tokens=len(mayc_rows),
         mayc_errors=len(errors),mayc_mismatches=len(mismatches))
    return records,summary,json_manifest


def image_audit():
    items=[]
    start=time.perf_counter()
    for i,path in enumerate(sorted((ROOT/'Images').glob('*.png')),1):
        row=dict(collection='BAYC',path=str(path.relative_to(ROOT)),size=path.stat().st_size,
                 mtime_ns=path.stat().st_mtime_ns)
        try:
            row['token_id']=int(path.stem)
            raw=path.read_bytes()
            row['sha256']=hashlib.sha256(raw).hexdigest()
            with Image.open(path) as im:
                im.load()
                rgba=im.convert('RGBA')
                alpha=rgba.getchannel('A').histogram()
                rgb=im.convert('RGB')
                row.update(mode=im.mode,width=im.width,height=im.height,
                           rgba_sha256=hashlib.sha256(rgba.tobytes()).hexdigest(),
                           nonopaque_pixels=sum(alpha[:255]),zero_alpha_pixels=alpha[0],
                           total_pixels=im.width*im.height,
                           flat_rgb=all(a==b for a,b in rgb.getextrema()),decode_ok=True)
        except Exception as exc: row.update(decode_ok=False,error=str(exc))
        items.append(row)
        if i%1000==0: emit('bayc_image_progress',images=i,seconds=time.perf_counter()-start)
    good=[r for r in items if r['decode_ok']]
    raw_hash=Counter(r['sha256'] for r in good)
    pixel_hash=Counter(r['rgba_sha256'] for r in good)
    nobg={}
    for folder in ('Images_NoBG','Images_NoBG_Clean'):
        paths=list((ROOT/folder).glob('*.png'))
        sample=sorted(paths)[::max(1,len(paths)//64)][:64]
        errors=[]
        alpha_counts=0
        for p in sample:
            try:
                with Image.open(p) as im:
                    im.load()
                    alpha_counts+=im.convert('RGBA').getchannel('A').getextrema()[0]<255
            except Exception as exc: errors.append([p.name,str(exc)])
        nobg[folder]=dict(files=len(paths),numeric_ids=len({int(p.stem) for p in paths if p.stem.isdigit()}),
                         decoded_sample=len(sample),sample_errors=errors,sample_nonopaque=int(alpha_counts),
                         full_decode_audit=False,mask_provenance_verified=False)
    mayc_files=[]
    scan_dirs=[]
    for current,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in {'.git','.codex','.agents','.claude','.obsidian',
                                           'tmp','node_modules','__pycache__','.venv'}]
        relative=str(Path(current).relative_to(ROOT))
        if 'mayc' in relative.lower() or 'mutant' in relative.lower():
            scan_dirs.append(relative)
            mayc_files.extend(str((Path(current)/f).relative_to(ROOT)) for f in files
                              if Path(f).suffix.lower() in {'.png','.jpg','.jpeg','.webp','.gif'})
    summary=dict(BAYC=dict(files=len(items),decoded=len(good),errors=[r for r in items if not r['decode_ok']],
        modes=Counter(r['mode'] for r in good),sizes=Counter(f"{r['width']}x{r['height']}" for r in good),
        nonopaque_images=sum(r['nonopaque_pixels']>0 for r in good),flat_rgb_images=sum(r['flat_rgb'] for r in good),
        duplicate_file_hash_groups=sum(v>1 for v in raw_hash.values()),
        duplicate_pixel_hash_groups=sum(v>1 for v in pixel_hash.values()),
        source_identity_verified=False),NoBG=nobg,
        MAYC=dict(scanned_directories=scan_dirs,found_files=mayc_files,found_count=len(mayc_files),
                  verified_token_mapping_count=0,search_limit='named MAYC/mutant paths outside tmp and tooling directories'),
        seconds=time.perf_counter()-start)
    jsonl('bayc_image_manifest.jsonl',items)
    save('image_audit.json',summary)
    emit('image_complete',bayc_decoded=len(good),mayc_found=len(mayc_files),seconds=summary['seconds'])
    return {r['token_id']:r for r in good if not r['flat_rgb']},summary,items


def past_counts(frame,days=20):
    """Counts only: [t-days,t), excluding every event at the current timestamp."""
    output=pd.DataFrame(index=frame.index,columns=['past_trade_count','past_token_count'],dtype='int64')
    window=deque()
    tokens=Counter()
    span=pd.Timedelta(days=days).value
    for stamp,group in frame.sort_values('time').groupby('time',sort=True):
        now=stamp.value
        while window and window[0][0]<now-span:
            _,token=window.popleft()
            tokens[token]-=1
            if tokens[token]==0: del tokens[token]
        output.loc[group.index,'past_trade_count']=len(window)
        output.loc[group.index,'past_token_count']=len(tokens)
        for token in group['token'].tolist():
            window.append((now,token)); tokens[token]+=1
    return output


def count_summary(frame,mask):
    sub=frame.loc[mask]
    return dict(trades=len(sub),tokens=int(sub.token.nunique()))


def trade_audit(frames,metadata,images):
    pooled=pd.concat([f[['tx_hash','nft_contract_address','token_id']].assign(collection=c)
                      for c,f in frames.items()],ignore_index=True)
    pooled['nft_key']=pooled.nft_contract_address.str.lower()+':'+pooled.token_id
    multikey=pooled.groupby('tx_hash').nft_key.nunique()
    outputs={}
    all_token_rows=[]
    for collection,df in frames.items():
        df=df.copy()
        df['source_row']=np.arange(len(df))+2
        df['token']=pd.to_numeric(df.token_id,errors='coerce')
        df['time']=pd.to_datetime(df.block_time.str.replace(' UTC','',regex=False),
                                  format='%Y-%m-%d %H:%M:%S.%f',errors='coerce',utc=True)
        df['price']=pd.to_numeric(df.amount_original,errors='coerce')
        original_columns=list(frames[collection].columns)
        flags={}
        flags['exact_duplicate_extra']=df.duplicated(original_columns)
        duplicate_id=df.unique_trade_id.duplicated(keep=False)
        flags['missing_or_conflicting_trade_id']=(df.unique_trade_id=='') | (duplicate_id & ~flags['exact_duplicate_extra'])
        flags['wrong_chain_contract_standard']=(df.blockchain.str.lower()!='ethereum') | (
            df.nft_contract_address.str.lower()!=CONTRACTS[collection]) | (df.token_standard.str.lower()!='erc721')
        flags['invalid_token_id']=df.token.isna() | (df.token<0) | (df.token%1!=0)
        flags['invalid_time_or_date']=df.time.isna() | (df.time.dt.strftime('%Y-%m-%d')!=df.block_date)
        flags['outside_complete_day_cutoff']=df.time>=END
        flags['not_sale_event']=df.evt_type.str.lower()!='trade'
        flags['explicit_bundle']=df.trade_type.str.lower()=='bundle trade'
        flags['item_count_unknown_or_not_one']=pd.to_numeric(df.number_of_items,errors='coerce')!=1
        address_pattern=r'^0x[0-9a-f]{40}$'
        buyers,sellers=df.buyer.str.lower(),df.seller.str.lower()
        zero={'0x'+'0'*40,'0x'+'0'*36+'dead'}
        flags['invalid_or_burn_parties']=~buyers.str.match(address_pattern) | ~sellers.str.match(address_pattern) | buyers.isin(zero) | sellers.isin(zero)
        flags['self_trade']=buyers==sellers
        flags['price_missing_nonfinite_or_nonpositive']=~np.isfinite(df.price) | (df.price<=0)
        expected=df.currency_symbol.map(CURRENCIES)
        flags['unsupported_or_mismatched_currency']=expected.isna() | (df.currency_contract.str.lower()!=expected)
        raw=pd.to_numeric(df.amount_raw,errors='coerce')
        flags['raw_unit_mismatch']=(~flags['unsupported_or_mismatched_currency']) & df.price.notna() & (
            ~np.isfinite(raw) | ~np.isclose(raw/1e18,df.price,rtol=1e-9,atol=1e-10,equal_nan=False))
        flags['ambiguous_same_nft_same_tx']=df.duplicated(['tx_hash','nft_contract_address','token_id'],keep=False)
        alive=pd.Series(True,index=df.index)
        first_reason=pd.Series('',index=df.index,dtype='object')
        flow=[]
        for reason,mask in flags.items():
            excluded=alive & mask
            first_reason.loc[excluded]=reason
            alive &= ~mask
            flow.append(dict(rule=reason,flagged_rows=int(mask.sum()),excluded_at_step=int(excluded.sum()),
                             remaining_rows=int(alive.sum()),remaining_tokens=int(df.loc[alive,'token'].nunique())))
        df['base_clean']=alive
        df['first_exclusion']=first_reason
        df['multiple_nfts_in_observed_tx']=df.tx_hash.map(multikey)>1
        df['reverse_pair_24h']=False
        clean=df.loc[alive].sort_values(['time','block_number','unique_trade_id'])
        prev_buyer=clean.groupby('token').buyer.shift()
        prev_seller=clean.groupby('token').seller.shift()
        prev_time=clean.groupby('token').time.shift()
        reverse=(clean.buyer.str.lower()==prev_seller.str.lower()) & (clean.seller.str.lower()==prev_buyer.str.lower()) & (
            clean.time-prev_time<=pd.Timedelta(hours=24)) & (clean.time>prev_time)
        df.loc[clean.index,'reverse_pair_24h']=reverse
        df['strict_clean']=alive & ~df.multiple_nfts_in_observed_tx & ~df.reverse_pair_24h
        # Audit support counts, not price targets; all metadata states contribute to the market pool.
        support=past_counts(df.loc[alive])
        df['past_trade_count']=0
        df['past_token_count']=0
        df.loc[support.index,['past_trade_count','past_token_count']]=support
        df['baseline_eligible']=alive & (df.past_trade_count>=20) & (df.past_token_count>=10)
        df['metadata_present']=df.token.isin(metadata[collection])
        valid_meta={t for t,r in metadata[collection].items() if r['valid_for_primary']}
        mega={t for t,r in metadata[collection].items() if r['is_mega']}
        df['is_mega']=df.token.isin(mega)
        df['metadata_primary_valid']=df.token.isin(valid_meta)
        df['local_image_ready']=df.token.isin(images) if collection=='BAYC' else False
        df['candidate_primary']=df.baseline_eligible & df.metadata_primary_valid
        df['runnable_primary']=df.candidate_primary & df.local_image_ready
        df['split']=np.where(df.time<SPLIT,'development',np.where(df.time<END,'temporal_test','outside_cutoff'))
        training_tokens=set(df.loc[df.candidate_primary & (df.time<SPLIT),'token'])
        history_tokens=set(df.loc[df.base_clean & (df.time<SPLIT),'token'])
        df['known_training_token']=df.token.isin(training_tokens)
        df['observed_pretest_trade']=df.token.isin(history_tokens)
        periods={}
        for label,mask in [('development',df.time<SPLIT),('temporal_test',(df.time>=SPLIT)&(df.time<END)),
                           ('post_submission_descriptive',(df.time>=pd.Timestamp('2026-03-19',tz='UTC'))&(df.time<END))]:
            periods[label]={stage:count_summary(df,mask & df[stage]) for stage in
                             ['base_clean','baseline_eligible','metadata_present','candidate_primary','runnable_primary']}
            periods[label]['known_training']=count_summary(df,mask & df.candidate_primary & df.known_training_token)
            periods[label]['unseen_training']=count_summary(df,mask & df.candidate_primary & ~df.known_training_token)
        folds=[]
        for name,begin,end in OUTERS:
            begin,end=pd.Timestamp(begin,tz='UTC'),pd.Timestamp(end,tz='UTC')
            train=df.candidate_primary & (df.time<begin)
            val=df.candidate_primary & (df.time>=begin)&(df.time<end)
            train_ids=set(df.loc[train,'token'])
            folds.append(dict(name=name,train_end_exclusive=str(begin),validation_end_exclusive=str(end),
                              train=count_summary(df,train),validation=count_summary(df,val),
                              unseen_validation=count_summary(df,val & ~df.token.isin(train_ids))))
        years=[]
        for year,group in df.groupby(df.time.dt.year):
            years.append(dict(year=int(year),raw=count_summary(group,pd.Series(True,index=group.index)),
                base_clean=count_summary(group,group.base_clean),candidate_primary=count_summary(group,group.candidate_primary),
                missing_metadata=count_summary(group,~group.metadata_present)))
        raw_tokens=set(df.token.dropna().astype(int))
        missing=sorted(raw_tokens-set(metadata[collection]))
        token_counts=df.groupby('token').size()
        candidate_counts=df[df.candidate_primary].groupby('token').size()
        for token in sorted(raw_tokens|set(metadata[collection])):
            meta=metadata[collection].get(token)
            all_token_rows.append(dict(collection=collection,contract=CONTRACTS[collection],token_id=token,
                raw_trade_count=int(token_counts.get(token,0)),candidate_trade_count=int(candidate_counts.get(token,0)),
                metadata_present=meta is not None,generation=meta['generation'] if meta else None,
                is_mega=meta['is_mega'] if meta else None,
                image_ready=(token in images if collection=='BAYC' else False),
                image_uri=meta.get('image_uri') if meta else None))
        jsonl(f'{collection.lower()}_trade_eligibility.jsonl',(
            dict(collection=collection,source_row=int(r.source_row),unique_trade_id=r.unique_trade_id,
                 token_id=int(r.token) if pd.notna(r.token) else None,time=str(r.time),
                 first_exclusion=r.first_exclusion,base_clean=bool(r.base_clean),strict_clean=bool(r.strict_clean),
                 observed_multi_nft_tx=bool(r.multiple_nfts_in_observed_tx),reverse_pair_24h=bool(r.reverse_pair_24h),
                 prior_20d_trades=int(r.past_trade_count),prior_20d_tokens=int(r.past_token_count),
                 metadata_present=bool(r.metadata_present),is_mega=bool(r.is_mega),
                 candidate_primary=bool(r.candidate_primary),runnable_primary=bool(r.runnable_primary),
                 split=r.split,known_training_token=bool(r.known_training_token),observed_pretest_trade=bool(r.observed_pretest_trade))
            for r in df.itertuples()))
        summary=dict(raw_rows=len(df),raw_tokens=len(raw_tokens),time_min=str(df.time.min()),time_max=str(df.time.max()),
            cleaning_flow=flow,base_clean=count_summary(df,df.base_clean),strict_clean=count_summary(df,df.strict_clean),
            missing_baseline_support=count_summary(df,df.base_clean & ~df.baseline_eligible),
            metadata_linked_raw=count_summary(df,df.metadata_present),missing_metadata_ids=missing,
            missing_metadata_over_19493=sum(t>19493 for t in missing),mega_raw=count_summary(df,df.is_mega),
            candidate_primary=count_summary(df,df.candidate_primary),runnable_primary=count_summary(df,df.runnable_primary),
            raw_currency_counts=df.currency_symbol.value_counts().to_dict(),
            excluded_bpeth_sensitivity=count_summary(df,df.candidate_primary & (df.currency_symbol=='bpETH')),
            candidate_strict_flags=count_summary(df,df.candidate_primary & ~df.strict_clean),
            multiple_nft_tx_raw=count_summary(df,df.multiple_nfts_in_observed_tx),
            reverse_pair_24h_clean=count_summary(df,df.base_clean & df.reverse_pair_24h),
            periods=periods,folds=folds,years=years,
            candidate_sales_per_token={str(k):int(v) for k,v in candidate_counts.value_counts().sort_index().items()},
            raw_usd_missing=int((df.amount_usd=='').sum()),source_query_verified=False)
        outputs[collection]=summary
        emit('trade_complete',collection=collection,base_clean=summary['base_clean'],
             candidate=summary['candidate_primary'],runnable=summary['runnable_primary'],test=periods['temporal_test'])
    jsonl('token_manifest.jsonl',all_token_rows)
    save('trade_audit.json',outputs)
    return outputs


def unit_tests():
    frame=pd.DataFrame({'time':pd.to_datetime(['2024-01-01','2024-01-01','2024-01-21','2024-01-22'],utc=True),
                        'token':[1,2,1,3]})
    counts=past_counts(frame)
    assert counts.past_trade_count.tolist()==[0,0,2,1]
    assert counts.past_token_count.tolist()==[0,0,2,1]
    assert normalize('None','hat')=='hat_None'
    assert normalize('M1 None','hat')=='M1 None'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    unit_tests()
    started=time.perf_counter()
    inputs=[ROOT/'tfidf20260103.csv',ROOT/'bayc_y_tfidf_20260103.csv',ROOT/'bayc_metadata_cache.json',
            ROOT/'revision'/'MAYC_속성.csv']+[ROOT/'revision'/f'{c}_all_trades.csv' for c in ('bayc','mayc')]
    fingerprints=[fingerprint(p) for p in inputs]
    metadata,meta_summary,json_manifest=metadata_audit()
    images,image_summary,image_manifest=image_audit()
    frames={c:pd.read_csv(ROOT/'revision'/f'{c.lower()}_all_trades.csv',dtype=str,keep_default_na=False)
            for c in ('BAYC','MAYC')}
    trade_summary=trade_audit(frames,metadata,images)
    for record in fingerprints+json_manifest+image_manifest:
        path=ROOT/record['path']
        assert path.stat().st_size==record['size'] and path.stat().st_mtime_ns==record['mtime_ns'], record['path']
    qa=dict(unit_tests_passed=True,source_sizes_mtimes_unchanged=True,
            raw_csvs_hashed=True,mayc_jsons_hashed=True,bayc_images_hashed=True,
            trained_models=False,downloaded_images=False,performance_metrics_computed=False,
            no_test_price_targets_saved=True,seconds=time.perf_counter()-started,
            pandas=pd.__version__,numpy=np.__version__,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save('source_manifest.json',fingerprints)
    save('qa.json',qa)
    save('summary.json',dict(metadata=meta_summary,images=image_summary,trades=trade_summary,qa=qa))
    emit('complete',**qa)


if __name__=='__main__': main()
