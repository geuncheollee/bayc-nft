"""Validate package integrity, public boundaries, row alignment and source coverage."""
from pathlib import Path
import ast,csv,hashlib,json,re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import reproduce
def main():
    n=reproduce.verify_integrity();catalog=reproduce.readj(ROOT/'data/catalog.json');contract=reproduce.readj(ROOT/'expected/reproduction_contract.json')
    expected_keys={'row_id','token_group','token_cluster_order','target_lrp','predictions_lrp','model_names','calendar_block_14d','seen_in_2022_2024','strict_clean','strict_no_wei'}
    loaded={x['id']:reproduce.load_dataset(x) for x in catalog}
    for item in catalog:
        d=loaded[item['id']]
        assert set(d)-{'token_index'}==expected_keys
        assert len(d['target_lrp'])==item['n_transactions']
        assert len(d['token_cluster_order'])==item['n_tokens']
        assert list(d['model_names'])==item['models']
        assert np.all(d['strict_no_wei']<=d['strict_clean'])
        base=loaded[item['collection']+'_original_all'];ix={int(x):i for i,x in enumerate(base['row_id'])}
        indexes=np.array([ix[int(x)] for x in d['row_id']])
        for key in ['target_lrp','token_group','calendar_block_14d','seen_in_2022_2024','strict_clean','strict_no_wei']:
            assert np.array_equal(d[key],base[key][indexes]),(item['id'],key)
    for c in ['BAYC','MAYC']:
        all_=loaded[c+'_original_all'];seen=loaded[c+'_original_seen'];unseen=loaded[c+'_original_unseen']
        assert not set(seen['row_id'])&set(unseen['row_id'])
        assert set(seen['row_id'])|set(unseen['row_id'])==set(all_['row_id'])
        assert not set(seen['token_group'])&set(unseen['token_group'])
        assert seen['seen_in_2022_2024'].all() and not unseen['seen_in_2022_2024'].any()
    source_tables={}
    for r in contract:
        if r['source_file'] not in source_tables:
            with (ROOT/'expected/source_tables'/r['source_file']).open(encoding='utf-8-sig',newline='') as f:source_tables[r['source_file']]=list(csv.DictReader(f))
        source=source_tables[r['source_file']][r['source_row']]
        assert float(source[r['source_field']])==r['expected']
        names=list(loaded[r['dataset']]['model_names'])
        for key in ['model','candidate','reference']:
            if key in r:assert r[key] in names,(r['dataset'],r[key])
    for path,rs in source_tables.items():
        assert {r['source_row'] for r in contract if r['source_file']==path}==set(range(len(rs))),path
    forbidden={'.docx','.pdf','.joblib','.pkl','.parquet','.jsonl','.npy'}
    secret=re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
    # Inspect only immutable manifest-listed files, never unrelated user outputs.
    for record in reproduce.readj(ROOT/'MANIFEST_SHA256.json'):
        p=ROOT/record['path']
        assert p.suffix.lower() not in forbidden,p
        assert p.name not in {'Comments.txt','Revised_Manuscript.docx'}
        if p.suffix=='.npz':assert p.parent==ROOT/'data'
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'))
        if p.suffix in {'.py','.md','.json','.csv','.txt','.cff'}:assert not secret.search(p.read_text(encoding='utf-8-sig')),p
    for r in reproduce.readj(ROOT/'provenance/october_source_files.json'):
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['source_sha256']==r['public_sha256']
    result=dict(passed=True,manifest_files=n,datasets=len(catalog),source_tables=len(source_tables),
        covered_source_rows=sum(len(x) for x in source_tables.values()),expected_numeric_cells=len(contract),
        boundary='Pseudonymised numeric outputs; no direct transaction/token/wallet identifiers in released arrays')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
