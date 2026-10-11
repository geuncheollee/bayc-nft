"""Export one numerical dataset as CSV (no original identifiers)."""
from pathlib import Path
import argparse,csv,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import reproduce
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--output-file',type=Path,required=True);a=ap.parse_args()
    reproduce.verify_integrity();item=next(x for x in reproduce.readj(reproduce.ROOT/'data/catalog.json') if x['id']==a.dataset);d=reproduce.load_dataset(item)
    cols=['row_id','token_group','target_lrp','calendar_block_14d','seen_in_2022_2024','strict_clean','strict_no_wei']
    with a.output_file.open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(cols+['prediction__'+n for n in d['model_names']])
        for i in range(len(d['target_lrp'])):w.writerow([d[k][i] for k in cols]+list(d['predictions_lrp'][i]))
    print('Wrote',len(d['target_lrp']),'rows. Preserve the NPZ token_cluster_order for exact seeded bootstrap replay.')
if __name__=='__main__':main()
