"""Export tables from released aggregates, without fitting models or private data."""
from pathlib import Path
import argparse, csv, json, shutil

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('reproduced_tables'))
    args = parser.parse_args()
    out = args.output_dir.resolve()
    # Never write generated tables over the frozen public evidence.
    if out == ROOT or out in (ROOT/'summary', ROOT/'provenance') or ROOT/'results' in out.parents:
        parser.error('Choose a separate output directory, not the archived evidence.')
    out.mkdir(parents=True, exist_ok=True)
    results = json.loads((ROOT/'summary/performance.json').read_text(encoding='utf-8'))
    keys = ['collection','group','transaction_rmse','transaction_mae','transaction_r2','equal_token_rmse']
    with (out/'representative_performance.csv').open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    lines = ['# Development-selected representatives', '',
             'Selection uses pre-2025 validation, not evaluation ranking. Lower RMSE is better.', '',
             '| Collection | Group | RMSE | MAE | R2 |', '|---|---|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['collection']} | {r['group']} | {r['transaction_rmse']:.6f} | {r['transaction_mae']:.6f} | {r['transaction_r2']:.6f} |")
    (out/'representative_performance.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for p in (ROOT/'summary').glob('*.csv'):
        shutil.copy2(p, out/p.name)
    print(json.dumps({'output': str(out), 'representatives': len(results),
                      'models_refitted': False, 'private_data_accessed': False}))

if __name__ == '__main__':
    main()
