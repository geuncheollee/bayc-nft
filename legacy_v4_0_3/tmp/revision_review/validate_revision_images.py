"""Read-only image/source audit; writes only a new derived audit directory.

No price targets, model fitting, image modification, or network requests.
"""
import hashlib
import io
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import PIL
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'revision/data_audit_20260908'
OUT = ROOT / 'revision/image_validation_20260909'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def jsonlines(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(name, data):
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def snapshot(path):
    stat = path.stat()
    return dict(path=str(path.relative_to(ROOT)), size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, sha256=sha(path))


def inspect_bytes(raw):
    if raw[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('Not PNG magic')
    with Image.open(io.BytesIO(raw)) as image:
        image.verify()  # PNG chunk integrity, separate from full pixel decoding.
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        rgba = image.convert('RGBA')
        alpha = rgba.getchannel('A').histogram()
        rgb_extrema = rgba.convert('RGB').getextrema()
        return dict(format=image.format, mode=image.mode, width=image.width,
                    height=image.height, frames=getattr(image, 'n_frames', 1),
                    rgba_sha256=hashlib.sha256(rgba.tobytes()).hexdigest(),
                    nonopaque_pixels=sum(alpha[:255]), zero_alpha_pixels=alpha[0],
                    total_pixels=image.width * image.height,
                    flat_rgb=all(a == b for a, b in rgb_extrema),
                    has_icc_profile=bool(image.info.get('icc_profile')),
                    exif_orientation=image.getexif().get(274), decode_ok=True)


def inspect(task):
    collection, tid, path, previous, expected_uri = task
    record = dict(collection=collection, token_id=tid,
                  path=str(path.relative_to(ROOT)), expected_image_uri=expected_uri)
    issues = []
    try:
        before = path.stat()
        raw = path.read_bytes()
        record.update(size_bytes=len(raw), mtime_ns=before.st_mtime_ns,
                      sha256=hashlib.sha256(raw).hexdigest())
        record.update(inspect_bytes(raw))
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            issues.append('file_changed_during_read')
        if previous is None:
            issues.append('missing_prior_manifest_record')
        else:
            record['previous_sha256_matches'] = record['sha256'] == previous.get('sha256')
            if not record['previous_sha256_matches']:
                issues.append('sha256_mismatch')
            if len(raw) != previous.get('size_bytes', previous.get('size')):
                issues.append('size_mismatch')
            for key in ('width', 'height', 'mode'):
                if record[key] != previous.get(key):
                    issues.append(key + '_mismatch')
            if collection == 'BAYC' and record['rgba_sha256'] != previous.get('rgba_sha256'):
                issues.append('previous_rgba_hash_mismatch')
        if record['frames'] != 1:
            issues.append('animated_image')
        if record['flat_rgb'] or record['zero_alpha_pixels'] == record['total_pixels']:
            issues.append('flat_or_invisible_image')
        if collection == 'MAYC':
            cid = (previous or {}).get('cid')
            record.update(recorded_cid=cid, recorded_gateway=(previous or {}).get('gateway'),
                          source_reference_status='missing_cid' if not cid else 'cid_matches_json')
            if cid and 'ipfs://' + cid != expected_uri:
                issues.append('metadata_cid_mismatch')
                record['source_reference_status'] = 'cid_mismatch'
            if record['format'] != 'PNG' or (record['width'], record['height']) != (1262, 1262):
                issues.append('unexpected_mayc_spec')
    except Exception as exc:
        record['error'] = type(exc).__name__ + ': ' + str(exc)
        issues.append('read_or_decode_error')
    record['issues'] = issues
    record['local_integrity_ok'] = not issues
    return record


def unit_tests():
    buffer = io.BytesIO()
    image = Image.new('RGBA', (2, 2), (1, 2, 3, 255))
    image.putpixel((0, 0), (9, 8, 7, 0))
    image.save(buffer, format='PNG')
    raw = buffer.getvalue()
    result = inspect_bytes(raw)
    assert result['nonopaque_pixels'] == 1 and not result['flat_rgb']
    for bad in (b'not an image', raw[:35]):
        try:
            inspect_bytes(bad)
        except Exception:
            pass
        else:
            raise AssertionError('Corrupt bytes accepted')
    return dict(alpha_pixel_check=True, full_decode=True, corrupt_bytes_rejected=True)


def main():
    start = time.perf_counter()
    tests = unit_tests()
    OUT.mkdir(exist_ok=False)
    candidates = json.loads((OLD / 'candidate_token_ids_v1.json').read_text(encoding='utf-8'))
    bayc_previous = jsonlines(OLD / 'bayc_image_manifest.jsonl')
    mayc_previous = jsonlines(ROOT / 'MAYC_Images/download_manifest.jsonl')
    meta = jsonlines(OLD / 'metadata_normalized.jsonl')
    meta_map = {(r['collection'], r['token_id']): r for r in meta}
    inputs = [p for p in OLD.iterdir() if p.suffix in ('.json', '.jsonl')]
    inputs += [ROOT / 'MAYC_Images/download_manifest.jsonl', Path(__file__)]
    input_snapshots = [snapshot(p) for p in inputs]
    save('input_snapshot.json', input_snapshots)
    originals = json.loads((OLD / 'source_manifest.json').read_text(encoding='utf-8'))
    source_checks = []
    for previous in originals:
        actual = snapshot(ROOT / previous['path'])
        actual['matches_previous_sha256'] = actual['sha256'] == previous['sha256']
        source_checks.append(actual)
    assert all(r['matches_previous_sha256'] for r in source_checks), 'Raw source changed; stop'
    save('raw_source_recheck.json', source_checks)
    metadata_checks = []
    for previous in jsonlines(OLD / 'mayc_json_manifest.jsonl'):
        actual = snapshot(ROOT / previous['path'])
        metadata_checks.append(dict(token_id=previous['token_id'], **actual,
                                    matches_previous_sha256=actual['sha256'] == previous['sha256']))
    save('mayc_metadata_recheck.json', metadata_checks)
    assert all(r['matches_previous_sha256'] for r in metadata_checks), 'MAYC source JSON changed'
    tasks = []
    inventories = {}
    for collection, folder, rows in [('BAYC', ROOT/'Images', bayc_previous),
                                      ('MAYC', ROOT/'MAYC_Images', mayc_previous)]:
        counts = Counter(r['token_id'] for r in rows)
        assert all(n == 1 for n in counts.values()), 'Duplicate manifest IDs require review'
        previous_map = {r['token_id']: r for r in rows}
        files = {int(p.stem): p for p in folder.glob('*.png') if p.stem.isdecimal()}
        candidates_set = set(candidates[collection])
        inventories[collection] = dict(file_tokens=len(files), manifest_tokens=len(rows),
            missing_candidate_ids=sorted(candidates_set-set(files)),
            files_without_manifest=sorted(set(files)-set(previous_map)),
            manifest_without_files=sorted(set(previous_map)-set(files)),
            additional_file_tokens_outside_candidates=len(set(files)-candidates_set))
        for tid, path in sorted(files.items()):
            tasks.append((collection, tid, path, previous_map.get(tid),
                          meta_map.get((collection, tid), {}).get('image_uri')))
    results = []
    with (OUT / 'image_integrity_manifest.jsonl').open('x', encoding='utf-8') as stream:
        with ThreadPoolExecutor(max_workers=4) as executor:
            for index, result in enumerate(executor.map(inspect, tasks), 1):
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                if index % 500 == 0 or index == len(tasks):
                    stream.flush()
                    print(json.dumps(dict(images_checked=index, total=len(tasks),
                        failed=sum(not r['local_integrity_ok'] for r in results),
                        seconds=round(time.perf_counter()-start, 1))), flush=True)
    duplicate_groups = {}
    for key in ('sha256', 'rgba_sha256'):
        groups = defaultdict(list)
        for r in results:
            if r.get(key):
                groups[r[key]].append([r['collection'], r['token_id']])
        duplicate_groups[key] = [dict(hash=k, tokens=v) for k,v in groups.items() if len(v)>1]
    unchanged = all((ROOT/r['path']).stat().st_size == r['size'] and
                    (ROOT/r['path']).stat().st_mtime_ns == r['mtime_ns'] for r in input_snapshots)
    images_unchanged = all((ROOT/r['path']).stat().st_size == r.get('size_bytes') and
                          (ROOT/r['path']).stat().st_mtime_ns == r.get('mtime_ns') for r in results)
    stats = {}
    for c in ('BAYC', 'MAYC'):
        subset = [r for r in results if r['collection'] == c]
        stats[c] = dict(checked=len(subset), passed=sum(r['local_integrity_ok'] for r in subset),
            errors=[r for r in subset if not r['local_integrity_ok']],
            total_bytes=sum(r.get('size_bytes',0) for r in subset),
            specs=dict(Counter(str((r.get('width'),r.get('height'),r.get('mode'))) for r in subset)),
            images_with_nonopaque_pixels=sum(r.get('nonopaque_pixels',0)>0 for r in subset),
            source_reference_missing_ids=[r['token_id'] for r in subset if r.get('source_reference_status')=='missing_cid'])
    summary = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
        seconds=round(time.perf_counter()-start, 2), pillow_version=PIL.__version__, workers=4,
        inventory=inventories, collections=stats, duplicates=duplicate_groups,
        inputs_unchanged_during_run=unchanged, images_unchanged_during_run=images_unchanged,
        unit_tests=tests, raw_sources_rehashed=True, mayc_json_rehashed=len(metadata_checks),
        source_images_written=False, images_downloaded=False, models_fit=False,
        upstream_onchain_authentication=False, independent_cid_dag_verification=False)
    save('image_validation_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    assert unchanged and images_unchanged
    assert all(r['local_integrity_ok'] for r in results), 'Image issues need review'


if __name__ == '__main__':
    main()
