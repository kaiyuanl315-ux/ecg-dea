"""Unpack lossless row_to_json records exported by Navicat read-only queries.

Run into a new directory to verify against delivered CSV content. Source files are
never modified. CSV nulls use empty fields; textual values are preserved.
"""
from pathlib import Path
import argparse, collections, csv, gzip, io, json

def unpack(source, output):
    output.mkdir(parents=True, exist_ok=True)
    writers, handles, counts = {}, {}, collections.Counter()
    try:
        with source.open(newline='') as stream:
            for row in csv.DictReader(stream):
                name = row['source_table']
                if name not in {'ventilation','vasoactive_agent','d_items',
                                'chartevents_resp','inputevents_pressors','procedureevents_support'}:
                    raise ValueError('Unexpected table name')
                record = json.loads(row['record'])
                if name not in writers:
                    path = output / (name + '.csv.gz')
                    if path.exists():
                        raise FileExistsError(path)
                    raw = path.open('wb')
                    zipped = gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0)
                    handle = io.TextIOWrapper(zipped, newline='')
                    handles[name] = (handle, raw)
                    writers[name] = csv.DictWriter(handle, fieldnames=list(record))
                    writers[name].writeheader()
                writers[name].writerow(record)
                counts[name] += 1
    finally:
        for handle, raw in handles.values():
            handle.close()
            raw.close()
    print(json.dumps(dict(counts), indent=2))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('source', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    unpack(args.source, args.output)
