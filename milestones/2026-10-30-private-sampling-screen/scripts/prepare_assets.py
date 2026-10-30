"""Retrieve public pinned assets locally; never publish weights or corpus text."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = args.asset_root.resolve()
    model = json.loads((root/'models/Pythia-14M/download_manifest.json').read_text())
    wiki = json.loads((root/'data/manifest.json').read_text())
    ptb = json.loads((root/'data/ptb_test_manifest.json').read_text())
    items = [(root/'models/Pythia-14M'/x['file'], x['url'], x['sha256']) for x in model['files']]
    items += [(root/'data/wikitext2_test.txt', wiki['url'], wiki['sha256']),
        (root/'data/ptb_test.txt', 'https://raw.githubusercontent.com/wojzaremba/lstm/'+ptb['commit']+'/data/ptb.test.txt', ptb['sha256'])]
    for path, url, expected in items:
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)
        else:
            assert not args.verify_only, f'Missing asset: {path}'
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            assert hashlib.sha256(data).hexdigest() == expected, f'Upstream bytes changed: {url}'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        print('verified', path.name)


if __name__ == '__main__':
    main()
