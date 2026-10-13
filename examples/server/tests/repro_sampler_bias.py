#!/usr/bin/env python3
"""Run against a single-slot PXA server with a model that has an EOS token.

With natural EOS enabled at server startup, unpatched v3.1 exits 1; a fixed server (or isolating gateway) exits 0.
Uses three public one-token prompts and no model-specific token IDs.
API key, if needed, is read from PXA_API_KEY and never printed.
"""
import argparse
import json
import os
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8080')
    args = parser.parse_args()
    results = []
    for case, extra in [('force', {'ignore_eos': True, 'logit_bias': {}}),
                        ('next_omitted', {}), ('next_explicit_reset', {'logit_bias': {}})]:
        data = json.dumps({'prompt': 'A short greeting:', 'n_predict': 1, **extra}).encode()
        headers = {'Content-Type': 'application/json'}
        if os.environ.get('PXA_API_KEY'):
            headers['Authorization'] = 'Bearer ' + os.environ['PXA_API_KEY']
        req = urllib.request.Request(args.base_url.rstrip('/') + '/completion', data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=90) as response:
            payload = json.load(response)
        results.append({'case': case, 'ignore_eos': payload['generation_settings']['ignore_eos']})
    print(json.dumps(results, indent=2))
    assert [x['ignore_eos'] for x in results] == [True, False, False], 'EOS suppression leaked into the next request'


if __name__ == '__main__':
    main()
