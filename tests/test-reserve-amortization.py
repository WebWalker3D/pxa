#!/usr/bin/env python3
"""CPU-only regression for the production reserve accounting expressions.

Run: python3 tests/test-reserve-amortization.py (requires a C++17 compiler).
Extracts and executes the actual C++ transition, latch condition and served-counter
block, so this is not a separately reimplemented copy of the algorithm.
GPU/inference qualification is described in the accompanying issue.
"""
from pathlib import Path
import os
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / 'src/llama.cpp').read_text()
header = (root / 'src/llama-context.h').read_text()
def extract(pattern, text=source):
    match = re.search(pattern, text, re.S)
    if not match:
        raise AssertionError('Reserve accounting regression: missing ' + pattern)
    return match.group(0)

transition = extract(r'lctx\.pxa_reserve_thrash_run\s*=.*?;')
reset = extract(r'lctx\.pxa_reserve_served\s*=\s*0;')
condition = re.search(r'if \((pxa_reserve_guard_enabled\(\) && [^\n]+)\) \{', source).group(1)
served = extract(r'if \(!pxa_gs\.s && lctx\.pxa_reserve_served < INT_MAX\) \{.*?\}')
fields = '\n'.join(extract(r'int\s+' + name + r'\s*=\s*0;', header) for name in
                   ['pxa_reserve_count', 'pxa_reserve_thrash_run', 'pxa_reserve_served'])
code = '''#include <cassert>
#include <climits>
struct Context { FIELDS bool pxa_reserve_latched_off = false; };
static int pxa_reserve_min_served() { return 4; }
static int pxa_reserve_max() { return 24; }
static bool pxa_reserve_guard_enabled() { return true; }
void reserve(Context & lctx) {
    ++lctx.pxa_reserve_count;
    TRANSITION
    RESET
    if (CONDITION) lctx.pxa_reserve_latched_off = true;
}
void serve(Context & lctx, bool slot = false) {
    struct { bool s; } pxa_gs{slot};
    SERVED
}
int main() {
    for (int k : {4, 5, 8, 512}) {
        Context c;
        for (int n=0; n<1000; ++n) {
            reserve(c);
            assert(!c.pxa_reserve_latched_off);
            for (int j=0; j<k; ++j) serve(c);
        }
        assert(c.pxa_reserve_count == 1000);
    }
    for (int k : {1, 2, 3}) {
        Context c;
        for (int n=1; n<=25; ++n) {
            reserve(c);
            assert(c.pxa_reserve_latched_off == (n == 25));
            for (int j=0; j<k; ++j) serve(c);
        }
    }
    Context c;
    for (int n=0; n<20; ++n) { reserve(c); serve(c); }
    for (int j=0; j<4; ++j) serve(c);
    reserve(c);
    assert(c.pxa_reserve_thrash_run == 1 && !c.pxa_reserve_latched_off);
    c.pxa_reserve_served = INT_MAX;
    serve(c); assert(c.pxa_reserve_served == INT_MAX);
    c.pxa_reserve_served = 7;
    serve(c, true); assert(c.pxa_reserve_served == 7);
}
'''.replace('#include <climits>', '#include <climits>\n#include <initializer_list>')
for marker, value in [('FIELDS',fields),('TRANSITION',transition),('RESET',reset),('CONDITION',condition),('SERVED',served)]:
    code = code.replace(marker,value)
with tempfile.TemporaryDirectory() as directory:
    src = Path(directory)/'reserve.cpp'; binary = Path(directory)/'reserve'
    src.write_text(code)
    subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-Wall','-Wextra','-Werror',str(src),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
print('PASS: amortized handoffs, rapid-switch latch, reset, saturation, verify-slot exclusion')
