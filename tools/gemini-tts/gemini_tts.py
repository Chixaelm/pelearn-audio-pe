#!/usr/bin/env python3
"""Regenerate PELearn narration with Gemini TTS.

Reads the narration units exported from the built reader page (export.cjs), synthesises
every block with the Gemini API, joins each unit's blocks into one AAC .m4a and writes a
manifest.json in the format the reader expects:

  units[unitId] = {"f": file, "d": seconds, "s": bytes, "b": [[blockHash, start, end], ...]}

The reader matches b[ord][0] against the FNV-1a hash of each block's spoken text, so the
blocks must stay in export order and keep their hashes.

Subcommands:
  synth     call Gemini for every block not yet cached (resumable)
  assemble  build .m4a files + manifest.json into --out, removing stale .m4a files there
  sample    narrate one unit with several voices for comparison
  report    show cached/failed/outlier counts

Auth: the environment's network secret injects x-goog-api-key for the API host. If
GEMINI_API_KEY is set in the environment instead, it is sent as that header.
"""
import argparse
import base64
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed

API = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
RATE = 24000          # output sample rate (matches the existing files)
LEAD = 0.08           # silence before the first block
GAP_AFTER = {'heading': 0.55, 'text': 0.45}
TAIL = 0.35
STYLE_VERSION = 'v1'  # bump to force fresh synthesis when the prompt changes

PROFILE = (
    "# AUDIO PROFILE: Course narrator\n"
    "A calm, articulate British English narrator recording an audiobook-style course on "
    "private equity, accounting and corporate finance for professionals.\n\n"
    "### DIRECTOR'S NOTES\n"
    "Accent: British English, Southern England (Received Pronunciation).\n"
    "Style: clear, warm and measured, like a good lecturer explaining to one listener. Not theatrical.\n"
    "Pacing: steady and unhurried, with natural pauses at commas and full stops.\n"
    "Read the transcript exactly as written: do not add, skip or repeat words, and do not read these notes.\n"
)
HEADING_NOTE = "This line is a section heading: read it as a title, with a slight pause after it.\n"

# Acronyms said as words. The reader already rewrites EBITDA, EBIT, MOIC, SaaS and a few
# others phonetically (see PHONETIC); everything else in capitals is spelled out.
SAID_AS_WORDS = {'WACC': 'whack', 'PIK': 'pick', 'ARPU': 'are-poo', 'NOPAT': 'no-pat'}
PHONETIC = {'ee-bit-dah': 'EBITDA', 'ee-bit': 'EBIT', 'moyk': 'MOIC', 'sass': 'SaaS',
            'R-back': 'RBAC', 'A-back': 'ABAC', 'oh-auth': 'OAuth', 'cider': 'CIDR', 'sequel': 'SQL'}
SPELLED = re.compile(r'\b(?:IRR|LBO|GPs?|LPs?|EV|DCF|DPI|TVPI|RVPI|NAV|SPA|CIM|LTM|NTM|ARR|MRR|KPIs?|CFO|CEO|IPO|IC|FCF|CAGR|NDA|VDR|W&I)\b')

lock = threading.Lock()


def log(*a):
    with lock:
        print(*a, flush=True)


def block_text(block):
    return ' '.join(c.strip() for c in block['chunks'] if c.strip())


def prompt_for(block):
    text = block_text(block)
    notes = []
    for word, say in SAID_AS_WORDS.items():
        if re.search(r'\b' + re.escape(word) + r's?\b', text):
            notes.append(f'"{word}" is said "{say}"')
    phon = [f'"{w}" ({a})' for w, a in PHONETIC.items() if re.search(r'\b' + re.escape(w) + r'\b', text)]
    if phon:
        notes.append(', '.join(phon) + (' is a term' if len(phon) == 1 else ' are terms')
                     + ' written phonetically: say ' + ('it' if len(phon) == 1 else 'each')
                     + ' as one smooth word')
    if SPELLED.search(text) or 'P&L' in text:
        notes.append('other abbreviations are spelled out letter by letter ("P&L" is "P and L")')
    p = PROFILE
    if block['kind'] == 'heading':
        p += HEADING_NOTE
    if notes:
        p += 'Pronunciation: ' + '; '.join(notes) + '.\n'
    return p + '\n#### TRANSCRIPT\n' + text


def cache_key(model, voice, block):
    raw = json.dumps([STYLE_VERSION, model, voice, block['kind'], block_text(block)], ensure_ascii=False)
    return hashlib.sha1(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------------------
# PCM helpers (signed 16-bit little-endian mono)

def trim(pcm: bytes, rate=RATE, thresh=350, win_ms=10, pad_ms=40):
    s = array('h'); s.frombytes(pcm)
    w = max(1, rate * win_ms // 1000)
    n = len(s) // w
    def loud(i):
        seg = s[i * w:(i + 1) * w]
        return max(abs(min(seg)), abs(max(seg))) > thresh
    first = next((i for i in range(n) if loud(i)), None)
    if first is None:
        return b''
    last = next(i for i in range(n - 1, -1, -1) if loud(i))
    pad = rate * pad_ms // 1000
    a = max(0, first * w - pad)
    b = min(len(s), (last + 1) * w + pad)
    return s[a:b].tobytes()


def silence(sec):
    return b'\x00\x00' * int(round(sec * RATE))


def to_rate(pcm, src_rate):
    if src_rate == RATE:
        return pcm
    r = subprocess.run(['ffmpeg', '-v', 'error', '-f', 's16le', '-ar', str(src_rate), '-ac', '1', '-i', 'pipe:',
                        '-f', 's16le', '-ar', str(RATE), '-ac', '1', 'pipe:'], input=pcm, capture_output=True, check=True)
    return r.stdout


# ---------------------------------------------------------------------------------------
# Gemini call

class Fatal(Exception):
    pass


def tts(model, voice, prompt, timeout=180):
    body = {
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {
            'responseModalities': ['AUDIO'],
            'speechConfig': {'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': voice}}},
        },
    }
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('GEMINI_API_KEY'):
        headers['x-goog-api-key'] = os.environ['GEMINI_API_KEY']
    req = urllib.request.Request(API.format(model=model), data=json.dumps(body).encode(), headers=headers)
    delay = 5
    for attempt in range(9):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
            parts = (data.get('candidates') or [{}])[0].get('content', {}).get('parts', [])
            audio = [p['inlineData'] for p in parts if 'inlineData' in p]
            if not audio:
                reason = (data.get('candidates') or [{}])[0].get('finishReason')
                raise RuntimeError(f'no audio in response (finishReason={reason})')
            mime = audio[0].get('mimeType', '')
            m = re.search(r'rate=(\d+)', mime)
            rate = int(m.group(1)) if m else RATE
            pcm = b''.join(base64.b64decode(a['data']) for a in audio)
            usage = data.get('usageMetadata', {})
            return to_rate(pcm, rate), mime, usage
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors='replace')[:400]
            if e.code in (400, 401, 403, 404):
                raise Fatal(f'HTTP {e.code}: {msg}')
            ra = e.headers.get('Retry-After')
            wait = float(ra) if ra and ra.replace('.', '').isdigit() else delay
            log(f'  HTTP {e.code}, retry in {wait:.0f}s: {msg[:160]}')
        except (urllib.error.URLError, TimeoutError, RuntimeError, ConnectionError) as e:
            wait = delay
            log(f'  {type(e).__name__}: {e}; retry in {wait:.0f}s')
        time.sleep(wait + random.random() * 2)
        delay = min(delay * 2, 120)
    raise RuntimeError('gave up after retries')


def fake_tts(text):
    # Test-only stand-in: a quiet tone whose length follows the text. Never published.
    sec = max(0.6, len(text) / 15)
    n = int(sec * RATE)
    s = array('h', (int(3000 * ((i // 40) % 2 * 2 - 1)) for i in range(n)))
    return s.tobytes(), 'fake', {}


# ---------------------------------------------------------------------------------------
# Synthesis with cache and sanity checks

def load_units(path, only=None):
    units = json.load(open(path))
    if only:
        rx = re.compile(only)
        units = [u for u in units if rx.search(u['unit'])]
    return units


def cache_paths(cache, key):
    d = os.path.join(cache, key[:2])
    return os.path.join(d, key + '.pcm'), os.path.join(d, key + '.json')


def synth_block(args, block):
    key = cache_key(args.model, args.voice, block)
    pcm_p, meta_p = cache_paths(args.cache, key)
    if os.path.exists(meta_p):
        return key, json.load(open(meta_p)), False
    text = block_text(block)
    best = None
    for attempt in range(3):
        if args.fake:
            pcm, mime, usage = fake_tts(text)
        else:
            pcm, mime, usage = tts(args.model, args.voice, prompt_for(block))
        t = trim(pcm)
        sec = len(t) / 2 / RATE
        cps = len(text) / sec if sec else 0
        ok = sec > 0 and (len(text) < 60 or args.cps_lo <= cps <= args.cps_hi)
        cand = {'sec': round(sec, 3), 'cps': round(cps, 2), 'ok': ok, 'chars': len(text), 'mime': mime,
                'usage': usage, 'attempts': attempt + 1}
        if best is None or (ok and not best[0]['ok']) or (ok == best[0]['ok'] and abs(cps - 15) < abs(best[0]['cps'] - 15)):
            best = (cand, t)
        if ok:
            break
        log(f'  outlier {cps:.1f} chars/s ({sec:.1f}s for {len(text)} chars), retrying: {text[:70]!r}')
    cand, t = best
    os.makedirs(os.path.dirname(pcm_p), exist_ok=True)
    with open(pcm_p + '.tmp', 'wb') as f:
        f.write(t)
    os.replace(pcm_p + '.tmp', pcm_p)
    with open(meta_p, 'w') as f:
        json.dump(cand, f)
    return key, cand, True


def cmd_synth(args):
    units = load_units(args.units, args.only)
    blocks = [b for u in units for b in u['blocks']]
    if args.limit:
        blocks = blocks[:args.limit]
    todo = [b for b in blocks if not os.path.exists(cache_paths(args.cache, cache_key(args.model, args.voice, b))[1])]
    log(f'{len(blocks)} blocks, {len(blocks) - len(todo)} cached, {len(todo)} to synthesise with {args.model}/{args.voice}')
    done = 0
    audio_tokens = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(synth_block, args, b): b for b in todo}
        for f in as_completed(futs):
            try:
                key, meta, fresh = f.result()
            except Fatal as e:
                log('FATAL', e)
                ex.shutdown(cancel_futures=True)
                sys.exit(2)
            except Exception as e:
                log('FAILED block', futs[f]['hash'], e)
                continue
            done += 1
            for m in (meta.get('usage') or {}).get('candidatesTokensDetails', []) or []:
                if m.get('modality') == 'AUDIO':
                    audio_tokens += m.get('tokenCount', 0)
            if done % 25 == 0 or done == len(todo):
                el = time.time() - t0
                log(f'{done}/{len(todo)} blocks, {el / 60:.1f} min elapsed, ~{el / done * (len(todo) - done) / 60:.0f} min left, '
                    f'audio tokens this run {audio_tokens:,}')


def cmd_assemble(args):
    units = load_units(args.units, args.only)
    out = args.out
    man_p = os.path.join(out, 'manifest.json')
    manifest = json.load(open(man_p)) if os.path.exists(man_p) and args.only else {'units': {}}
    manifest.update({'version': 1, 'engine': args.model, 'voice': args.voice, 'voiceLabel': args.label})
    missing = []
    for u in units:
        pcm = bytearray(silence(LEAD))
        b = []
        keys = []
        for i, blk in enumerate(u['blocks']):
            key = cache_key(args.model, args.voice, blk)
            pcm_p, _ = cache_paths(args.cache, key)
            if not os.path.exists(pcm_p):
                missing.append((u['unit'], blk['hash']))
                break
            start = len(pcm) / 2 / RATE
            if b:
                b[-1][2] = round(start, 3)
            pcm += open(pcm_p, 'rb').read()
            pcm += silence(GAP_AFTER.get(blk['kind'], 0.45) if i < len(u['blocks']) - 1 else TAIL)
            b.append([blk['hash'], round(start, 3), None])
            keys.append(key)
        else:
            dur = len(pcm) / 2 / RATE
            b[-1][2] = round(dur, 3)
            fid = hashlib.sha1(json.dumps([args.model, args.voice, keys]).encode()).hexdigest()[:10]
            fname = f"{u['unit']}.{fid}.m4a"
            path = os.path.join(out, fname)
            if not os.path.exists(path):
                subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 's16le', '-ar', str(RATE), '-ac', '1', '-i', 'pipe:',
                                '-c:a', 'aac', '-b:a', args.bitrate, '-movflags', '+faststart', path + '.tmp.m4a'],
                               input=bytes(pcm), check=True)
                os.replace(path + '.tmp.m4a', path)
            probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path],
                                   capture_output=True, text=True, check=True)
            manifest['units'][u['unit']] = {'f': fname, 'd': round(float(probe.stdout.strip()), 2),
                                            's': os.path.getsize(path), 'b': b}
    if missing:
        log(f'{len(missing)} units incomplete (first: {missing[0]}); not written')
        if not args.allow_partial:
            sys.exit(1)
    keep = {v['f'] for v in manifest['units'].values()}
    removed = 0
    if not args.only:
        manifest['units'] = {u['unit']: manifest['units'][u['unit']] for u in units if u['unit'] in manifest['units']}
        keep = {v['f'] for v in manifest['units'].values()}
        for f in os.listdir(out):
            if f.endswith('.m4a') and f not in keep:
                os.remove(os.path.join(out, f)); removed += 1
    with open(man_p, 'w') as f:
        json.dump(manifest, f, separators=(',', ':'))
    tot = sum(v['d'] for v in manifest['units'].values())
    size = sum(v['s'] for v in manifest['units'].values())
    log(f"manifest: {len(manifest['units'])} units, {tot / 3600:.2f} h, {size / 1e6:.0f} MB; removed {removed} old files")


def cmd_sample(args):
    units = load_units(args.units, args.only)
    u = units[0]
    os.makedirs(args.out, exist_ok=True)
    for voice in args.voices.split(','):
        a = argparse.Namespace(**{**vars(args), 'voice': voice})
        pcm = bytearray(silence(LEAD))
        for i, blk in enumerate(u['blocks'][:args.blocks]):
            key, meta, _ = synth_block(a, blk)
            pcm += open(cache_paths(args.cache, key)[0], 'rb').read() + silence(GAP_AFTER[blk['kind']])
        path = os.path.join(args.out, f'sample-{voice}.m4a')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 's16le', '-ar', str(RATE), '-ac', '1', '-i', 'pipe:',
                        '-c:a', 'aac', '-b:a', '64k', path], input=bytes(pcm), check=True)
        log(f'{path}: {len(pcm) / 2 / RATE:.1f}s')


def cmd_report(args):
    units = load_units(args.units, args.only)
    blocks = [b for u in units for b in u['blocks']]
    metas = []
    for b in blocks:
        p = cache_paths(args.cache, cache_key(args.model, args.voice, b))[1]
        if os.path.exists(p):
            metas.append((b, json.load(open(p))))
    sec = sum(m['sec'] for _, m in metas)
    bad = [(b, m) for b, m in metas if not m['ok']]
    log(f'{len(metas)}/{len(blocks)} blocks cached, {sec / 3600:.2f} h speech, {len(bad)} flagged outliers')
    for b, m in bad[:40]:
        log(f"  {m['cps']:5.1f} c/s {m['sec']:6.1f}s  {block_text(b)[:90]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['synth', 'assemble', 'sample', 'report'])
    ap.add_argument('--units', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--model', default='gemini-3.8-flash-tts')
    ap.add_argument('--voice', default='Charon')
    ap.add_argument('--label', default='')
    ap.add_argument('--only', help='regex on unit id')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out')
    ap.add_argument('--bitrate', default='48k')
    ap.add_argument('--voices', default='Charon,Sadaltager,Iapetus,Kore')
    ap.add_argument('--blocks', type=int, default=4)
    ap.add_argument('--cps-lo', type=float, default=9.0)
    ap.add_argument('--cps-hi', type=float, default=24.0)
    ap.add_argument('--allow-partial', action='store_true')
    ap.add_argument('--fake', action='store_true', help='test the pipeline with tones instead of the API')
    args = ap.parse_args()
    {'synth': cmd_synth, 'assemble': cmd_assemble, 'sample': cmd_sample, 'report': cmd_report}[args.cmd](args)


if __name__ == '__main__':
    main()
