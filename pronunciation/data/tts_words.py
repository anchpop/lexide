"""Frozen corpus-word TTS plans: global eval exclusion and pre-request spend ledger.

Selection is offline except for the free Cloud voice-list request. Pronunciations
come from preprocess/examples/word_phones.rs, never from a speech model.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
LANGS = 'ara ces dan deu eng fas fra hin ita jpn kor por rus spa tha zho-hans'.split()
EVAL_ROOT = Path('/data/coding/yap/generate-data/data')
NONSPACE = {'jpn', 'tha', 'zho-hans'}
APOSTROPHES = "’‘ʼ＇`"
HYPHENS = "‐‑‒–—−﹘﹣－"
TRANSLATE = str.maketrans({**{c: "'" for c in APOSTROPHES}, **{c: '-' for c in HYPHENS}})


def normalize(text):
    return unicodedata.normalize('NFKC', text).casefold().translate(TRANSLATE).strip()


def constituents(text):
    text = normalize(text)
    # Include intact compounds AND every apostrophe/hyphen constituent.
    whole = re.findall(r"[^\W\d_]+(?:['-][^\W\d_]+)*", text, re.UNICODE)
    return {text, *whole, *(p for w in whole for p in re.split("['-]", w))} - {''}


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def eval_inventory(root=EVAL_ROOT):
    paths = sorted(root.glob('*/audio/*/manifest.jsonl'))
    assert paths, f'No eval manifests under {root}'
    excluded, nonspace_texts = set(), []
    for path in paths:
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            text = rec['text']
            excluded.update(constituents(text))
            if path.parents[2].name in NONSPACE:
                nonspace_texts.append(normalize(text))
    return excluded, nonspace_texts, [fingerprint(p) for p in paths]


def eligible(word, excluded, nonspace_texts):
    word = normalize(word)
    return (bool(word) and len(word) <= 40
            and any(unicodedata.category(c).startswith('L') for c in word)
            and all(unicodedata.category(c)[0] in 'LM' or c in "'-‌" for c in word)
            and not constituents(word).intersection(excluded)
            # Conservative superset of any tokenizer's segments. This also
            # excludes subwords crossing an ambiguous segmentation boundary.
            and not any(word in text for text in nonspace_texts))


def corpus_words(lang):
    repo = ROOT.parent
    path = repo / 'tagging/data/big' / lang / 'target_language_sentences_tokenization.jsonl'
    tokenized = path.exists()
    if not tokenized:
        path = ROOT / 'data/tts_sentences' / f'{lang}.jsonl'
    if not path.exists():
        path = ROOT / 'data/audio' / lang / 'manifest.jsonl'
    assert path.exists(), f'No corpus for {lang}'
    counts = Counter()
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('source') == 'tts_word':
                continue
            if tokenized:
                words = [t['text'] for t in rec['tokens'] if t.get('pos') not in {'PUNCT', 'SYM', 'X', 'NUM'}]
            else:
                assert lang not in NONSPACE, f'{lang} requires existing corpus tokenizer data'
                words = re.findall(r"[^\W\d_\s]+(?:['’\-‌][^\W\d_\s]+)*", rec['sentence'])
            counts.update(normalize(word) for word in words)
    return counts, fingerprint(path)


def candidates(output, limit=6000):
    excluded, nonspace, evals = eval_inventory()
    provenance = {'normalization': 'NFKC-casefold-apostrophe-hyphen-constituents-v1',
                  'eval_manifests': evals, 'corpora': {}, 'eligible_counts': {}}
    with output.open('x') as f:
        for lang in LANGS:
            counts, corpus = corpus_words(lang)
            words = sorted((w for w in counts if eligible(w, excluded, nonspace)),
                           key=lambda w: (-counts[w], w))
            provenance['corpora'][lang] = corpus
            provenance['eligible_counts'][lang] = len(words)
            for word in words[:limit]:
                f.write(json.dumps({'lang': lang, 'word': word, 'frequency': counts[word]}, ensure_ascii=False) + '\n')
            print(lang, len(words), 'eligible;', min(limit, len(words)), 'candidates')
    output.with_suffix('.provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')


def final_prone(row):
    final = row['phonemes'][-1]
    if row['lang'] == 'fra':
        return '̃' in final  # prioritize isolated French final nasal vowels
    return final in {'s', 'z', 'ʃ', 'ʒ', 'f', 'v', 'θ', 'ð', 't', 'd', 'k', 'ɡ', 'p', 'b', 'n', 'm', 'ŋ', 'ɴ'} or '̃' in final


def freeze(phones, provenance, output):
    from generate_tts import LANG_CONFIG, make_client, get_chirp3_voices

    evidence = json.loads(provenance.read_text())
    excluded, nonspace, evals = eval_inventory()
    assert evals == evidence['eval_manifests'], 'Eval inventory changed'
    rows = defaultdict(list)
    for line in phones.read_text().splitlines():
        row = json.loads(line)
        if row.get('phonemes'):
            assert eligible(row['word'], excluded, nonspace)
            rows[row['lang']].append(row)
    client = make_client()
    requests, stats = [], {}
    for lang in LANGS:
        ranked = sorted(rows[lang], key=lambda r: (-r['frequency'], r['word']))
        # Reserve one quarter for vulnerable final phones, then fill by
        # corpus frequency. If fewer exist, retain every eligible candidate.
        selected = [r for r in ranked if final_prone(r)][:500]
        seen = {r['word'] for r in selected}
        selected += [r for r in ranked if r['word'] not in seen][:2000 - len(selected)]
        if lang == 'fas':
            voices = ['Kore', 'Puck', 'Zephyr']
        else:
            voices = sorted(get_chirp3_voices(client, LANG_CONFIG[lang]))[:3]
            assert len(voices) == 3, f'{lang}: expected three supported voices'
        stats[lang] = {'words': len(selected), 'final_prone': sum(map(final_prone, selected)), 'voices': voices}
        for row in selected:
            for voice in voices:
                req = {**row, 'backend': 'gemini' if lang == 'fas' else 'chirp3',
                       'model': 'gemini-3.8-flash-lite-tts' if lang == 'fas' else 'Chirp3-HD',
                       'voice': voice, 'language_code': 'fa-IR' if lang == 'fas' else LANG_CONFIG[lang],
                       'source': 'tts_word'}
                identity = json.dumps([req[k] for k in ('source', 'backend', 'model', 'lang', 'language_code', 'voice', 'word')], ensure_ascii=False)
                req['file'] = 'tts_word_' + hashlib.sha256(identity.encode()).hexdigest()[:24] + '.wav'
                req['cost_usd'] = ((len(req['word'].encode()) + 256) * 0.5 / 1_000_000
                                   + 128 * 6 / 1_000_000 if lang == 'fas'
                                   else len(req['word']) * 30 / 1_000_000)
                # Gemini reserves the FULL documented serving limits until a
                # usage receipt arrives; never trust audio maxOutputTokens for
                # the hard spending cap. Failed/ambiguous calls keep this debit.
                req['reserve_usd'] = 0.1024 if lang == 'fas' else req['cost_usd']
                requests.append(req)
    estimate = sum(r['cost_usd'] for r in requests)
    assert estimate <= 20, f'Plan exceeds synthesis cap: ${estimate:.6f}'
    plan = {'version': 1, 'provenance': evidence, 'phone_candidates': fingerprint(phones),
            'pricing_url': 'https://cloud.google.com/text-to-speech/pricing',
            'pricing_checked': '2026-10-06', 'cap_usd': 20, 'estimate_usd': estimate,
            'retries': 0, 'stats': stats, 'requests': requests}
    with output.open('x') as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    print(json.dumps({'estimate_usd': estimate, 'requests': len(requests), 'stats': stats}, ensure_ascii=False))


def gemini_word(req, out_dir):
    """One request, no implicit retry; return record and a billed usage receipt."""
    import base64
    import io
    import urllib.request
    import soundfile as sf
    from generate_tts import read_env_key, write_pcm_as_wav

    key = read_env_key('GEMINI_API_KEY')
    assert key, 'Missing GEMINI_API_KEY'
    body = {'contents': [{'role': 'user', 'parts': [
        {'text': req['word'], 'speech_metadata': {'style': 'neutral, natural pace'}}]}],
        'generationConfig': {'responseModalities': ['AUDIO'], 'maxOutputTokens': 128,
                             'speechConfig': {'voiceConfig': {'voice': req['voice']}}}}
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{req['model']}:generateContent",
        json.dumps(body).encode(), {'Content-Type': 'application/json', 'x-goog-api-key': key})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.load(response)
    usage = payload.get('usageMetadata', {})
    cost = req['reserve_usd']
    if 'promptTokenCount' in usage and 'candidatesTokenCount' in usage:
        cost = usage['promptTokenCount'] * 0.5e-6 + usage['candidatesTokenCount'] * 6e-6
        assert cost <= req['reserve_usd'], 'Provider exceeded documented serving limits'
    else:
        return None, cost, usage  # pilot must establish metering before bulk dispatch
    candidates = payload.get('candidates', [])
    if not candidates or candidates[0].get('finishReason') != 'STOP':
        return None, cost, usage
    parts = candidates[0].get('content', {}).get('parts', [])
    inline = next((p['inlineData'] for p in parts if 'inlineData' in p), None)
    if inline is None:
        return None, cost, usage
    audio = base64.b64decode(inline['data'])
    # 3.8 returns WAV, unlike the legacy 3.1 raw PCM path.
    samples, sr = sf.read(io.BytesIO(audio), dtype='int16')
    assert samples.ndim == 1 and samples.size, 'Expected nonempty mono audio'
    duration = write_pcm_as_wav(samples.astype('<i2').tobytes(), sr, out_dir / req['file'])
    return {'file': req['file'], 'sentence': req['word'], 'source': 'tts_word',
            'voice': f"gemini:{req['lang']}:{req['voice']}", 'duration_sec': duration,
            'tts_backend': 'gemini', 'tts_model': req['model']}, cost, usage


def assert_no_running_audit():
    for process in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command = process.read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        assert not (command[0].endswith(b'lexide-preprocess') and b'audit' in command), \
            'Existing CF audit must finish before manifest writes'


def execute(plan_path, output, workers, rps):
    """Reserve before every request; an uncertain attempt remains fully charged.

    Ledger is immutable events with fsync, protected by a process lock. Resume
    never retries a reserved request, including one interrupted before receipt.
    Explicit URLError failures may be retried, retaining the uncertain original
    debit as well as reserving the retry. Successful requests are never repeated.
    """
    import fcntl
    import os
    import threading
    from google.cloud import texttospeech
    from generate_tts import make_client, synthesize_one, run_pool

    # Synthesis may populate an isolated staging tree while a main-corpus
    # audit is running. Installation into the audited tree is a separate gate.
    if output.resolve() == (ROOT / 'data/audio').resolve():
        assert_no_running_audit()
    plan = json.loads(plan_path.read_text())
    assert plan['version'] == 1 and plan['retries'] == 0
    assert plan['estimate_usd'] <= plan['cap_usd'] <= 20
    excluded, nonspace, evals = eval_inventory()
    assert evals == plan['provenance']['eval_manifests'], 'Eval inventory changed'
    assert all(eligible(r['word'], excluded, nonspace) for r in plan['requests'])
    plan_hash = fingerprint(plan_path)['sha256']
    ledger_path = plan_path.with_suffix('.ledger.jsonl')
    with ledger_path.open('a+') as ledger:
        fcntl.flock(ledger, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger.seek(0)
        events = [json.loads(line) for line in ledger]
        assert all(e['plan_sha256'] == plan_hash for e in events), 'Plan changed after spending'
        latest = {event['file']: event for event in events}
        debits = {file: event['debit_usd'] for file, event in latest.items()}
        mutex = threading.Lock()
        total_debit = sum(debits.values())

        def log(req, debit, state, **extra):
            nonlocal total_debit
            ledger.write(json.dumps({'plan_sha256': plan_hash, 'file': req['file'],
                                    'debit_usd': debit, 'state': state, **extra}) + '\n')
            ledger.flush()
            os.fsync(ledger.fileno())
            total_debit += debit - debits.get(req['file'], 0)
            debits[req['file']] = debit

        client = make_client()
        config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                                         sample_rate_hertz=16000)
        # Complete the predictable character-priced requests before spending
        # the remaining budget on Gemini's metered audio output.
        for lang in sorted(LANGS, key=lambda lang: lang == 'fas'):
            out_dir = output / lang
            out_dir.mkdir(parents=True, exist_ok=True)
            todo = [(r,) for r in plan['requests'] if r['lang'] == lang and (
                r['file'] not in debits or (
                    latest[r['file']]['state'] == 'failed'
                    and latest[r['file']].get('error') == 'URLError'))]

            def work(req):
                with mutex:
                    prior_debit = debits.get(req['file'], 0)
                    if total_debit + req['reserve_usd'] > plan['cap_usd']:
                        return None  # drain in-flight successes; never lose their manifests
                    log(req, prior_debit + req['reserve_usd'], 'reserved')
                try:
                    if req['backend'] == 'gemini':
                        record, cost, usage = gemini_word(req, out_dir)
                    else:
                        record = synthesize_one(client, req['word'], req['voice'], req['language_code'],
                                                config, out_dir, lang, file=req['file'],
                                                source='tts_word', attempts=1)
                        cost, usage = req['reserve_usd'], {}
                    with mutex:
                        log(req, prior_debit + cost, 'received' if record else 'rejected',
                            usage=usage, actual_cost_usd=cost if record or usage else None)
                    if record:
                        record.update(word_plan_sha256=plan_hash, word_frequency=req['frequency'],
                                      tts_backend=req['backend'], tts_model=req['model'])
                    return record
                except Exception as error:
                    with mutex:
                        log(req, prior_debit + req['reserve_usd'], 'failed',
                            error=type(error).__name__, detail=str(error))
                    print(f"{req['file']}: {type(error).__name__}: {error}; reservation retained", flush=True)
                    return None

            # Verify a new backend/locale on its first three already-planned
            # requests before dispatching thousands. These are not extra probes.
            pilot = todo[:3]
            written = run_pool(pilot, work, out_dir / 'manifest.jsonl', lang, min(workers, 3), rps)
            if written != len(pilot):
                print(f'{lang}: pilot failed; leaving the remaining requests unspent')
                continue
            run_pool(todo[3:], work, out_dir / 'manifest.jsonl', lang, workers, rps)
        print(f'Conservative synthesis debit including ambiguous failures: ${sum(debits.values()):.6f}')


def install(plan_path, staging, output):
    """Append completed staged clips; never mutate a running audit's corpus."""
    import fcntl
    import os
    import shutil

    assert_no_running_audit()
    assert staging.resolve() != output.resolve(), 'Install requires a separate staging tree'
    plan = json.loads(plan_path.read_text())
    plan_hash = fingerprint(plan_path)['sha256']
    excluded, nonspace, evals = eval_inventory()
    assert evals == plan['provenance']['eval_manifests'], 'Eval inventory changed'
    requests = {r['file']: r for r in plan['requests']}
    installed = Counter()
    with plan_path.with_suffix('.ledger.jsonl').open() as ledger:
        fcntl.flock(ledger, fcntl.LOCK_EX | fcntl.LOCK_NB)
        events = [json.loads(line) for line in ledger]
        assert all(e['plan_sha256'] == plan_hash for e in events)
        latest = {e['file']: e for e in events}
        for manifest in sorted(staging.glob('*/manifest.jsonl')):
            lang = manifest.parent.name
            directory = output / lang
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / 'manifest.jsonl'
            raw = target.read_bytes() if target.exists() else b''
            assert not raw or raw.endswith(b'\n'), f'{target}: missing final newline'
            existing = {r['file']: r for r in map(json.loads, raw.splitlines())}
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            assert len({r['file'] for r in rows}) == len(rows), 'Duplicate staged rows'
            with target.open('ab') as dest:
                for row in rows:
                    req = requests[row['file']]
                    assert row['lang'] == req['lang'] == lang
                    assert row['source'] == 'tts_word' and row['sentence'] == req['word']
                    assert row['word_plan_sha256'] == plan_hash
                    assert latest[row['file']]['state'] == 'received'
                    assert eligible(req['word'], excluded, nonspace)
                    source_audio = manifest.parent / row['file']
                    target_audio = directory / row['file']
                    if target_audio.exists():
                        assert fingerprint(source_audio)['sha256'] == fingerprint(target_audio)['sha256']
                    else:
                        assert row['file'] not in existing, 'Existing manifest has missing audio'
                        shutil.copyfile(source_audio, target_audio)
                    if row['file'] in existing:
                        assert existing[row['file']] == row, 'Refusing to replace an existing row'
                        continue
                    dest.write((json.dumps(row, ensure_ascii=False) + '\n').encode())
                    installed[lang] += 1
                dest.flush()
                os.fsync(dest.fileno())
    print(json.dumps({'installed': dict(installed)}))


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('candidates')
    c.add_argument('output', type=Path)
    c.add_argument('--limit', type=int, default=6000)
    p = sub.add_parser('freeze')
    p.add_argument('phones', type=Path)
    p.add_argument('provenance', type=Path)
    p.add_argument('output', type=Path)
    i = sub.add_parser('install')
    i.add_argument('plan', type=Path)
    i.add_argument('--staging', type=Path, required=True)
    i.add_argument('--output', type=Path, default=ROOT / 'data/audio')
    args = parser.parse_args()
    if args.command == 'candidates':
        candidates(args.output, args.limit)
    elif args.command == 'install':
        install(args.plan, args.staging, args.output)
    else:
        freeze(args.phones, args.provenance, args.output)


if __name__ == '__main__':
    main()
