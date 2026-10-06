#!/usr/bin/env python3
"""Import an authorized Mozilla Data Collective Common Voice scripted archive.

This does NOT bypass MDC login/terms or obtain signed download URLs. Download
through your own MDC account after accepting the dataset's terms, then supply
--archive and --dataset-url. Unknown nonempty accents are conservatively dropped.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
import csv
import json
from pathlib import Path
import random
import re
import shutil

import soundfile as sf
from corpus_import import extract_archive, importer, append_audio, summarize, atomic_json

LOCALES = dict(zip('ara ces dan deu eng fas fra hin ita jpn kor por rus spa tha zho-hans'.split(),
                  'ar cs da de en fa fr hi it ja ko pt ru es th zh-CN'.split()))
# Native regional descriptions from CV's free-text/locale accent metadata. Do not
# infer native status from gender/age or silently admit unrecognized descriptions.
REGIONS = {
    'ara':'arabic|egyptian|egypt|saudi|saudi arabian|levantine|iraqi|syrian|jordanian|lebanese|moroccan|algerian|tunisian|gulf|yemeni|sudanese|العربية',
    'ces':'czech|czech republic|bohemian|moravian|čeština',
    'dan':'danish|denmark|dansk|jysk|sjællandsk',
    'deu':'german|germany|deutsch|deutschland|austrian|austria|österreich|schweiz|swiss german|bavarian|bayerisch|sächsisch|berlinerisch|norddeutsch|hochdeutsch',
    'eng':'united states english|united states|united kingdom|england english|england|british english|british|american english|american|canadian english|canadian|australian english|australian|new zealand english|new zealand|scottish english|scottish|irish english|irish|welsh english|welsh|south african english|us|uk',
    'fas':'persian|farsi|iranian|iran|tehrani|tehran|فارسی',
    'fra':'french|france|français|français de france|belgian french|belgium|belgique|québécois|québec|quebec|canadian french|canada|swiss french|suisse',
    'hin':'hindi|india|delhi|uttar pradesh|madhya pradesh|rajasthan|bihar|हिन्दी|हिंदी',
    'ita':'italian|italia|italy|siciliano|toscano|lombardo|veneto|romano|sardo|napoletano',
    'jpn':'japanese|japan|tokyo|kansai|osaka|日本|標準語|関西弁',
    'kor':'korean|korea|seoul|gyeongsang|한국|표준어|서울',
    'por':'portuguese|portugal|european portuguese|brazilian portuguese|brazil|brasil|lisboa|paulista|carioca|mineiro|gaúcho',
    'rus':'russian|russia|moscow|русский|московский|петербургский',
    'spa':'spanish|spain|españa|español|castellano|mexico|méxico|mexican|argentina|colombia|chile|peru|perú|venezuela|uruguay|ecuador|bolivia|cuba|costa rica|puerto rico|dominican|andaluz|canario',
    'tha':'thai|thailand|bangkok|ไทย|กลาง',
    'zho-hans':'mandarin|china|beijing|putonghua|普通话|普通話|北京|北方|南方|中国|中國|台湾|台灣|新加坡|马来西亚',
}


# g2p labels Spanish and Portuguese per variety, and CV's accent field is the
# only evidence of it. A bare "spanish" or "portuguese" names no variety.
VARIETIES = {
    # Canarian Spanish is seseante like the Americas; Andalusian mixes seseo,
    # ceceo and distinción, so it names no labelable variety.
    'spa': {'european': 'spain|españa|castellano',
            'latin_american': 'canario|mexico|méxico|mexican|argentina|colombia|chile|peru|perú|venezuela|uruguay|ecuador|bolivia|cuba|costa rica|puerto rico|dominican'},
    'por': {'european': 'portugal|european portuguese|lisboa',
            'brazilian': 'brazilian portuguese|brazil|brasil|paulista|carioca|mineiro|gaúcho'},
}


def variety(lang, accent):
    accent = accent.strip().lower()
    return next((name for name, pattern in VARIETIES[lang].items()
                 if re.fullmatch(rf'(?:{pattern})(?: accent)?', accent)), None)


def accent_status(lang, accent):
    accent = accent.strip().lower()
    if not accent:
        return 'blank'
    if re.search(r'non[ -]?native|second language|foreign|learner|非母语|非母語|nicht.*mutter|non.*natif', accent):
        return 'nonnative'
    if re.fullmatch(rf'(?:{REGIONS[lang]})(?: accent)?', accent):
        return 'native_region'
    return 'unknown'


def candidates(path, lang):
    coverage = Counter()
    by_speaker = defaultdict(list)
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            coverage['validated_rows'] += 1
            for field in ('client_id','accents','accent','age','gender','locale'):
                if row.get(field, '').strip():
                    coverage[field+'_present'] += 1
            accent = row.get('accents') or row.get('accent') or ''
            status = accent_status(lang, accent)
            coverage['accent_'+status] += 1
            if int(row['up_votes']) < 2 or int(row['down_votes']) != 0:
                coverage['rejected_votes'] += 1
                continue
            if status not in {'blank','native_region'}:
                continue
            if not row.get('client_id') or not row.get('sentence', '').strip():
                coverage['rejected_missing_identity_or_text'] += 1
                continue
            if row.get('locale') and row['locale'] != LOCALES[lang]:
                coverage['rejected_locale'] += 1
                continue
            by_speaker[row['client_id']].append(row)
    rng = random.Random(42)
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    queues = []
    for speaker in speakers:
        rows = sorted(by_speaker[speaker], key=lambda r:r['path'])
        rng.shuffle(rows)
        queues.append(deque(rows))
    ordered = []
    active = deque(queues)
    while active:
        queue = active.popleft()
        ordered.append(queue.popleft())
        if queue:
            active.append(queue)
    return ordered, coverage


def acquire(args):
    if not args.dataset_url.startswith('https://mozilladatacollective.com/'):
        raise ValueError('--dataset-url must identify the authorized MDC dataset')
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.cache_dir).free < max(50*1024**3, 4*args.archive.stat().st_size):
        raise RuntimeError('Insufficient free space to extract authorized CV archive safely')
    extracted = args.cache_dir / ('cv-'+args.lang)
    extract_archive(args.archive, extracted)
    validated, = extracted.rglob('validated.tsv')
    eligible, coverage = candidates(validated, args.lang)
    with importer(args.data_dir, args.lang, 'cv', args.cache_dir) as (root, rows, stream):
        existing = {r['file'] for r in rows}
        used = defaultdict(float)
        for row in rows:
            if row.get('source') == 'cv':
                used[row['voice']] += sf.info(root/row['file']).duration
        total = sum(used.values())
        for rec in eligible:
            if total >= args.hours*3600:
                break
            original = Path(rec['path'])
            if original.name != rec['path']:
                raise ValueError(f'Unsafe clip filename: {rec["path"]}')
            name = 'cv_'+original.stem+'.wav'
            if name in existing:
                continue
            voice = 'cv:'+rec['client_id']
            path = validated.parent/'clips'/original
            duration = sf.info(path).duration
            if not 0 < duration <= 16 or total+duration > args.hours*3600 or used[voice]+duration > 900:
                coverage['rejected_duration_or_budget'] += 1
                continue
            row = {'file':name,'sentence':rec['sentence'],'source':'cv','voice':voice,
                   'license':args.license,'dataset_url':args.dataset_url,'locale':LOCALES[args.lang],
                   'client_id':rec['client_id'],'up_votes':int(rec['up_votes']), 'down_votes':int(rec['down_votes']),
                   'accent':rec.get('accents') or rec.get('accent') or '', 'original_audio':rec['path']}
            if args.lang in VARIETIES:
                row['variety'] = variety(args.lang, row['accent'])
                if row['variety'] is None:
                    coverage['rejected_unknown_variety'] += 1
                    continue
            if append_audio(root, stream, row, path, max_seconds=min(16, args.hours*3600-total, 900-used[voice])):
                existing.add(name)
                total += row['duration_sec']
                used[voice] += row['duration_sec']
    report = summarize(args.data_dir,args.lang,'cv')
    report['metadata_coverage'] = dict(coverage)
    report['target_hours'] = args.hours
    atomic_json(args.cache_dir/f'cv-{args.lang}-summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--dataset-url',required=True)
    parser.add_argument('--license',required=True,help='License from the accepted dataset datasheet')
    parser.add_argument('--lang',choices=LOCALES,required=True)
    parser.add_argument('--hours',type=float,default=15)
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).parent/'audio')
    parser.add_argument('--cache-dir',type=Path,required=True)
    args = parser.parse_args()
    if not 0 < args.hours <= 15:
        parser.error('--hours must be in (0, 15]')
    acquire(args)


if __name__ == '__main__':
    main()
