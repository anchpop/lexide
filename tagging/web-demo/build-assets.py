#!/usr/bin/env python3
"""Publish immutable browser assets and point demo HTML at matching versions.

Run after editing www source files. Keep old assets on gh-pages so cached HTML
can still load the exact modules and styles it was built with.
"""
from hashlib import sha256
from pathlib import Path
import re

ASSETS = {
    'theme.js', 'demo.css', 'pronunciation.css', 'pronunciation.js',
    'pronunciation-decoder.mjs', 'audio-explorer.js', 'spectrogram-worker.js',
    'spectrogram.mjs',
}
REFERENCE = re.compile(r'''(["'])(\./(?:assets/)?[a-zA-Z0-9_.-]+\.(?:js|mjs|css))\1''')


def build(root):
    output = root / 'assets'
    output.mkdir(exist_ok=True)
    emitted = {}
    visiting = set()

    def original(path):
        name = Path(path).name
        return re.sub(r'\.[0-9a-f]{16}(?=\.(?:js|mjs|css)$)', '', name)

    def emit(name):
        if name in emitted:
            return emitted[name]
        if name in visiting:
            raise ValueError(f'Circular asset import: {name}')
        visiting.add(name)

        def dependency(match):
            target = original(match[2])
            if target not in ASSETS:
                return match[0]
            return f'{match[1]}./{emit(target)}{match[1]}'

        content = REFERENCE.sub(dependency, (root / name).read_text())
        digest = sha256(content.encode()).hexdigest()[:16]
        filename = f'{Path(name).stem}.{digest}{Path(name).suffix}'
        (output / filename).write_text(content)
        emitted[name] = filename
        visiting.remove(name)
        return filename

    for name in sorted(ASSETS):
        emit(name)

    for page in ('index.html', 'pronunciation.html'):
        def reference(match):
            name = original(match[2])
            return f'{match[1]}./assets/{emitted[name]}{match[1]}' if name in emitted else match[0]
        path = root / page
        path.write_text(REFERENCE.sub(reference, path.read_text()))
    return emitted


if __name__ == '__main__':
    result = build(Path(__file__).resolve().parent / 'www')
    print(f'Built {len(result)} immutable assets; updated demo HTML.')
