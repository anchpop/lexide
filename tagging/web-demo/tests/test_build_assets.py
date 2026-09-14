import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

DEMO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_assets', DEMO / 'build-assets.py')
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class AssetBuildTests(unittest.TestCase):
    def test_stable_build_and_transitive_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in assets.ASSETS | {'index.html', 'pronunciation.html'}:
                shutil.copyfile(DEMO / 'www' / name, root / name)
            first = assets.build(root)
            self.assertEqual(first, assets.build(root))
            with (root / 'spectrogram.mjs').open('a') as source:
                source.write('\n// Updated spectrum implementation\n')
            updated = assets.build(root)
            for name in ('spectrogram.mjs', 'spectrogram-worker.js', 'audio-explorer.js', 'pronunciation.js'):
                self.assertNotEqual(first[name], updated[name])
                self.assertTrue((root / 'assets' / first[name]).exists())
            self.assertEqual(first['theme.js'], updated['theme.js'])
            self.assertIn(updated['pronunciation.js'], (root / 'pronunciation.html').read_text())
            self.assertIn(updated['spectrogram-worker.js'], (root / 'assets' / updated['audio-explorer.js']).read_text())
            decoder = (root / "assets" / updated["pronunciation-decoder.mjs"]).read_text()
            self.assertIn('import("../pkg/parsley_web_demo.js")', decoder)
            self.assertNotIn("node-pkg", decoder)


if __name__ == '__main__':
    unittest.main()
