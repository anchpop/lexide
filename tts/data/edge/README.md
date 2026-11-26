# English edge-case sentence sets

Source text for the English TTS edge-case recordings (Nov 2026; see `scripts/edge_grams.py` and `../../README.md`).

| file | what | origin and license |
|---|---|---|
| heteronyms.jsonl, heteronym_readings.json | heteronyms in context, with each occurrence's intended phones | written for this project |
| punctuation.jsonl | quotes, dashes, lists, long and multi-sentence lines | written for this project |
| intonation.jsonl | questions, focus, emotion, assistant and demo text, verse | written for this project |
| rare.jsonl | rare words, names, numbers, acronyms (`spoken`: the line as read) | written for this project |
| books.jsonl | sentences and short passages from 81 public-domain books | Project Gutenberg texts, public domain in the US; `source` is the title |
| nonfiction.jsonl | expository sentences and passages | 482 lines from Wikipedia and 23 from Wikivoyage, under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (`source` names the article; text lightly edited: American spelling, "%" as "percent"); 38 from VOA Learning English, 40 from NOAA and 30 from NASA, US-government works in the public domain |

This README is the attribution for the CC BY-SA lines; anything derived from them keeps that license.
