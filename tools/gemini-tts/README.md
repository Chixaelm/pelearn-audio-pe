# Gemini TTS narration build

Regenerates this repo's narration with Gemini TTS from the published reader page.

```sh
# 1. Export narration units (text + block hashes) from the reader build
NODE_PATH=$(npm root -g) node tools/gemini-tts/export.cjs ../pelearn-reader/index.html /tmp/units.json
# 2. Synthesise every block (resumable; cached per block)
GEMINI_API_KEY=... python3 tools/gemini-tts/gemini_tts.py synth --units /tmp/units.json --cache ~/.gemini-tts-cache --voice <Voice>
# 3. Build .m4a files and manifest.json into this repo
python3 tools/gemini-tts/gemini_tts.py assemble --units /tmp/units.json --cache ~/.gemini-tts-cache --voice <Voice> --label "<Label>" --out .
```

Narration only covers the blocks the reader marks as readable, so diagrams and other figures do not affect it.
