# Test Assets — Images

These images are **not committed to the repository** due to file size.
All images are **public domain** (NASA or pre-1928).

Run the fetch script to download them:

```bash
bash scripts/fetch_test_images.sh
```

If automated download fails (some hosts block scripts), download manually
from the URLs listed below and place in `test_assets/images/`.

---

## Image Catalog

### `earthrise.jpg` — Color / spatial orientation test
**Subject:** Earth rising over the lunar surface
**Credit:** NASA, Apollo 8, William Anders, December 24 1968
**Source:** https://upload.wikimedia.org/wikipedia/commons/a/a8/NASA-Apollo8-Dec24-Earthrise.jpg
**License:** Public domain (NASA)
**Why:** Tests color accuracy (Earth's blue/white, gray lunar surface, black space)
and spatial orientation. Models should identify both bodies and describe the orientation.

### `pale_blue_dot.png` — Hallucination resistance test
**Subject:** Earth photographed from 3.7 billion miles (Voyager 1)
**Credit:** NASA / JPL-Caltech, February 14 1990
**Source:** https://upload.wikimedia.org/wikipedia/commons/7/73/Pale_Blue_Dot.png
**License:** Public domain (NASA)
**Why:** Deliberately minimal visual content. Tests hallucination resistance.
A model that confidently describes rich scene detail is confabulating.
Correct response: acknowledges minimal content, identifies the pale speck.

### `lunch_skyscraper.jpg` — Scene / counting test
**Subject:** 11 ironworkers eating lunch on a steel beam, 840 feet above Manhattan
**Credit:** Unknown photographer, September 29 1932, Rockefeller Center
**Source:** https://upload.wikimedia.org/wikipedia/commons/e/e7/Lunch_atop_a_Skyscraper.jpg
**License:** Public domain (pre-1928 publication threshold)
**Why:** Tests spatial understanding, counting (~11 men), scene context,
and depth perception. Models should count the workers and describe the skyline below.

### `hubble_deep_field.jpg` — Cosmic scale / galaxy recognition
**Subject:** Hubble Deep Field — nearly 3,000 galaxies in a tiny patch of sky
**Credit:** R. Williams (STScI), the HDF Team, NASA, 1996
**Source:** https://upload.wikimedia.org/wikipedia/commons/5/5f/HubbleDeepField.800px.jpg
**License:** Public domain (NASA)
**Why:** Tests whether the model can identify astronomical content and
understand cosmic scale. Also a secondary hallucination test — models
shouldn't describe it as a cityscape or terrestrial scene.

### `blue_marble.jpg` — Whole-Earth recognition
**Subject:** Earth photographed from Apollo 17 ("The Blue Marble")
**Credit:** NASA, Apollo 17 crew, December 7 1972
**Source:** https://upload.wikimedia.org/wikipedia/commons/9/97/The_Earth_seen_from_Apollo_17.jpg
**License:** Public domain (NASA)
**Why:** Tests whole-planet recognition, continent identification (Africa clearly
visible), cloud pattern description, and ocean color.

### `migrant_mother.jpg` — Portrait / emotional context test
**Subject:** Florence Owens Thompson with children, pea-pickers camp, Nipomo CA
**Credit:** Dorothea Lange / Farm Security Administration / US Government, March 1936
**Source:** `File:Lange-MigrantMother02.jpg` on Wikimedia Commons
**License:** Public domain (US Government work)
**Why:** Iconic portrait — tests emotion reading, context understanding (Depression-era hardship),
and multi-subject composition. Model should describe the woman's expression and the children.

### `great_wave.jpg` — Art recognition / composition test
**Subject:** The Great Wave off Kanagawa
**Credit:** Katsushika Hokusai, c.1831
**Source:** `File:The_Great_Wave_off_Kanagawa.jpg` on Wikimedia Commons
**License:** Public domain (pre-1928)
**Why:** One of the most recognized artworks ever made. Tests art recognition, color description
(blue/white waves, cream background), and identification of Mt. Fuji in the background.

### `girl_pearl_earring.jpg` — Portrait detail test (non-NASA)
**Subject:** Girl with a Pearl Earring
**Credit:** Johannes Vermeer, c.1665, Mauritshuis, The Hague
**Source:** `File:Girl_with_a_Pearl_Earring.jpg` on Wikimedia Commons
**License:** Public domain (pre-1928)
**Why:** Iconic portrait — tests fine color accuracy (blue/yellow headscarf, pearl earring,
dark background) and facial detail description. Ground truth is extremely well documented.

### `pillars_of_creation.jpg` — Nebula / cosmic structure test
**Subject:** Pillars of Creation, Eagle Nebula (M16)
**Credit:** NASA / ESA / Hubble Heritage Team, 2014 (WFC3/UVIS)
**Source:** `File:Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg`
**License:** Public domain (NASA/ESA)
**Why:** Dramatic nebula with distinct column structures. Tests ability to describe
gas/dust formations, color (green/blue/red), and star-forming context. Differs from
Hubble Deep Field — structure rather than point sources.

---

## Standard Prompts

See `test_assets/prompts/standard.txt` for the standard prompt set used in
`docs/PERFORMANCE.md` benchmarks.

---

## Usage with llm-runner

```bash
LLM_RUNNER=~/git/apple/coreai-models/.build/out/Products/Debug/llm-runner
BUNDLE=exports/fastvlm-0.5b.vlmasset
IMG=test_assets/images

# Color / spatial
$LLM_RUNNER --model $BUNDLE --image $IMG/earthrise.jpg \
  --prompt "What do you see in this image? Describe the colors and spatial arrangement." \
  --max-tokens 300 --temperature 0

# Hallucination resistance
$LLM_RUNNER --model $BUNDLE --image $IMG/pale_blue_dot.png \
  --prompt "Describe exactly what you see in this image." \
  --max-tokens 200 --temperature 0

# Counting
$LLM_RUNNER --model $BUNDLE --image $IMG/lunch_skyscraper.jpg \
  --prompt "How many people are in this image? What are they doing and where?" \
  --max-tokens 300 --temperature 0

# Cosmic scale
$LLM_RUNNER --model $BUNDLE --image $IMG/hubble_deep_field.jpg \
  --prompt "Describe what you see in this image." \
  --max-tokens 200 --temperature 0

# Whole Earth
$LLM_RUNNER --model $BUNDLE --image $IMG/blue_marble.jpg \
  --prompt "What is shown in this image? Identify any continents or geographic features visible." \
  --max-tokens 300 --temperature 0
```
