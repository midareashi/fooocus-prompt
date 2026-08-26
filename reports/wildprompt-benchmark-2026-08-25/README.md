# Luna checkpoint and wildprompt benchmark — 2026-08-25

## Bottom line

**CyberRealistic XL v100 is the best all-around choice.** It placed second for Luna facial likeness, essentially tied for first on upskirt adherence, second by a tiny margin on feet adherence, and had the lowest prompt-failure rate of the four checkpoints.

The specialized winners were:

- **Best average Luna face match:** `lifeIsGood_v1` — but its advantage over CyberRealistic and Juggernaut was small, and it followed the shot prompts very poorly.
- **Best individual Luna face matches:** `juggernautXL_ragnarok` produced the top three individual similarity scores, despite placing third on average.
- **Highest literal prompt-adherence score:** `lustifyNSFWCheckpoint_zenithV9`, narrowly ahead of CyberRealistic. It was much worse at preserving Luna's face.
- **Best dependable compromise:** `cyberrealisticXL_v100`.

Your original hunch was directionally right: CyberRealistic and Juggernaut are both good practical Luna models. The surprise was that LifeIsGood won the average face-only metric, while Lustify won the narrow prompt-only metric.

## Test scope and fairness

The cutoff was local time **2026-08-25 17:30**. The matching files actually ran from 19:01:49 through 19:26:29.

| Property | Value |
|---|---:|
| Images reviewed | 192 |
| Checkpoints | 4 |
| Images per checkpoint | 48 |
| Feet images | 96 |
| Upskirt images | 96 |
| Unique shot prompts | 24 |
| Repeats per prompt/checkpoint | 2 |
| Unique seeds | 48, each repeated across all 4 checkpoints |
| Luna reference images | 9 |

Every image used 896×1152, Speed mode, 30 steps, `dpmpp_2m_sde_gpu`, Karras, CFG 3, sharpness 2, Fooocus V2 plus Fooocus Negative, identity strength 1, face weight 0, and face-weight start 0. The checkpoint was the controlled variable.

## Luna facial-likeness ranking

The primary score is cosine similarity between a detected generated face and the normalized centroid of the nine Luna reference-face embeddings. Faces were aligned with Fooocus's face cropper and embedded with Fooocus's PhotoMaker vision encoder. Higher is better; the score is comparative, not a percentage.

| Rank | Checkpoint | Detected | Mean | Median | SD | P10 | Interpretation |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `lifeIsGood_v1` | 48/48 | **0.6764** | **0.6815** | **0.0432** | **0.6228** | Highest and most consistent average face match |
| 2 | `cyberrealisticXL_v100` | 48/48 | 0.6663 | 0.6763 | 0.0589 | 0.5948 | Close second; strong consistency and much better prompt control |
| 3 | `juggernautXL_ragnarok` | 47/48 | 0.6570 | 0.6798 | 0.0782 | 0.5405 | Best peaks, but more variable; one deliberate foot close-up had no face |
| 4 | `lustifyNSFWCheckpoint_zenithV9` | 48/48 | 0.5367 | 0.5516 | 0.0911 | 0.3955 | Clear identity drift from the Luna references |

The top three should be treated as a close group rather than a definitive separation. In paired bootstrap comparisons, LifeIsGood minus CyberRealistic was +0.0101 with a 95% interval of -0.0075 to +0.0280; LifeIsGood minus Juggernaut was +0.0182 with an interval of -0.0042 to +0.0425. CyberRealistic exceeded Lustify by +0.1296 with an interval of +0.1043 to +0.1579, which is the only unambiguous face-likeness gap here.

### Best individual face matches

| Rank | Image | Checkpoint | Score |
|---:|---|---|---:|
| 1 | [19-08-31](../../outputs/2026-08-25/2026-08-25_19-08-31_2293.png) | Juggernaut | **0.7812** |
| 2 | [19-22-37](../../outputs/2026-08-25/2026-08-25_19-22-37_7011.png) | Juggernaut | 0.7723 |
| 3 | [19-22-44](../../outputs/2026-08-25/2026-08-25_19-22-44_1378.png) | Juggernaut | 0.7680 |
| 4 | [19-17-00](../../outputs/2026-08-25/2026-08-25_19-17-00_7211.png) | CyberRealistic | 0.7504 |
| 5 | [19-03-29](../../outputs/2026-08-25/2026-08-25_19-03-29_3460.png) | CyberRealistic | 0.7489 |

## Checkpoint prompt-adherence ranking

Each image was manually scored from 0 to 5 against its exact shot wildprompt. A 4 or 5 means the central pose/framing intent was clearly followed; 0 or 1 is a failure. General attractiveness and facial similarity were intentionally excluded from this score.

### Overall

| Rank | Checkpoint | Mean / 5 | Strong 4–5 | Failures 0–1 | Result |
|---:|---|---:|---:|---:|---|
| 1 | `lustifyNSFWCheckpoint_zenithV9` | **3.063** | **45.8%** | 10.4% | Highest literal hit rate, but face drift |
| 2 | `cyberrealisticXL_v100` | **3.042** | 37.5% | **4.2%** | Essentially tied for first and much more dependable |
| 3 | `juggernautXL_ragnarok` | 2.813 | 27.1% | 6.3% | Good, but less literal than CyberRealistic |
| 4 | `lifeIsGood_v1` | 1.479 | 8.3% | 62.5% | Frequently replaced the requested composition with a generic explicit portrait |

The 0.021 gap between Lustify and CyberRealistic is below the precision of a subjective five-point review. The practical distinction is that Lustify generated more strong hits, while CyberRealistic generated far fewer complete failures.

### By category

| Category | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|
| Upskirt | Lustify **2.667**, tied mean with Cyber but more 4–5 hits | CyberRealistic **2.667**, fewer failures | Juggernaut 2.417 | LifeIsGood 2.000 |
| Feet | Lustify **3.458** | CyberRealistic **3.417** | Juggernaut 3.208 | LifeIsGood 0.958 |

## Upskirt wildprompt ranking

| Rank | Wildprompt | Mean / 5 | Strong 4–5 | Contact sheet |
|---:|---|---:|---:|---|
| 1 | Staircase, one step above lens, looking back, lifted skirt | **3.625** | **62.5%** | [Review](contact_sheets/upskirt_11_staged-staircase-upskirt-style-fashion-shot-of-an-adult-woma.jpg) |
| 2 | Reclining, knees bent, skirt gathered at thighs | 3.000 | 25.0% | [Review](contact_sheets/upskirt_10_staged-reclining-upskirt-style-photograph-of-an-adult-woman-.jpg) |
| 3 | Twirling pleated skirt with upward camera angle | 2.875 | 37.5% | [Review](contact_sheets/upskirt_09_playful-upward-camera-angle-capturing-an-adult-woman-twirlin.jpg) |
| 4 | Holding both sides of skirt above low camera | 2.750 | 37.5% | [Review](contact_sheets/upskirt_12_teasing-low-angle-portrait-of-an-adult-woman-holding-both-si.jpg) |
| 5 | Lifting hem, symmetric under-skirt viewpoint | 2.625 | 12.5% | [Review](contact_sheets/upskirt_01_carefully-staged-under-skirt-viewpoint-with-an-adult-woman-l.jpg) |
| 6 | Dramatic full-body composition from below | 2.250 | 12.5% | [Review](contact_sheets/upskirt_06_dramatic-full-body-fashion-composition-from-below-an-adult-w.jpg) |
| 7 | Seated low-angle at chair edge | 2.250 | 12.5% | [Review](contact_sheets/upskirt_04_consensual-seated-low-angle-shot-of-an-adult-woman-near-the-.jpg) |
| 8 | Cinematic backlit composition above lens | 2.125 | 12.5% | [Review](contact_sheets/upskirt_02_cinematic-consensual-upskirt-style-composition-of-an-adult-w.jpg) |
| 9 | Leaning toward lens while holding skirt edge | 2.000 | 25.0% | [Review](contact_sheets/upskirt_08_playful-floor-level-composition-of-an-adult-woman-leaning-to.jpg) |
| 10 | Close over-the-knee viewpoint | 2.000 | 12.5% | [Review](contact_sheets/upskirt_03_consensual-over-the-knee-viewpoint-of-an-adult-woman-deliber.jpg) |
| 11 | Generic staged low-angle above camera | 1.875 | 0.0% | [Review](contact_sheets/upskirt_05_consensual-staged-low-angle-fashion-photograph-of-an-adult-w.jpg) |
| 12 | Floor-level beneath plaid skirt | 1.875 | 0.0% | [Review](contact_sheets/upskirt_07_floor-level-photograph-beneath-a-short-plaid-skirt-with-an-a.jpg) |

The staircase prompt is the only strong upskirt prompt across checkpoints. It gives the model concrete geometry—stairs, vertical separation, rear three-quarter orientation, and a look-back gesture. Generic phrases such as “low angle” and “above the camera” were routinely ignored.

## Feet wildprompt ranking

| Rank | Wildprompt | Mean / 5 | Strong 4–5 | Contact sheet |
|---:|---|---:|---:|---|
| 1 | Feet centered against a textured rug | **3.250** | **75.0%** | [Review](contact_sheets/feet_23_top-down-fashion-composition-centered-on-her-bare-feet-again.jpg) |
| 2 | Symmetrical reclining, both soles nearest lens | 3.125 | 50.0% | [Review](contact_sheets/feet_22_symmetrical-reclining-composition-with-both-bare-soles-neare.jpg) |
| 3 | Macro toes against rumpled white linen | 3.000 | 62.5% | [Review](contact_sheets/feet_18_intimate-macro-detail-of-polished-toes-curling-gently-agains.jpg) |
| 4 | Warm backlight, full body, rising onto tiptoe | 3.000 | 50.0% | [Review](contact_sheets/feet_24_warm-backlit-full-body-silhouette-emphasizing-elegant-feet-a.jpg) |
| 5 | Sheer pantyhose over foreground toes | 3.000 | 37.5% | [Review](contact_sheets/feet_15_close-editorial-detail-of-both-feet-inside-ultra-sheer-panty.jpg) |
| 6 | One foot reaching toward camera | 2.875 | 62.5% | [Review](contact_sheets/feet_20_playful-forced-perspective-portrait-with-one-bare-foot-reach.jpg) |
| 7 | Stockinged feet together on velvet | 2.750 | 37.5% | [Review](contact_sheets/feet_14_close-beauty-shot-of-both-stockinged-feet-resting-together-o.jpg) |
| 8 | Full body, floor level, bare feet nearest lens | 2.625 | 25.0% | [Review](contact_sheets/feet_17_full-body-low-floor-level-composition-with-her-relaxed-bare-.jpg) |
| 9 | Reclining sideways, one foot fills foreground | 2.500 | 37.5% | [Review](contact_sheets/feet_21_reclining-sideways-with-one-leg-stretched-toward-the-camera-.jpg) |
| 10 | Side-profile close-up of one arched foot | 2.500 | 12.5% | [Review](contact_sheets/feet_16_elegant-side-profile-close-up-of-one-arched-bare-foot-with-p.jpg) |
| 11 | On stomach, crossed feet kicked up behind | 2.375 | 12.5% | [Review](contact_sheets/feet_19_lying-on-her-stomach-with-ankles-crossed-and-both-feet-kicke.jpg) |
| 12 | Floor-level barefoot walking step | 2.125 | 0.0% | [Review](contact_sheets/feet_13_cinematic-floor-level-photograph-timed-during-a-slow-barefoo.jpg) |

Concrete anchors again won: rug, bed linen, symmetry, and “soles nearest lens” survived checkpoint changes. The walking-step prompt tried to control timing, camera height, heel state, toe motion, and background balance simultaneously; every checkpoint simplified it into a generic standing or seated pose.

## Recommended use

1. Use **CyberRealistic** as the default for future Luna wildprompt batches.
2. Use **Juggernaut** when hunting for the very best individual Luna face and accept higher variance.
3. Use **LifeIsGood** only for face-focused portraits; it is a poor choice for composition-sensitive feet shots.
4. Use **Lustify** when literal fetish framing matters more than identity consistency.
5. Keep the staircase upskirt prompt and the rug, symmetrical-soles, linen macro, and pantyhose feet prompts.
6. Rewrite or retire the generic low-angle, plaid-floor, and walking-step prompts.

## Supporting data

- [Per-image automated scores](image_scores.csv)
- [Automated face summary](automated_summary.json)
- [Per-image manual adherence scores](manual_adherence_scores.csv)
- [Manual adherence summary](manual_adherence_summary.json)
- [Manual rating rationale](manual_prompt_ratings.json)
- [Contact-sheet manifest](contact_sheet_manifest.json)
- [Repeatable benchmark method](../WILDPROMPT_BENCHMARK_METHOD.md)

The complete visual review is in the [contact_sheets](contact_sheets/) directory.
