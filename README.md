# Crop Doctor

Photograph a leaf, pick the crop, get a diagnosis you can act on today — or an honest "I'm not sure" when the model doesn't have enough to go on. Cassava, rice, maize, and tomato.

![Empty state](docs/screenshots/01-empty-state.png)

## Why this exists

Most plant disease classifiers you'll find online report accuracy in the high 90s and call it done. Almost all of them are trained on PlantVillage — a dataset of single leaves photographed on a grey card in a lab. I started this project the same way everyone does, got a model to 97.9% on that kind of data, and then checked what happened when I gave it a photograph from an actual field.

It fell to 33%.

That gap is the reason this project exists. Not "can a CNN classify a leaf" — that part's been solved for years — but "does the number on the slide survive contact with a real photograph," which turns out to be a much harder and much less asked question. Everything below follows from taking that question seriously: which datasets to trust, how to know when a model is cheating, and what to do when it admits it doesn't know.

## What it does

You pick a crop, upload a photo, and the app returns four things:

- **A diagnosis and an urgency level.** Not just a class name — whether this needs attention today, this week, or can be monitored.
- **A confidence-based refusal.** Below a threshold calibrated per crop, it says so instead of guessing. A wrong "this is fine" costs someone their crop; an honest "not sure, ask an expert" costs a phone call.
- **A heatmap of what it looked at.** So the answer is checkable rather than a black box. This single feature caught two of my own models cheating — more on that below.
- **Guidance keyed to what the condition actually is**, not just its name — a fungicide does nothing for an insect, and the app knows the difference.

There's also a local history log and a review queue for the cases it declines to call, so a referred photo doesn't just disappear into nothing.

## The four models, honestly

| Crop | Classes | Macro-F1 | Refers below | Background dependence |
|---|---|---|---|---|
| Cassava | 4 | 0.954 | 80% confidence | 6.1 pts |
| Rice | 10 | 0.888 | 70% confidence | not measured |
| Maize | 6 | 0.896 | 80% confidence | 4.6 pts |
| Tomato | 5 | 0.802 | 70% confidence | 14.5 pts |

Tomato is the weakest, and I'm not going to pretend otherwise. Verticillium wilt and leaf blight genuinely look alike on a single leaf — an agronomist tells them apart partly by which leaves show symptoms first, which one photograph can't show. A lower number that's honest is worth more than a higher one that isn't measuring what it claims to.

"Background dependence" is explained properly in the next section — it's arguably the most useful number on this table.

## The finding that shaped everything else

Early on I trained a tomato classifier on PlantVillage the way most tutorials do it, and it scored 0.9788 on its own held-out test set. Good enough to stop there, if I hadn't gotten curious about *why* it was that good.

So I ran a test: took the same test images, kept every leaf pixel exactly where it was, and replaced only the background — the grey studio card behind the leaf — with random noise.

Accuracy dropped by 29.8 points. The model wasn't reading the plant nearly as much as I thought. It was reading the photo shoot.

That single result changed the whole trajectory of the project. It's why none of the four shipped models are trained on PlantVillage. It's why I spent a lot more time than planned hunting down datasets that were actually photographed in fields — Paddy Doctor from Tamil Nadu, a Ugandan farmer-photographed cassava collection, CCMT from Ghana — rather than the convenient lab ones everyone uses. And it's why I built the same background-swap test into every model afterward, as a standing check rather than a one-off:

| Model | Dataset type | Background dependence |
|---|---|---|
| PlantVillage tomato (rejected) | Studio, grey card | 29.8 pts |
| A curated maize collection (also rejected) | Studio, mixed sources | 32.4 pts |
| CCMT tomato (shipped) | Field-captured | 14.5 pts |
| Cassava (shipped) | Field-captured | 6.1 pts |
| CCMT maize (shipped) | Field-captured | 4.6 pts |

The pattern holds across five separate models on three different crops: studio data clusters around 30 points of background dependence, field data clusters under 15. That's not a coincidence, and once I saw it clearly I stopped trusting any accuracy number that hadn't been checked this way.

Grad-CAM caught the same thing visually, independently. On the PlantVillage model, heat sat on the grey card behind correctly-classified leaves as often as it sat on the leaf. On a separate maize model that scored 0.986, it was reading the black corners left over from a rotation-augmentation step in the dataset. Both models would have looked completely fine from their accuracy alone.

## Other things that turned out to matter

**A resolution mismatch nearly became a fifth kind of shortcut.** One maize dataset had four disease classes at a consistent 256×256 and a fifth class — insect damage — at 450×600. A model can tell those apart by image dimensions alone without ever looking at the plant. I dropped the class rather than ship a number that measured file metadata.

**"Expert-validated" doesn't mean clean.** The CCMT collection I ended up using for maize and tomato had corrupted files that failed to open, others that opened but broke on the actual pixel data, and others still that a decoder happily read that TensorFlow's JPEG decoder flatly rejected — three different failure modes, caught by three different checks, none of which showed up until I went looking.

**Confidence calibration lives or dies on the data, not the code.** The abstain threshold works because the gap between "confident and right" and "confident and wrong" is real — cassava separates by 0.29, rice by wide margins too. On the PlantVillage model, that separation was only 0.06, which meant no threshold could have saved it. You can't calibrate your way out of a model that fundamentally doesn't know what it's looking at.

## Running it

```
git clone https://github.com/kaushiksridhar17/crop-doctor.git
cd crop-doctor
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
python src/app.py
```

The download script pulls the four trained models (~500 MB total) from this repo's GitHub Release — they're not tracked in git because GitHub caps individual files at 100 MB.

## What it doesn't do

**Each crop's model only knows its own list.** Point the rice model at a leaf with sheath blight — a real rice disease not in its training data — and it won't say "unknown." It'll confidently pick whichever of its ten known conditions looks closest. The abstain threshold catches some of this when confidence is genuinely low, but not all of it.

**The photo quality gate uses color, not object recognition.** It checks for blur, exposure, and whether enough of the frame looks like plant material based on hue and saturation. I fed it a photo of a pair of dice once during testing and it passed — the warm tones and lighting happened to satisfy the "looks like a plant" heuristic. It's good at catching genuinely bad photos; it is not a general-purpose "is this actually a leaf" detector.

**The treatment advice has not been reviewed by a plant pathologist.** I wrote it from general agronomic sources — ICAR and TNAU-style guidance — for a project, not a clinic. No specific product names appear anywhere, because approved chemicals vary by country and a wrong recommendation there does real damage. Every entry defers to a local extension officer for anything more specific than "this is fungal, act this week."

**Rice has no background-dependence number.** I built that diagnostic partway through the project, after training rice, and never went back to run it. The other three models all have it; rice doesn't. That's a gap, not a deliberate omission.

## What's under the hood

```
src/
    registry.py    every crop's model path, thresholds, and metadata in one place
    quality.py     rejects unusable photos before they reach a model
    predict.py     inference and the abstain logic
    gradcam.py     attention maps — the thing that caught the cheating models
    advice.py      treatment guidance, validated against what the models can output
    storage.py     local history and the review queue
    app.py         the interface
data/
    advice.json    the guidance content, separate from code so it can be reviewed on its own
results/           the actual numbers behind every claim above — test metrics, calibration
                   sweeps, background diagnostics, one folder per crop
notebooks/         the four training notebooks, as documentation more than as a run-it-yourself path
docs/screenshots/  the app in use
```

Every model is EfficientNetV2-S, fine-tuned in two phases — head first with the backbone frozen, then the top of the backbone unfrozen at a much lower learning rate. Splits are deduplicated by both exact and perceptual hash before training, because more than one of these datasets turned out to contain near-duplicate images that would otherwise leak across train and test.

## A few screenshots

**A confident, urgent result:**

![Act now result](docs/screenshots/02a-act-now.png)
![Grad-CAM heatmap](docs/screenshots/02b-act-now.png)

**Something less urgent:**

![Monitor result](docs/screenshots/03a-monitor.png)
![Monitor result, continued](docs/screenshots/03b-monitor.png)

**The model declining to guess:**

![Refer to expert](docs/screenshots/04-refer-to-expert.png)

**A photo it correctly won't touch:**

![Rejected photo](docs/screenshots/05-rejected-photo.png)

**History and the review queue:**

![History tab](docs/screenshots/06-history.png)
![Review queue](docs/screenshots/07-review-queue.png)

## If I kept going

The obvious next step is Docker, so this runs the same way regardless of what's already on someone's machine. After that, TFLite conversion for an on-device version — I tested this on the tomato model and got it to 13.5 MB with no measurable accuracy loss, versus a complete collapse when I tried int8 quantization, which is its own small lesson in not assuming smaller is free. And rice needs its background-dependence number run, since it's the one gap in an otherwise complete set of diagnostics.
