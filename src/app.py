"""Crop Doctor — the interface.

Designed as a plant clinic prescription form. Not decoration: CABI's Plantwise
clinics work this way in practice, where a grower brings a sample and leaves
with a written recommendation carrying a diagnosis, an urgency, and a list of
actions. That is exactly the shape of what this produces.

Urgency is the loudest thing on the page and the only place colour carries
meaning. Someone deciding whether to walk back to the shed for a sprayer needs
"act now" before they need a confidence score.

Every diagnosis is logged locally. Referred cases — where the model was not
confident enough to answer — go into a review queue, which is the only place
"Refer to an expert" actually leads.

Run from the project root:
    python src/app.py
"""

from __future__ import annotations

import datetime
import html
import os

import gradio as gr
from PIL import Image

import advice
import gradcam
import predict
import quality
import registry
import storage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT, "test_images")
REVIEW_IMAGES_DIR = os.path.join(ROOT, "data", "review_images")

CROP_CHOICES = [(c.name, c.key) for c in registry.CROPS.values()]

# The four levels, in order of escalation. The strip under the masthead shows
# them as a scale, which doubles as the page's identity and as a legend for the
# band that appears on every result.
URGENCY = {
    "none":          ("var(--u-none)",    "No action needed"),
    "monitor":       ("var(--u-monitor)", "Monitor"),
    "act this week": ("var(--u-week)",    "Act this week"),
    "act now":       ("var(--u-now)",     "Act now"),
    "seek advice":   ("var(--u-refer)",   "Refer to an expert"),
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bitter:wght@400;500;700;900&family=Public+Sans:wght@400;500;600;700&display=swap');

:root {
    --ink:        #14211A;
    --ink-2:      #4A5D51;
    --ink-3:      #7B8A81;
    --paper:      #E7EBE5;
    --paper-2:    #DDE2D9;
    --paper-3:    #D2D8CD;
    --rule:       #9FAAA2;
    --u-none:     #2F6B45;
    --u-monitor:  #8A6D1F;
    --u-week:     #A6541F;
    --u-now:      #8C2F26;
    --u-refer:    #445A6B;
}

html, body, gradio-app, .gradio-container, .app, .main, .wrap, .contain, .fillable {
    background: var(--paper) !important;
}
.gradio-container {
    max-width: 1240px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 0 26px 60px 26px !important;
    font-family: 'Public Sans', system-ui, sans-serif !important;
    color: var(--ink) !important;
}
.gradio-container *, .gradio-container label { color: var(--ink); }
footer, .built-with, .show-api { display: none !important; }

/* ---- masthead ---- */
#head { padding: 30px 0 0 0; }
#head .scale { display: grid; grid-template-columns: repeat(4, 1fr); height: 5px; margin-bottom: 22px; }
#head .scale i { display: block; }
#head h1 {
    font-family: 'Bitter', Georgia, serif;
    font-weight: 900; font-size: clamp(38px, 5.2vw, 58px);
    line-height: 0.98; letter-spacing: -0.025em; margin: 0 0 14px 0;
}
#head .lede { max-width: 60ch; font-size: 16.5px; line-height: 1.5; color: var(--ink-2); margin: 0 0 26px 0; }
#head .rule { border-top: 2px solid var(--ink); border-bottom: 1px solid var(--ink); height: 4px; margin-bottom: 8px; }

/* ---- tabs ---- */
.gradio-container .tab-nav { border-bottom: none !important; margin-top: 18px !important; }
.gradio-container .tab-nav button {
    font-family: 'Bitter', Georgia, serif !important; font-weight: 500 !important;
    font-size: 15px !important; border-radius: 0 !important;
    border: 1px solid var(--rule) !important; border-bottom: none !important;
    background: var(--paper-2) !important; color: var(--ink-2) !important;
}
.gradio-container .tab-nav button.selected {
    background: var(--paper) !important; color: var(--ink) !important;
    border-color: var(--ink) !important; font-weight: 700 !important;
}

/* ---- form steps ---- */
.step {
    display: flex; align-items: baseline; gap: 11px;
    margin: 0 0 12px 0; padding-bottom: 7px; border-bottom: 1px solid var(--rule);
}
.step b {
    font-family: 'Bitter', Georgia, serif; font-weight: 700; font-size: 13px;
    color: var(--paper); background: var(--ink); width: 21px; height: 21px;
    display: inline-flex; align-items: center; justify-content: center; flex: none;
}
.step span { font-family: 'Bitter', Georgia, serif; font-weight: 500; font-size: 16px; }
.step em { margin-left: auto; font-style: normal; font-size: 12px; color: var(--ink-3); }

.crop-note {
    font-size: 13px; line-height: 1.55; color: var(--ink-2);
    margin: 10px 0 0 0; padding-left: 12px; border-left: 2px solid var(--rule);
}
.crop-note b { font-weight: 600; color: var(--ink); }

/* ---- the prescription ---- */
.rx { border: 1.5px solid var(--ink); background: var(--paper-2); }
.rx-band { background: var(--accent); padding: 13px 22px; display: flex; align-items: baseline; justify-content: space-between; gap: 14px; }
.rx-band strong { font-family: 'Bitter', Georgia, serif; font-weight: 900; font-size: 21px; line-height: 1; letter-spacing: -0.01em; color: #FFFFFF; }
.rx-band span { font-size: 12px; color: rgba(255,255,255,0.82); font-variant-numeric: tabular-nums; white-space: nowrap; }

.rx-body { padding: 22px 24px 24px 24px; }
.rx-name { font-family: 'Bitter', Georgia, serif; font-weight: 700; font-size: 30px; line-height: 1.1; letter-spacing: -0.015em; margin: 0 0 5px 0; }
.rx-kind { font-size: 13.5px; color: var(--ink-2); margin: 0 0 16px 0; padding-bottom: 14px; border-bottom: 1px solid var(--rule); }
.rx-summary { font-size: 15.5px; line-height: 1.6; margin: 0 0 4px 0; max-width: 60ch; }

.rx h3 { font-family: 'Bitter', Georgia, serif; font-weight: 500; font-size: 14.5px; margin: 22px 0 9px 0; }
.rx ul { margin: 0; padding-left: 0; list-style: none; }
.rx li { font-size: 14.5px; line-height: 1.55; margin-bottom: 8px; max-width: 60ch; padding-left: 17px; position: relative; }
.rx li::before { content: ""; position: absolute; left: 0; top: 9px; width: 7px; height: 1.5px; background: var(--accent); }
.rx-note { font-size: 14px; line-height: 1.55; color: var(--ink-2); margin: 22px 0 0 0; padding: 14px 0 0 0; border-top: 1px solid var(--rule); max-width: 60ch; }
.rx-note b { color: var(--ink); font-weight: 600; }

/* ---- probability bars ---- */
.bars { margin-top: 2px; }
.bar { display: grid; grid-template-columns: 1fr 44px; align-items: center; gap: 10px; margin-bottom: 5px; }
.bar-track { position: relative; height: 22px; background: var(--paper-3); }
.bar-fill { position: absolute; inset: 0 auto 0 0; background: var(--ink-3); opacity: 0.5; }
.bar--top .bar-fill { background: var(--accent); opacity: 0.55; }
.bar-name { position: relative; z-index: 1; font-size: 13px; line-height: 22px; padding-left: 9px; white-space: nowrap; }
.bar-val { font-size: 13px; text-align: right; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.bar-track--mark::after { content: ""; position: absolute; top: -4px; bottom: -4px; left: var(--thr); width: 1.5px; background: var(--ink); }
.thr-key { font-size: 12px; color: var(--ink-3); margin: 9px 0 0 0; }

/* ---- rejected photo ---- */
.reject { border: 1.5px solid var(--u-week); background: var(--paper-2); }
.reject .rx-band { background: var(--u-week); }
.reject-body { padding: 22px 24px 24px 24px; }
.reject h2 { font-family: 'Bitter', Georgia, serif; font-weight: 700; font-size: 26px; margin: 0 0 14px 0; line-height: 1.15; }
.reject p { font-size: 15px; line-height: 1.6; margin: 0 0 9px 0; max-width: 60ch; }
.reject .last { color: var(--ink-2); font-size: 14px; margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--rule); }

/* ---- empty state ---- */
.awaiting { border: 1.5px dashed var(--rule); background: transparent; padding: 54px 26px; text-align: center; }
.awaiting h2 { font-family: 'Bitter', Georgia, serif; font-weight: 500; font-size: 19px; margin: 0 0 8px 0; color: var(--ink-2); }
.awaiting p { font-size: 14px; color: var(--ink-3); margin: 0; }
.empty-note { font-size: 13.5px; color: var(--ink-3); padding: 14px 0; }

/* ---- history / queue tables ---- */
.tbl { width: 100%; border-collapse: collapse; margin-top: 12px; }
.tbl th {
    text-align: left; font-family: 'Bitter', Georgia, serif; font-weight: 500;
    font-size: 12px; color: var(--ink-2); border-bottom: 1.5px solid var(--ink);
    padding: 0 10px 8px 0;
}
.tbl td { font-size: 13.5px; padding: 9px 10px 9px 0; border-bottom: 1px solid var(--rule); }
.tbl .abstain-tag { color: var(--u-week); font-weight: 600; }
.tbl .conf-tag { font-variant-numeric: tabular-nums; white-space: nowrap; }

/* ---- gradio overrides ---- */
.gradio-container .block, .gradio-container .form, .gradio-container .panel { background: transparent !important; border: none !important; }

button.primary {
    background: var(--ink) !important; color: var(--paper) !important;
    border: none !important; border-radius: 0 !important;
    font-family: 'Public Sans', sans-serif !important; font-weight: 700 !important;
    font-size: 15px !important; letter-spacing: 0.01em !important;
}
button.primary:hover { background: var(--u-week) !important; }
button.secondary {
    background: var(--paper-2) !important; color: var(--ink) !important;
    border: 1px solid var(--ink) !important; border-radius: 0 !important;
    font-weight: 600 !important; font-size: 14px !important;
}

.gradio-container fieldset label {
    background: var(--paper-2) !important; border: 1px solid var(--rule) !important;
    border-radius: 0 !important; font-weight: 500 !important; font-size: 14px !important;
}
.gradio-container fieldset label input { display: none !important; }
.gradio-container fieldset label.selected { background: var(--ink) !important; border-color: var(--ink) !important; }
.gradio-container fieldset label.selected span { color: var(--paper) !important; }

.gradio-container [data-testid="image"] {
    border: 1px solid var(--rule) !important; border-radius: 0 !important; background: var(--paper-2) !important;
}
.gradio-container input[type="text"], .gradio-container input[type="number"], .gradio-container textarea {
    background: var(--paper-2) !important; border: 1px solid var(--rule) !important; border-radius: 0 !important;
}
.gradio-container .label-wrap {
    border-top: 1px solid var(--ink) !important; padding-top: 14px !important;
    font-family: 'Bitter', Georgia, serif !important; font-weight: 500 !important; font-size: 16px !important;
}
#limits p, #limits li { max-width: 74ch; font-size: 14.5px; line-height: 1.6; }

@media (max-width: 720px) {
    .gradio-container { padding: 0 16px 40px 16px !important; }
    #head h1 { font-size: 34px; }
    .rx-name { font-size: 25px; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
"""

AWAITING = (
    '<div class="awaiting"><h2>No diagnosis yet</h2>'
    '<p>Choose a crop, add a photograph of a single leaf, and press Diagnose.</p>'
    '</div>')


# ---------------------------------------------------------------- rendering --

def crop_note(crop_key: str) -> str:
    crop = registry.get(crop_key)
    conditions = ", ".join(crop.display[c] for c in crop.class_names)
    return (f'<p class="crop-note">Knows <b>{len(crop.class_names)} conditions</b> '
            f'&mdash; {html.escape(conditions)}. Anything outside that list is '
            f'misread as whichever of them looks closest. Macro-F1 '
            f'<b>{crop.test_macro_f1:.2f}</b> on held-out images; declines to '
            f'answer below <b>{crop.threshold:.0%}</b> confidence.</p>')


def _bars(scores: dict, top_name: str, accent: str, threshold: float) -> str:
    rows = []
    for name, value in sorted(scores.items(), key=lambda kv: -kv[1]):
        is_top = name == top_name
        mark = " bar-track--mark" if is_top else ""
        rows.append(
            f'<div class="bar{" bar--top" if is_top else ""}">'
            f'<div class="bar-track{mark}" style="--thr:{threshold * 100:.0f}%">'
            f'<div class="bar-fill" style="width:{value * 100:.1f}%"></div>'
            f'<span class="bar-name">{html.escape(name)}</span></div>'
            f'<span class="bar-val">{value:.2f}</span></div>')
    return (f'<div class="bars" style="--accent:{accent}">{"".join(rows)}</div>'
            f'<p class="thr-key">The vertical mark is the {threshold:.0%} '
            f'threshold. Below it, this crop&rsquo;s model refers rather than '
            f'answers.</p>')


def _list(items) -> str:
    return "<ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in items) + "</ul>"


def render_result(result: dict, guidance: dict) -> str:
    accent, band_label = URGENCY.get(guidance["urgency"],
                                     ("var(--u-refer)", guidance["urgency"]))
    crop = registry.get(result["crop"])

    if result["abstained"]:
        name = "Not determined"
        kind = (f'Closest match was {html.escape(result["top_display"])} at '
                f'{result["confidence"]:.0%} &mdash; below the '
                f'{result["threshold"]:.0%} threshold this model was calibrated to.')
        corner = f'{crop.name} &nbsp;&middot;&nbsp; no determination'
    else:
        name = guidance["name"]
        kind = (f'{html.escape(guidance["kind"])} &nbsp;&middot;&nbsp; '
                f'{result["confidence"]:.0%} confidence')
        corner = (f'{crop.name} &nbsp;&middot;&nbsp; '
                  f'{len(crop.class_names)} conditions considered')

    parts = [
        f'<div class="rx" style="--accent:{accent}">',
        f'<div class="rx-band"><strong>{html.escape(band_label)}</strong>'
        f'<span>{corner}</span></div>',
        '<div class="rx-body">',
        f'<h2 class="rx-name">{html.escape(name)}</h2>',
        f'<p class="rx-kind">{kind}</p>',
        f'<p class="rx-summary">{html.escape(guidance["summary"])}</p>',
        '<h3>What to do</h3>',
        _list(guidance["actions"]),
    ]
    if guidance.get("prevention"):
        parts += ['<h3>Preventing it next season</h3>', _list(guidance["prevention"])]
    if guidance.get("note"):
        parts.append(f'<p class="rx-note"><b>Worth knowing.</b> '
                     f'{html.escape(guidance["note"])}</p>')
    parts += [
        '<h3>All conditions considered</h3>',
        _bars(result["all_scores"], result["top_display"], accent, result["threshold"]),
        '</div></div>',
    ]
    return "".join(parts)


def render_rejection(report) -> str:
    hints = "".join(f"<p>{html.escape(h)}</p>" for h in report.hints)
    return (
        '<div class="reject">'
        '<div class="rx-band"><strong>Photo rejected</strong>'
        f'<span>{html.escape(", ".join(report.problems))}</span></div>'
        '<div class="reject-body"><h2>This photo cannot be read reliably</h2>'
        f'{hints}'
        '<p class="last">The model would still return an answer for it. That '
        'answer would not be worth acting on, which is why nothing is shown.</p>'
        '</div></div>')


def render_history_table(crop_key: str) -> str:
    records = storage.history(crop=crop_key or None, limit=50)
    if not records:
        return '<p class="empty-note">No diagnoses logged yet.</p>'
    rows = []
    for r in records:
        when = r.created_at.replace("T", " ")
        if r.abstained:
            result_cell = (f'<span class="abstain-tag">Referred</span> '
                           f'&mdash; closest {html.escape(r.display_name)}')
        else:
            result_cell = html.escape(r.display_name)
        field = html.escape(r.field_name) if r.field_name else "&mdash;"
        rows.append(f'<tr><td>{when}</td><td>{r.crop}</td><td>{result_cell}</td>'
                    f'<td class="conf-tag">{r.confidence:.0%}</td><td>{field}</td></tr>')
    return ('<table class="tbl"><thead><tr><th>When</th><th>Crop</th>'
            '<th>Result</th><th>Confidence</th><th>Field</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def render_stats() -> str:
    s = storage.stats()
    by_crop = (" &middot; ".join(f"{k}: {v}" for k, v in s["by_crop"].items())
              or "none yet")
    return (f'<p class="crop-note">Logged <b>{s["total_diagnoses"]}</b> diagnoses. '
            f'<b>{s["abstained"]}</b> referred ({s["abstain_rate"]:.0%}), '
            f'<b>{s["pending_review"]}</b> waiting for review.<br>{by_crop}</p>')


def render_queue_table(crop_key: str) -> str:
    rows = storage.pending_review(crop=crop_key or None)
    if not rows:
        return '<p class="empty-note">Nothing waiting for review.</p>'
    out = []
    for r in rows:
        when = r["created_at"].replace("T", " ")
        out.append(f'<tr><td>#{r["queue_id"]}</td><td>{when}</td><td>{r["crop"]}</td>'
                   f'<td>{html.escape(r["display_name"])}</td>'
                   f'<td class="conf-tag">{r["confidence"]:.0%}</td></tr>')
    return ('<table class="tbl"><thead><tr><th>ID</th><th>When</th><th>Crop</th>'
            '<th>Closest guess</th><th>Confidence</th></tr></thead>'
            f'<tbody>{"".join(out)}</tbody></table>')


# ---------------------------------------------------------------- actions --

def save_review_image(image: Image.Image, crop_key: str) -> str:
    """Keep the photo behind a referred case — a reviewer needs to see it,
    unlike a confident diagnosis where there is nothing left to check."""
    folder = os.path.join(REVIEW_IMAGES_DIR, crop_key)
    os.makedirs(folder, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(folder, f"{stamp}.jpg")
    image.convert("RGB").save(path, "JPEG", quality=90)
    return path


def diagnose(image, crop_key, field_name):
    """Quality gate, then classify, then explain, then log.

    Prediction and Grad-CAM share one preprocessed batch and one forward pass
    rather than each doing their own.
    """
    hide = gr.update(value=None, visible=False)

    if image is None:
        return AWAITING, hide
    if not crop_key:
        return ('<div class="reject">'
                '<div class="rx-band"><strong>No crop chosen</strong><span></span></div>'
                '<div class="reject-body"><h2>Choose a crop first</h2>'
                '<p>Each crop has its own model, trained and measured on its own '
                'images. Without knowing which plant this is, there is nothing '
                'to run.</p></div></div>'), hide

    report = quality.check(image, crop_key)
    if not report.ok:
        return render_rejection(report), hide

    crop = registry.get(crop_key)
    batch = predict.to_batch(image, crop.img_size)
    probs = predict.classify_batch(batch, crop_key)

    best = int(probs.argmax())
    label = crop.class_names[best]
    confidence = float(probs[best])
    abstained = confidence < crop.threshold

    result = {
        "crop": crop.key, "crop_name": crop.name,
        "label": None if abstained else label,
        "display_name": ("Uncertain — recommend expert review" if abstained
                         else crop.display[label]),
        "kind": None if abstained else crop.kinds[label],
        "confidence": confidence, "abstained": abstained,
        "top_class": label, "top_display": crop.display[label],
        "threshold": crop.threshold,
        "all_scores": {crop.display[c]: float(p)
                       for c, p in zip(crop.class_names, probs)},
    }

    guidance = advice.get(crop_key, result["label"])
    overlay_img = gradcam.explain(image, crop_key, class_index=best, batch=batch)

    image_path = save_review_image(image, crop_key) if abstained else None
    storage.log_diagnosis(result, field_name=(field_name or "").strip() or None,
                          image_path=image_path)

    return render_result(result, guidance), gr.update(value=overlay_img, visible=True)


def refresh_history(crop_key):
    return render_history_table(crop_key), render_stats()


def label_choices_for(crop_key):
    crop = registry.get(crop_key)
    return [(crop.display[c], c) for c in crop.class_names]


def update_label_choices(crop_key):
    return gr.update(choices=label_choices_for(crop_key), value=None)


def do_label(queue_id, crop_key, true_label, note):
    if not queue_id:
        return render_queue_table(crop_key), "Enter a queue ID from the table above."
    if not true_label:
        return render_queue_table(crop_key), "Choose the true condition before saving."

    storage.label_review(int(queue_id), true_label, note or "")

    # Update the display fields too, so History shows the resolved condition
    # rather than the model's original "Uncertain" text.
    crop = registry.get(crop_key)
    with storage.connect() as conn:
        row = conn.execute(
            "SELECT diagnosis_id FROM review_queue WHERE id = ?", (int(queue_id),)
        ).fetchone()
        conn.execute(
            "UPDATE diagnoses SET display_name = ? WHERE id = ?",
            (f"{crop.display[true_label]} (expert-confirmed)", row["diagnosis_id"]))

    return render_queue_table(crop_key), f"Saved — case #{int(queue_id)} labelled."

def do_dismiss(queue_id, crop_key, note):
    if not queue_id:
        return render_queue_table(crop_key), "Enter a queue ID from the table above."
    storage.dismiss_review(int(queue_id), note or "")
    return render_queue_table(crop_key), f"Dismissed case #{int(queue_id)}."


def gather_examples():
    rows = []
    exts = {".jpg", ".jpeg", ".png"}
    for key in registry.CROPS:
        folder = os.path.join(EXAMPLES_DIR, key)
        if not os.path.isdir(folder):
            continue
        found = []
        for dirpath, _, filenames in os.walk(folder):
            for f in sorted(filenames):
                if os.path.splitext(f)[1].lower() in exts:
                    found.append(os.path.join(dirpath, f))
        for path in found[:2]:
            rows.append([path, key])
    return rows


LIMITS = f"""
**Four separate models, one per crop.** You pick the crop because you know what
you planted. Asking a model to guess adds a failure nobody needed, and keeping
them separate lets each be trained and measured on its own data.

**Each model only knows its own list.** A condition outside that list is not
flagged as unknown — it is misread as whichever listed condition looks closest.
Rice has no entry for sheath blight; cassava has none for green mite.

**It declines when it is not sure.** Below a threshold fitted on validation data,
never on the test set, the result comes back as a referral rather than a guess,
and is added to the review queue. For cassava that means referring 7% of images
and reaching 98% accuracy on the rest; for tomato, referring 28% to reach 90%.

**Accuracy differs by crop, and these are the measured numbers.**
{" ".join(f"{c.name} {c.test_macro_f1:.2f}." for c in registry.CROPS.values())}
Tomato is lowest because verticillium wilt and leaf blight genuinely look alike
on a single leaf.

**The attention map is there to be checked, not admired.** During development it
caught two models scoring above 0.97 that were reading the grey card behind the
leaf rather than the leaf itself.

**Every diagnosis is logged locally, on this device only.** Referred cases keep
their photo so a reviewer can label them under the Review queue tab; confident
diagnoses do not, since there is nothing left to check.

**The guidance has not been reviewed by a plant pathologist.** It was written
from general agronomic sources for a portfolio project. No product names appear
anywhere, because approved products differ by country and a wrong chemical
recommendation wastes money at best.
"""


with gr.Blocks(theme=gr.themes.Base(), css=CSS, title="Crop Doctor") as demo:
    gr.HTML(
        '<div id="head">'
        '<div class="scale">'
        '<i style="background:var(--u-none)"></i>'
        '<i style="background:var(--u-monitor)"></i>'
        '<i style="background:var(--u-week)"></i>'
        '<i style="background:var(--u-now)"></i>'
        '</div>'
        '<h1>Crop Doctor</h1>'
        '<p class="lede">Photograph a leaf and get a diagnosis you can act on '
        'today &mdash; or an honest refusal when the model is not confident '
        'enough to be useful. Cassava, rice, maize and tomato.</p>'
        '<div class="rule"></div>'
        '</div>')

    with gr.Tabs():
        with gr.Tab("Diagnose"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=330):
                    gr.HTML('<div class="step"><b>1</b><span>Which crop?</span></div>')
                    crop_input = gr.Radio(choices=CROP_CHOICES, value="cassava",
                                          show_label=False, container=False)
                    crop_info = gr.HTML(crop_note("cassava"))

                    gr.HTML('<div class="step" style="margin-top:26px"><b>2</b>'
                            '<span>Photograph of a leaf</span>'
                            '<em>one leaf, filling the frame</em></div>')
                    image_input = gr.Image(type="pil", show_label=False, height=250)

                    field_input = gr.Textbox(
                        label="Field or plot name (optional)",
                        info="Helps spot a problem recurring in the same plot. "
                             "Stored on this device only.",
                        placeholder="e.g. North plot")

                    submit = gr.Button("Diagnose", variant="primary", size="lg")

                    examples = gather_examples()
                    if examples:
                        gr.Examples(examples=examples,
                                    inputs=[image_input, crop_input],
                                    label="Or try one of these")

                with gr.Column(scale=6, min_width=380):
                    result_html = gr.HTML(AWAITING)
                    heatmap = gr.Image(label="Which regions drove this answer",
                                       height=260, visible=False)

        with gr.Tab("History"):
            gr.HTML('<div class="step"><b>&middot;</b><span>Past diagnoses on this device</span></div>')
            history_stats = gr.HTML(render_stats())
            with gr.Row():
                history_crop = gr.Dropdown(
                    choices=[("All crops", "")] + CROP_CHOICES, value="",
                    label="Filter by crop", container=True, scale=3)
                history_refresh = gr.Button("Refresh", variant="secondary", scale=1)
            history_table = gr.HTML(render_history_table(""))

        with gr.Tab("Review queue"):
            gr.HTML('<div class="step"><b>&middot;</b>'
                    '<span>Cases the model referred rather than guessed</span></div>')
            gr.Markdown("Each row below is a photo the model was not confident "
                       "enough to diagnose. Enter its ID, choose what it actually "
                       "was, and save — that label is what a retrained model would "
                       "learn from.")

            queue_crop = gr.Dropdown(choices=CROP_CHOICES, value="cassava",
                                     label="Crop")
            queue_table = gr.HTML(render_queue_table("cassava"))

            with gr.Row():
                queue_id_input = gr.Number(label="Queue ID", precision=0)
                true_label_input = gr.Dropdown(
                    choices=label_choices_for("cassava"),
                    label="What it actually was")
            note_input = gr.Textbox(label="Note (optional)",
                                    placeholder="e.g. photo was of a different plant")
            with gr.Row():
                label_btn = gr.Button("Save label", variant="primary")
                dismiss_btn = gr.Button("Dismiss — unusable case", variant="secondary")
            review_status = gr.Markdown("")

    with gr.Accordion("What this can and cannot do", open=False):
        gr.Markdown(LIMITS, elem_id="limits")

    # ---- wiring ----
    crop_input.change(fn=crop_note, inputs=crop_input, outputs=crop_info)
    submit.click(fn=diagnose, inputs=[image_input, crop_input, field_input],
                outputs=[result_html, heatmap])

    history_crop.change(fn=refresh_history, inputs=history_crop,
                        outputs=[history_table, history_stats])
    history_refresh.click(fn=refresh_history, inputs=history_crop,
                          outputs=[history_table, history_stats])

    queue_crop.change(fn=render_queue_table, inputs=queue_crop, outputs=queue_table)
    queue_crop.change(fn=update_label_choices, inputs=queue_crop, outputs=true_label_input)
    label_btn.click(fn=do_label,
                    inputs=[queue_id_input, queue_crop, true_label_input, note_input],
                    outputs=[queue_table, review_status])
    dismiss_btn.click(fn=do_dismiss, inputs=[queue_id_input, queue_crop, note_input],
                      outputs=[queue_table, review_status])


if __name__ == "__main__":
    print("Warming up all four models...")
    for _crop in registry.CROPS.values():
        predict.get_model(_crop.key)
    print("Ready.")
    demo.launch()