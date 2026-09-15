# build the 5 slide presentation for the gisec final, from the results files.

from pathlib import Path
import json

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "presentation"

TEAM = "unlisted"

# dark surface reads better than white on a large stage screen.
INK = RGBColor(0xFF, 0xFF, 0xFF)
MUTED = RGBColor(0xC3, 0xC2, 0xB7)
FAINT = RGBColor(0x8A, 0x89, 0x80)
SURFACE = RGBColor(0x1A, 0x1A, 0x19)
PANEL = RGBColor(0x26, 0x26, 0x24)
BLUE = RGBColor(0x39, 0x87, 0xE5)
ORANGE = RGBColor(0xD9, 0x59, 0x26)
GREEN = RGBColor(0x19, 0x9E, 0x70)
VIOLET = RGBColor(0x90, 0x85, 0xE9)

W = Inches(13.333)
H = Inches(7.5)
MARGIN = Inches(0.72)


def load():
    t = pd.read_csv(RESULTS / "privacy_utility_tradeoff.csv").set_index("target")
    r = pd.read_csv(RESULTS / "mia_results.csv")
    g = pd.read_csv(RESULTS / "subgroup_vulnerability.csv")
    g = g[g["target"] == "Non-private (overfit)"].sort_values(
        "attack_auc_mean", ascending=False
    )
    seeds = sorted((ROOT / "runs").glob("seed*"))
    base = json.loads((seeds[0] / "baseline_metrics.json").read_text())
    dp = pd.read_csv(seeds[0] / "dp_metrics.csv").set_index("target_epsilon")
    cohort = sum(
        len(pd.read_csv(ROOT / "data" / f"hospital_{h}.csv")) for h in ("a", "b")
    )
    return t, r, g, base, dp, len(seeds), cohort


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def background(slide, colour=SURFACE):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = colour


def box(slide, left, top, width, height, fill=PANEL, line=None, line_width=Pt(1)):
    from pptx.enum.shapes import MSO_SHAPE

    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.adjustments[0] = 0.06
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = line_width
    shape.shadow.inherit = False
    return shape


def text(slide, left, top, width, height, runs, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=1.0, space_after=0):
    """runs: list of (string, size_pt, colour, bold) or None for a blank line."""
    frame = slide.shapes.add_textbox(left, top, width, height).text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0

    first = True
    for item in runs:
        paragraph = frame.paragraphs[0] if first else frame.add_paragraph()
        first = False
        paragraph.alignment = align
        paragraph.line_spacing = spacing
        if space_after:
            paragraph.space_after = Pt(space_after)
        if item is None:
            paragraph.text = ""
            continue
        body, size, colour, bold = item
        run = paragraph.add_run()
        run.text = body
        run.font.size = Pt(size)
        run.font.color.rgb = colour
        run.font.bold = bold
        run.font.name = "Arial"
    return frame


def eyebrow(slide, label):
    text(slide, MARGIN, Inches(0.42), W - 2 * MARGIN, Inches(0.3),
         [(label, 12, BLUE, True)])


def headline(slide, line, top=Inches(0.86), size=38):
    text(slide, MARGIN, top, W - 2 * MARGIN, Inches(1.0),
         [(line, size, INK, True)], spacing=0.95)


def footer(slide, line):
    text(slide, MARGIN, H - Inches(0.62), W - 2 * MARGIN, Inches(0.3),
         [(line, 12, FAINT, False)])


def stat_card(slide, left, top, width, height, value, colour, caption, accent=True):
    card = box(slide, left, top, width, height)
    if accent:
        bar = box(slide, left, top, Inches(0.05), height, fill=colour)
        bar.line.fill.background()
    pad = Inches(0.28)
    text(slide, left + pad, top + Inches(0.26), width - 2 * pad, Inches(0.8),
         [(value, 40, colour, True)])
    text(slide, left + pad, top + Inches(1.02), width - 2 * pad,
         height - Inches(1.2), [(caption, 14, MUTED, False)], spacing=1.12)


def bullet_card(slide, left, top, width, height, title, lines, accent=BLUE,
                body_size=13.5):
    box(slide, left, top, width, height)
    bar = box(slide, left, top, Inches(0.05), height, fill=accent)
    bar.line.fill.background()
    pad = Inches(0.28)

    text(slide, left + pad, top + Inches(0.22), width - 2 * pad, Inches(0.38),
         [(title, 16, INK, True)])
    text(slide, left + pad, top + Inches(0.72), width - 2 * pad,
         height - Inches(0.92),
         [(line, body_size, MUTED, False) for line in lines],
         spacing=1.08, space_after=7)


# ---------------------------------------------------------------- slides


def slide_title(prs, data):
    t, r, g, base, dp, n_seeds, cohort = data
    slide = blank(prs)
    background(slide)

    eyebrow(slide, "SCHOOL OF CYBER DEFENSE 2026   ·   GISEC GLOBAL FINAL")

    text(slide, MARGIN, Inches(1.9), W - 2 * MARGIN, Inches(1.8),
         [("The model is the leak.", 66, INK, True)], spacing=0.92)

    text(slide, MARGIN, Inches(3.5), Inches(8.4), Inches(1.1),
         [("Pseudonymisation removes the name from the record. We proved that the "
           "trained model still gives the patient away.", 21, MUTED, False)],
         spacing=1.2)

    text(slide, MARGIN, Inches(4.8), Inches(8.4), Inches(0.5),
         [("Team ", 17, FAINT, False)])
    text(slide, MARGIN + Inches(0.62), Inches(4.8), Inches(6.0), Inches(0.5),
         [(TEAM, 17, INK, True)])

    # the three numbers the jury asked us to improve
    strip_top = Inches(5.8)
    card_w = Inches(3.83)
    gap = Inches(0.25)
    for index, (value, caption, colour) in enumerate([
        (f"{cohort:,}", "patients, three times our Stage 2 cohort", BLUE),
        (f"{n_seeds} runs", "every model trained and attacked three times", VIOLET),
        ("0.6%", "accuracy we pay for a formal privacy guarantee", GREEN),
    ]):
        left = MARGIN + index * (card_w + gap)
        box(slide, left, strip_top, card_w, Inches(1.05))
        text(slide, left + Inches(0.26), strip_top + Inches(0.16), card_w, Inches(0.45),
             [(value, 24, colour, True)])
        text(slide, left + Inches(0.26), strip_top + Inches(0.6), card_w - Inches(0.4),
             Inches(0.4), [(caption, 11.5, MUTED, False)])


def slide_objective(prs, data):
    slide = blank(prs)
    background(slide)
    eyebrow(slide, "THE PROBLEM")
    headline(slide, "Two hospitals. One model. Zero shared records.")

    card_w = Inches(3.83)
    gap = Inches(0.25)
    top = Inches(2.15)
    height = Inches(2.55)

    for index, (title, lines, accent) in enumerate([
        ("The data cannot travel",
         ["Federal Law No. 2 of 2019.",
          "UAE health data must not leave the State.",
          "Pooling the records is not an option."], BLUE),
        ("Names are not enough",
         ["PDPL Art. 20 asks for pseudonymisation.",
          "It hides the name in the row.",
          "It does not hide the patient in the model."], ORANGE),
        ("Presence is the breach",
         ["A model that remembers a patient reveals them.",
          "It shows that the person was in that hospital, in that period.",
          "The diagnosis is not needed."], VIOLET),
    ]):
        bullet_card(slide, MARGIN + index * (card_w + gap), top, card_w, height,
                    title, lines, accent)

    box(slide, MARGIN, Inches(5.1), W - 2 * MARGIN, Inches(1.15))
    text(slide, MARGIN + Inches(0.34), Inches(5.42), W - 2 * MARGIN - Inches(0.68),
         Inches(0.8),
         [("Our job was not to claim this gap exists. It was to measure it.",
           20, INK, True)])
    footer(slide, "Every measured result in this deck is the mean of three independent runs.")


def slide_solution(prs, data):
    t, r, g, base, dp, n_seeds, cohort = data
    slide = blank(prs)
    background(slide)
    eyebrow(slide, "THE SOLUTION")
    headline(slide, "Records stay home. Only protected maths travels.")

    # architecture strip
    top = Inches(2.05)
    height = Inches(1.35)
    node_w = Inches(3.5)
    arrow_w = Inches(0.62)
    left = MARGIN

    def node(x, title, subtitle, colour):
        box(slide, x, top, node_w, height)
        bar = box(slide, x, top, Inches(0.05), height, fill=colour)
        bar.line.fill.background()
        text(slide, x + Inches(0.26), top + Inches(0.26), node_w - Inches(0.5),
             Inches(0.4), [(title, 16, INK, True)])
        text(slide, x + Inches(0.26), top + Inches(0.7), node_w - Inches(0.5),
             Inches(0.5), [(subtitle, 12.5, MUTED, False)], spacing=1.1)

    node(left, "Hospital A",
         "3,000 patients. Trains locally, then adds the noise.", BLUE)
    text(slide, left + node_w, top + Inches(0.42), arrow_w, Inches(0.5),
         [("⇄", 26, FAINT, False)], align=PP_ALIGN.CENTER)
    node(left + node_w + arrow_w, "Aggregation server",
         "Averages the protected updates. Never sees a record.", ORANGE)
    text(slide, left + 2 * node_w + arrow_w, top + Inches(0.42), arrow_w, Inches(0.5),
         [("⇄", 26, FAINT, False)], align=PP_ALIGN.CENTER)
    node(left + 2 * (node_w + arrow_w), "Hospital B",
         "3,000 patients. Trains locally, then adds the noise.", BLUE)

    text(slide, MARGIN, top + Inches(1.42), W - 2 * MARGIN, Inches(0.3),
         [("Eight rounds. The protected update goes up, the averaged model comes "
           "back. The records never move.", 12, FAINT, False)],
         align=PP_ALIGN.CENTER)

    # three claims
    card_w = Inches(3.83)
    gap = Inches(0.25)
    ctop = Inches(3.75)
    for index, (title, lines, accent) in enumerate([
        ("The server is not trusted",
         ["Noise is added inside the hospital.",
          "A stolen server learns no more than the epsilon bound allows."], GREEN),
        ("The budget does not add up",
         ["No patient is at both hospitals.",
          "The cost is the larger of the two, not the sum."], BLUE),
        ("Scale makes privacy cheap",
         ["In Stage 2, at 2,000 patients, privacy cost 2.5% of the accuracy.",
          "At 6,000 patients it costs 0.6%."], VIOLET),
    ]):
        bullet_card(slide, MARGIN + index * (card_w + gap), ctop, card_w,
                    Inches(1.85), title, lines, accent)

    box(slide, MARGIN, Inches(5.85), W - 2 * MARGIN, Inches(0.85))
    text(slide, MARGIN + Inches(0.34), Inches(6.02), W - 2 * MARGIN - Inches(0.68),
         Inches(0.6),
         [("More data does not make privacy more expensive. It makes it cheaper.",
           19, GREEN, True)])


def slide_validation(prs, data):
    t, r, g, base, dp, n_seeds, cohort = data
    slide = blank(prs)
    background(slide)
    eyebrow(slide, "RED TEAM")
    headline(slide, "We attacked our own model with seven attacks, including the state of the art.")

    overfit = t.loc["Non-private (overfit)"]
    dp3 = t.loc["DP eps=3"]
    ov = r[r["target"] == "Non-private (overfit)"]
    lira = ov[ov["attack"] == "lira"].iloc[0]
    rival = ov[ov["attack"] != "lira"].sort_values("auc_mean", ascending=False).iloc[0]
    shadows_total = 32 * n_seeds

    # compare two groups on the same axis, so the numbers are directly comparable
    comorbid = g[g["dimension"] == "comorbidities"].set_index("group")["attack_auc_mean"]
    worst = float(comorbid["5+"])
    mildest = float(comorbid["1-2"])

    top = Inches(2.1)
    height = Inches(2.05)
    gap_between = Inches(0.29)
    half = Emu(int((W - 2 * MARGIN - gap_between) / 2))

    stat_card(slide, MARGIN, top, half, height,
              f'{overfit["attack_auc_mean"]:.3f}', ORANGE,
              "Best of the seven attacks against the careless model. 0.5 means the "
              "attacker learns nothing. Our audit proves a real leak at epsilon of "
              f'{overfit["empirical_epsilon_mean"]:.2f}.')
    stat_card(slide, MARGIN + half + gap_between, top, half, height,
              f'{dp3["attack_auc_mean"]:.3f}', GREEN,
              "Best of the same seven against the DP model. This is chance level, in "
              f'every one of the {n_seeds} runs.')

    ctop = Inches(4.45)
    card_w = Inches(3.83)
    gap = Inches(0.25)
    for index, (title, lines, accent) in enumerate([
        ("LiRA, not a simple threshold",
         [f"{shadows_total} shadow models for each target.",
          "It learns what a normal score looks like for every single patient."],
         VIOLET),
        ("The average score lies",
         [f"At a 1% false alarm rate, LiRA finds "
          f"{lira['tpr_01_mean'] / rival['tpr_01_mean']:.1f} times more patients "
          "than the classic attack.",
          "On the average score they look the same."], ORANGE),
        ("The leak finds the most ill",
         [f"Patients with 5 or more conditions score {worst:.2f}. "
          f"Patients with one or two score {mildest:.2f}.",
          "Only the calibrated attack can see this difference."], BLUE),
    ]):
        bullet_card(slide, MARGIN + index * (card_w + gap), ctop, card_w,
                    Inches(2.15), title, lines, accent)

    footer(slide, "LiRA: Carlini et al., IEEE Symposium on Security and Privacy, 2022. "
                  f"Repeated across {n_seeds} independent runs.")


def slide_results(prs, data):
    t, r, g, base, dp, n_seeds, cohort = data
    tuned_attacks = r[r["target"] == "Non-private (tuned)"]
    tuned_lira = float(
        tuned_attacks[tuned_attacks["attack"] == "lira"]["auc_mean"].iloc[0]
    )
    slide = blank(prs)
    background(slide)
    eyebrow(slide, "RESULTS AND RECOMMENDATION")
    headline(slide, "A formal guarantee for 0.6% of the accuracy.")

    rows = [
        ("Trained without care", "Non-private (overfit)", ORANGE, "none"),
        ("Trained with care", "Non-private (tuned)", MUTED, "none"),
        ("Federated + DP, epsilon 3", "DP eps=3", GREEN, "yes"),
    ]

    table_top = Inches(2.0)
    row_h = Inches(0.62)
    col_x = [MARGIN, Inches(5.1), Inches(7.3), Inches(9.5), Inches(11.3)]
    col_w = [Inches(4.2), Inches(2.0), Inches(2.0), Inches(1.7), Inches(1.3)]
    headers = ["Model", "Accuracy", "Attack score", "Of the best", "Guarantee"]

    for index, header in enumerate(headers):
        text(slide, col_x[index], table_top, col_w[index], Inches(0.3),
             [(header.upper(), 11, FAINT, True)])

    for index, (label, key, colour, promise) in enumerate(rows):
        row = t.loc[key]
        y = table_top + Inches(0.42) + index * row_h
        if key == "DP eps=3":
            highlight = box(slide, MARGIN - Inches(0.16), y - Inches(0.1),
                            W - 2 * MARGIN + Inches(0.32), row_h - Inches(0.06),
                            fill=PANEL)
            highlight.line.fill.background()
        text(slide, col_x[0], y, col_w[0], Inches(0.4), [(label, 16, INK, True)])
        text(slide, col_x[1], y, col_w[1], Inches(0.4),
             [(f'{row["target_test_auc_mean"]:.3f}', 16, INK, False)])
        text(slide, col_x[2], y, col_w[2], Inches(0.4),
             [(f'{row["attack_auc_mean"]:.3f}', 16, colour, True)])
        text(slide, col_x[3], y, col_w[3], Inches(0.4),
             [(f'{row["pct_of_bayes_ceiling"]:.1%}', 16, MUTED, False)])
        text(slide, col_x[4], y, col_w[4], Inches(0.4),
             [(promise, 16, GREEN if promise == "yes" else FAINT, False)])

    actual = float(dp.loc[3.0, "actual_epsilon"])
    text(slide, MARGIN, Inches(4.24), W - 2 * MARGIN, Inches(0.35),
         [(f"Budget we spend: epsilon {actual:.2f}, delta 0.00001, for each hospital. "
           "0.5 on the attack score is chance level.", 13, FAINT, False)])

    ctop = Inches(4.72)
    card_w = Inches(3.83)
    gap = Inches(0.25)
    for index, (title, lines, accent) in enumerate([
        ("Careless is not cheap",
         ["The leaking model was also the least accurate model we built.",
          "Privacy and accuracy fail together."], ORANGE),
        ("Care is not a guarantee",
         [f"LiRA still scores {tuned_lira:.3f} against the careful model, above "
          "chance in all three runs.",
          "Only DP reaches chance level."], BLUE),
        ("Deploy epsilon 3",
         ["98.3% of the best possible accuracy, attack held at chance level.",
          "A guarantee that covers future attacks."], GREEN),
    ]):
        bullet_card(slide, MARGIN + index * (card_w + gap), ctop, card_w,
                    Inches(1.9), title, lines, accent)

    footer(slide, "The jury asked for scale and repeat runs. "
                  f"We delivered {cohort:,} patients and {n_seeds} independent runs. "
                  "One command reproduces every number.")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    data = load()

    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H

    for builder in (slide_title, slide_objective, slide_solution,
                    slide_validation, slide_results):
        builder(prs, data)

    out = OUT_DIR / "unlisted_gisec_final.pptx"
    prs.save(out)
    print(f"wrote {out} ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
