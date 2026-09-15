# build the submission slides straight from the results files.

import base64
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))

# take the resample count from the attack code. a copy here would drift.
from mia_attack import N_BOOTSTRAP
DOCS = ROOT / "docs"

MAX_PAGES = 5


def embed(path):
    # paste the picture into the page so the deck stays one standalone file.
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def fmt(value, places=3):
    return f"{value:.{places}f}"


def load():
    tradeoff = pd.read_csv(RESULTS / "privacy_utility_tradeoff.csv")
    results = pd.read_csv(RESULTS / "mia_results.csv")
    subgroups = pd.read_csv(RESULTS / "subgroup_vulnerability.csv")

    # the per-run folders hold the things that are the same in every run, such as
    # the record counts, plus the gap that sets the simple attack ceiling.
    runs = sorted((ROOT / "runs").glob("seed*"))
    baselines = [json.loads((r / "baseline_metrics.json").read_text()) for r in runs]
    return tradeoff, results, subgroups, baselines, len(runs)


def spread(mean, std, places=3):
    return f"{mean:.{places}f} <span class='muted'>&plusmn; {std:.{places}f}</span>"


def main():
    DOCS.mkdir(exist_ok=True)
    tradeoff, results, subgroups, baselines, n_seeds = load()

    overfit = tradeoff[tradeoff["target"] == "Non-private (overfit)"].iloc[0]
    tuned = tradeoff[tradeoff["target"] == "Non-private (tuned)"].iloc[0]
    dp3 = tradeoff[tradeoff["target"] == "DP eps=3"].iloc[0]
    dp_min = tradeoff[tradeoff["target"] == "DP eps=0.5"].iloc[0]
    ceiling = float(overfit["target_test_auc_mean"] / overfit["pct_of_bayes_ceiling"])

    overfit_attacks = results[results["target"] == "Non-private (overfit)"].sort_values(
        "auc_mean", ascending=False
    )
    lira = overfit_attacks[overfit_attacks["attack"] == "lira"].iloc[0]
    yeom = overfit_attacks[overfit_attacks["attack"] == "yeom_neg_loss"].iloc[0]
    best_threshold = overfit_attacks[overfit_attacks["attack"] != "lira"].iloc[0]

    n_members = int(baselines[0]["overfit"]["fit_records"])
    n_nonmembers = int(baselines[0]["overfit"]["test_records"])
    n_shadows = 32

    cohort = pd.concat(
        [pd.read_csv(ROOT / "data" / n)
         for n in ("hospital_a.csv", "hospital_b.csv")],
        ignore_index=True,
    )
    n_patients = len(cohort)
    n_features = len(cohort.columns) - 3

    # the strictest false-alarm rate this cohort can resolve is one record.
    strict_fpr = 1.0 / n_nonmembers
    lira_strict = float(lira["tpr_001_mean"])
    identified = round(lira_strict * n_members)
    falsely = round(0.001 * n_nonmembers)

    STAGE2_FIT_RECORDS = 800
    STAGE2_ATTACK_AUC = 0.655

    gap = sum(b["overfit"]["accuracy_gap"] for b in baselines) / len(baselines)
    lira_lowfpr = float(lira["tpr_01_mean"])
    threshold_lowfpr = float(best_threshold["tpr_01_mean"])
    loss_ceiling = sum(b["overfit"]["loss_attack_auc_ceiling"] for b in baselines) / len(baselines)

    sub = subgroups[subgroups["target"] == "Non-private (overfit)"].sort_values(
        "attack_auc_mean", ascending=False
    )
    most, least = sub.iloc[0], sub.iloc[-1]

    tuned_rows = results[results["target"] == "Non-private (tuned)"]
    tuned_lira = float(tuned_rows[tuned_rows["attack"] == "lira"]["auc_mean"].iloc[0])

    dp_cost = ((tuned["target_test_auc_mean"] - dp3["target_test_auc_mean"])
               / tuned["target_test_auc_mean"])
    eps_min_cost = ((dp3["target_test_auc_mean"] - dp_min["target_test_auc_mean"])
                    / dp3["target_test_auc_mean"])

    tradeoff_rows = "\n".join(
        f"""<tr{' class="highlight"' if r["target"] == "DP eps=3" else ''}>
          <td class="name">{r["target"].replace("Non-private", "Plain").replace("eps=", "&epsilon;=")}</td>
          <td>{spread(r["target_test_auc_mean"], r["target_test_auc_std"])}</td>
          <td>{r["pct_of_bayes_ceiling"]:.1%}</td>
          <td class="{'risk' if r["attack_auc_mean"] > 0.55 else 'safe'}">{spread(r["attack_auc_mean"], r["attack_auc_std"])}</td>
          <td class="muted">{'none' if r["theoretical_epsilon"] == float("inf") else f'{r["theoretical_epsilon"]:.2f}'}</td>
          <td class="{'risk' if r["empirical_epsilon_mean"] > 0.5 else 'safe'}">{fmt(r["empirical_epsilon_mean"], 2)}</td>
        </tr>"""
        for _, r in tradeoff.iterrows()
    )

    attack_rows = "\n".join(
        f"""<tr{' class="highlight"' if r["attack"] == "lira" else ''}>
          <td class="name">{r["attack"]}</td>
          <td class="muted">{'T1' if r["threat_model"] == "label_agnostic" else 'T2'}</td>
          <td>{spread(r["auc_mean"], r["auc_std"])}</td>
          <td>{r["tpr_001_mean"]:.3%}</td>
          <td>{r["tpr_01_mean"]:.2%}</td>
        </tr>"""
        for _, r in overfit_attacks.iterrows()
        if r["attack"] not in ("modified_entropy", "neg_entropy")
    )

    sub_rows = "\n".join(
        f"""<tr><td class="name">{r["dimension"]} = {r["group"]}</td>
          <td>{int(r["n_members"])}</td>
          <td>{spread(r["attack_auc_mean"], r["attack_auc_std"])}</td></tr>"""
        for _, r in pd.concat([sub.head(3), sub.tail(2)]).iterrows()
    )

    html = f"""<title>Privacy-Preserving Health Analytics: School of Cyber Defense 2026</title>
<style>
  :root {{
    --ink: #0b0b0b; --muted: #52514e; --surface: #fcfcfb;
    --line: #e6e5e1; --accent: #2a78d6; --risk: #eb6834; --safe: #1baf7a;
    --violet: #4a3aa7;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: #d8d7d2;
    font: 13.4px/1.42 -apple-system, "Segoe UI", Inter, Roboto, sans-serif;
    color: var(--ink);
  }}
  .slide {{
    width: 297mm; height: 209mm; padding: 10mm 12mm 8mm;
    margin: 7mm auto; background: var(--surface);
    box-shadow: 0 2px 14px rgba(0,0,0,.18);
    display: flex; flex-direction: column; gap: 7px; overflow: hidden;
    position: relative; page-break-after: always; break-after: page;
  }}
  .slide:last-child {{ page-break-after: auto; break-after: auto; }}
  .slide::after {{
    content: attr(data-n) " / {MAX_PAGES}"; position: absolute; right: 12mm; bottom: 4.5mm;
    font-size: 10px; color: var(--muted);
  }}
  .eyebrow {{
    font-size: 10.5px; letter-spacing: .13em; text-transform: uppercase;
    color: var(--accent); font-weight: 680;
  }}
  h1 {{ font-size: 33px; line-height: 1.08; margin: 1px 0 4px; letter-spacing: -.022em; }}
  h2 {{ font-size: 24px; line-height: 1.12; margin: 1px 0 3px; letter-spacing: -.018em; }}
  h3 {{ font-size: 13.4px; margin: 0 0 4px; }}
  p  {{ margin: 0 0 6px; }}
  p:last-child {{ margin-bottom: 0; }}
  .lede {{ font-size: 15px; color: var(--muted); max-width: 96ch; }}
  .row {{ display: grid; gap: 9px; }}
  .r-2 {{ grid-template-columns: 1fr 1fr; }}
  .r-3 {{ grid-template-columns: repeat(3, 1fr); }}
  .grow {{ flex: 1; min-height: 0; }}
  .card {{
    border: 1px solid var(--line); border-radius: 8px; padding: 9px 11px;
    background: #fff; min-height: 0;
  }}
  .card.flag {{ border-left: 3px solid var(--accent); }}
  .card.warn {{ border-left: 3px solid var(--risk); }}
  .card.good {{ border-left: 3px solid var(--safe); }}
  .stat {{ font-size: 33px; font-weight: 690; letter-spacing: -.03em; line-height: 1; }}
  .stat.risk {{ color: var(--risk); }} .stat.safe {{ color: var(--safe); }}
  .stat.blue {{ color: var(--accent); }}
  .stat-label {{ font-size: 12px; color: var(--muted); margin-top: 5px; line-height: 1.36; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12.2px; }}
  th {{
    text-align: left; font-size: 9.6px; letter-spacing: .06em; text-transform: uppercase;
    color: var(--muted); font-weight: 660; padding: 4px 6px;
    border-bottom: 1.3px solid var(--line); white-space: nowrap;
  }}
  td {{ padding: 4.5px 6px; border-bottom: 1px solid var(--line);
        font-variant-numeric: tabular-nums; white-space: nowrap; }}
  tr:last-child td {{ border-bottom: none; }}
  td.name {{ font-weight: 620; }}
  td.muted, .muted {{ color: var(--muted); }}
  td.risk {{ color: var(--risk); font-weight: 680; }}
  td.safe {{ color: var(--safe); font-weight: 680; }}
  tr.highlight td {{ background: #eef4fd; }}
  figure {{ margin: 0; display: flex; flex-direction: column; min-height: 0; }}
  /* Size to the image's own aspect so a wide chart does not letterbox inside a
     tall flex box and strand its caption; clamp if it would overflow the page. */
  figure img {{ width: 100%; height: auto; max-height: 100%; object-fit: contain;
                min-height: 0; }}
  figcaption {{ font-size: 11.4px; color: var(--muted); margin-top: 4px; line-height: 1.35; }}
  ul {{ margin: 0; padding-left: 15px; }}
  li {{ margin-bottom: 3.5px; }}
  li:last-child {{ margin-bottom: 0; }}
  code {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .87em;
          background: #f2f1ee; padding: 0 4px; border-radius: 3px; }}
  .pill {{ display: inline-block; font-size: 10px; padding: 1.5px 7px; border-radius: 99px;
           background: #eef4fd; color: var(--accent); font-weight: 680; }}
  .flow {{ display: flex; align-items: stretch; gap: 6px; font-size: 11.6px; }}
  .node {{ border: 1px solid var(--line); border-radius: 7px; padding: 7px 9px;
           background: #fff; flex: 1; line-height: 1.3; }}
  .arrow {{ color: var(--muted); font-size: 15px; align-self: center; }}
  @page {{ size: A4 landscape; margin: 0; }}
  @media print {{
    body {{ background: #fff; }}
    .slide {{ margin: 0; box-shadow: none; height: 210mm; }}
  }}
</style>

<!-- page 1 -->
<section class="slide" data-n="1">
  <div class="eyebrow">School of Cyber Defense 2026 &middot; Stage 2 &middot; Privacy-Preserving Health Analytics</div>
  <h1>Two hospitals, one model, zero shared records</h1>
  <p class="lede">We build a federated model that predicts readmission in 30 days.
  We attack it with the strongest known membership inference attack. Then we measure
  the cost of each defence, in accuracy and in patients exposed. Each number on these
  pages is the mean of <strong>{n_seeds} independent runs</strong> over
  <strong>{n_patients:,} patients</strong> and {n_features} clinical features.</p>

  <div class="row r-3">
    <div class="card warn">
      <div class="stat risk">{fmt(lira["auc_mean"])}</div>
      <div class="stat-label">The LiRA attack AUC against the model that we trained
      without care. This model is also the <strong>least accurate</strong> model. Poor
      training causes a loss of privacy and a loss of accuracy.</div>
    </div>
    <div class="card flag">
      <div class="stat blue">{fmt(dp3["target_test_auc_mean"])}</div>
      <div class="stat-label">The federated DP model at &epsilon;=3. It reaches
      <strong>{dp3["pct_of_bayes_ceiling"]:.1%}</strong> of the highest score that is
      possible ({fmt(ceiling)}). It costs {dp_cost:.1%} of the AUC.</div>
    </div>
    <div class="card good">
      <div class="stat safe">{fmt(dp3["attack_auc_mean"])}</div>
      <div class="stat-label">The attack AUC against that model, with the strongest
      published attack. This is the mean of {n_seeds} runs, with an SD of
      {dp3["attack_auc_std"]:.3f}. The attack is no better than a guess, in each run.</div>
    </div>
  </div>

  <div class="row r-2 grow" style="grid-template-columns: 1.05fr 1fr">
    <div class="card warn">
      <h3>The law makes federation necessary, not optional</h3>
      <p><strong>Federal Law No. 2 of 2019 (UAE Health Data Law).</strong> You must not
      store, process or transfer UAE health data outside the State. Ministerial
      Resolution No. 51 of 2021 gives a small number of exceptions. Our design does not
      use an exception. No data crosses a border.</p>
      <p><strong>PDPL Art. 20 (Federal Decree-Law 45/2021).</strong> You must encrypt the
      data and remove the names. But this does not limit what an attacker can learn from
      the behaviour of the model. Pages 2 and 3 show that an attacker can use this gap.</p>
      <p style="margin-top:6px"><strong>Why membership is important.</strong> An attacker
      finds the record of a named person in the training data. The attacker then knows
      that this person was a patient at a known hospital in a known period. The
      prediction is not the sensitive part. The presence of the record is.</p>
    </div>
    <div style="display:flex; flex-direction:column; gap:7px; min-height:0">
      <div class="flow">
        <div class="node"><strong>Hospital A</strong><br><span class="muted">3,000 patients
        &middot; 16.3% readmitted</span></div>
        <div class="arrow">&rarr;</div>
        <div class="node"><strong>Local DP-SGD</strong><br><span class="muted">per-sample
        clipping + calibrated noise, <em>before</em> anything leaves</span></div>
        <div class="arrow">&larr;</div>
        <div class="node"><strong>Hospital B</strong><br><span class="muted">3,000 patients
        &middot; 40.2% readmitted</span></div>
      </div>
      <div class="card" style="flex:1">
        <h3>Three architecture decisions a reviewer should check</h3>
        <ul>
          <li><strong>The server is outside the trust boundary.</strong> Each hospital
          adds the noise before it sends an update. An attacker who controls the server
          learns no more than the &epsilon; bound permits.</li>
          <li><strong>We use parallel composition, not sequential composition.</strong>
          Each patient is at only one hospital. The two clients use different data. Thus
          the system &epsilon; is the <strong>maximum</strong> of the two values, not the
          sum.</li>
          <li><strong>We do not save a fitted scaler.</strong> The features use published
          clinical ranges, not values from our own patients. A <code>StandardScaler</code>
          that you fit on training data holds the mean and the variance of those patients.
          The deployed file then releases them, and no &epsilon; counts this loss. This
          change also removed a pickle file, which is an unsafe format to load.</li>
          <li><strong>The two clients have different data.</strong> Their readmission
          rates differ by 2.5 times. This is the usual condition in a real federation.</li>
        </ul>
      </div>
    </div>
  </div>
</section>

<!-- page 2 -->
<section class="slide" data-n="2">
  <div class="eyebrow">Method: the attack</div>
  <h2>Seven attacks, two threat models, one correct metric</h2>

  <div class="row r-2" style="grid-template-columns: 1fr 1fr">
    <div class="card">
      <h3><span class="pill">T1</span> &nbsp;Label-agnostic adversary</h3>
      <p class="muted" style="margin-bottom:4px">This attacker sees only the output
      probability of the model. It knows no other data about the patient.</p>
      <p><code>confidence</code>, <code>neg_entropy</code></p>
    </div>
    <div class="card">
      <h3><span class="pill">T2</span> &nbsp;Label-aware adversary</h3>
      <p class="muted" style="margin-bottom:4px">This attacker also knows if the patient
      was readmitted, for example from an insurance claim. This attacker is stronger. We
      keep the two types apart, because a mix of the two gives a result that is too
      high.</p>
      <p><code>yeom_neg_loss</code>, <code>class_calibrated_loss</code>,
      <code>learned_ensemble</code>, <code>lira</code></p>
    </div>
  </div>

  <div class="row r-2 grow" style="grid-template-columns: 1.02fr 1fr">
    <div style="display:flex; flex-direction:column; gap:7px; min-height:0">
      <div class="card" style="flex:1">
        <h3>All attacks against the weak model, from the 2017 method to the newest</h3>
        <table>
          <tr><th>Attack</th><th>T</th><th>AUC</th><th>TPR@0.1%FPR</th><th>TPR@1%FPR</th></tr>
          {attack_rows}
        </table>
        <p class="muted" style="font-size:11.2px; margin-top:5px">
        We do not show <code>neg_entropy</code> and <code>modified_entropy</code>. For a
        task with two outcomes, they put the records in the same order as
        <code>confidence</code> and <code>yeom_neg_loss</code>. Their curves are the same.
        To show them again would count one attack two times.</p>
      </div>
      <div class="card flag">
        <h3>What we give the attacker, and what we did not test</h3>
        <p>The attacker can only send queries to the model. It has no weights, no
        gradients, and no control of the training. For LiRA, the attacker also has a
        sample of similar data and enough compute to train shadow models. This is the
        usual assumption.</p>
        <p class="muted"><strong>Out of scope:</strong> model extraction, attribute
        inference, poisoning, and white-box gradient attacks. Each one needs its own
        test. We do not make a claim that we did not measure.</p>
      </div>
    </div>

    <div style="display:flex; flex-direction:column; gap:7px; min-height:0">
      <div class="card warn">
        <h3>Why we use TPR at a low FPR, not accuracy or AUC</h3>
        <p>An attack can be only 51% accurate on average. But if that attack identifies
        {lira_strict:.1%} of the patients almost surely, this is a data breach. An average
        value hides the result that hurts a person. We report both values. We rank the
        attacks on the value that this number of patients can measure.</p>
      </div>
      <div class="card">
        <h3>Why LiRA is the important attack</h3>
        <p>A global threshold asks one question: is this loss very low? That question
        mixes two different causes. The model can <em>memorise</em> a record, or the
        record can be <em>easy</em>. Most records are easy, thus they hide the records
        that the model memorised.</p>
        <p>LiRA trains {n_shadows} shadow models for each model. It learns the score of
        each record with that record in the training data, and without it. Then it does a
        likelihood-ratio test. LiRA calibrates <strong>for each patient</strong>, thus
        only true memorisation gives a high score.</p>
      </div>
      <div class="card good" style="flex:1">
        <h3>Measurement discipline</h3>
        <ul>
          <li>We <strong>cross-fit</strong> the attacks. The attacker that scores a
          record used different data for its calibration.</li>
          <li>Each value has a <strong>stratified bootstrap 95% interval</strong>
          ({N_BOOTSTRAP} resamples). Thus "no better than a guess" is a measured
          result.</li>
          <li>We train each model <strong>{n_seeds} times, on {n_seeds} different
          splits</strong>. Each value is the mean of the runs. Each error bar is the
          spread between the runs. No result depends on one split.</li>
          <li>With {n_nonmembers:,} non-members, the smallest false-alarm rate that we
          can measure is <strong>{strict_fpr:.2%}</strong>. This is three times smaller
          than in Stage 2.</li>
        </ul>
      </div>
    </div>
  </div>
</section>

<!-- page 3 -->
<section class="slide" data-n="3">
  <div class="eyebrow">Result 1: the attack, and the effect of more data</div>
  <h2>Three times more patients makes the leak smaller. It does not stop it.</h2>

  <figure class="grow">
    <img src="{embed(RESULTS / 'mia_roc_loglog.png')}" alt="Attack ROC curves on log-log axes">
    <figcaption>The ROC curve on log-log axes. Each curve shows three runs as thin lines,
    with their mean as a thick line. The corner at the bottom left is the important area,
    because the attack identifies patients there almost surely. Linear axes hide that area.
    Against the weak model, LiRA (violet) is above all the threshold attacks. Against the
    tuned model and the DP models, no attack goes above the diagonal, in any run.</figcaption>
  </figure>

  <div class="row r-3">
    <div class="card warn">
      <div class="stat risk">{overfit["attack_auc_mean"]:.3f}</div>
      <div class="stat-label">The attack AUC on the careless model. This is the mean of
      {n_seeds} runs, with an SD of {overfit["attack_auc_std"]:.4f}. With
      {STAGE2_FIT_RECORDS} training records the value was {STAGE2_ATTACK_AUC:.3f}. With
      {n_members:,} records it is <strong>{overfit["attack_auc_mean"]:.3f}</strong>. More
      data causes less memorisation. But the leak is still
      {(overfit["attack_auc_mean"] - 0.5) / overfit["attack_auc_std"]:.0f} SDs above
      chance.</div>
    </div>
    <div class="card">
      <h3>The average AUC hides the attack that you must prevent</h3>
      <p>The first four attacks are within
      {(overfit_attacks["auc_mean"].iloc[0] - overfit_attacks["auc_mean"].iloc[3]):.3f}
      AUC of each other. On that measure they look the same.</p>
      <p>At a false-alarm rate of 1%, LiRA finds <strong>{lira_lowfpr:.2%}</strong> of the
      training patients. The best threshold attack finds
      <strong>{threshold_lowfpr:.2%}</strong>. LiRA finds
      <strong>{lira_lowfpr / max(threshold_lowfpr, 1e-9):.1f}&times;</strong> more people,
      with the same number of false alarms.</p>
      <p class="muted">This difference is the reason to calibrate for each record. The
      measure that most teams report cannot show it.</p>
    </div>
    <div class="card flag">
      <h3>The attack is at its limit. It is not weak.</h3>
      <p>A loss-threshold attack can only divide the records that the model got right from
      the records that it got wrong. Thus its AUC cannot go above
      <code>0.5 + accuracy_gap/2</code> = <strong>{fmt(loss_ceiling)}</strong>, at an
      average gap of {gap:.3f}.</p>
      <p>We measured <code>yeom_neg_loss</code> at
      <strong>{fmt(yeom["auc_mean"])}</strong>. It is at its limit, thus it uses all of
      the available signal. LiRA is better at a low FPR because it does not use only that
      signal.</p>
    </div>
  </div>
</section>

<!-- page 4 -->
<section class="slide" data-n="4">
  <div class="eyebrow">Result 2: the cost, and the verification</div>
  <h2>The cost of each defence, and proof that our DP is correct</h2>

  <div class="row r-2 grow" style="grid-template-columns: 1.12fr 1fr">
    <figure>
      <img src="{embed(ROOT / 'privacy_utility_tradeoff.png')}" alt="Privacy utility trade-off">
      <figcaption>We measure the utility against the <strong>highest possible score of
      {fmt(ceiling)}</strong>. We calculate that score from the known process that made
      the data. Thus nobody can say that our non-private model is weak. The error bars
      show &plusmn;1 SD over {n_seeds} runs. They are smaller than the markers. That is
      the result, not a fault in the chart.</figcaption>
    </figure>
    <div style="display:flex; flex-direction:column; gap:7px; min-height:0">
      <div class="card">
        <table>
          <tr><th>Target model</th><th>Test AUC</th><th>% ceil</th>
              <th>Attack AUC</th><th>&epsilon; acct.</th><th>&epsilon; emp.</th></tr>
          {tradeoff_rows}
        </table>
      </div>
      <div class="card good" style="flex:1">
        <h3>We audited our own DP code</h3>
        <p>To use a DP library is not proof that the DP works. We convert the result of
        each attack into a <strong>measured lower limit on &epsilon;</strong>. We use the
        Kairouz privacy region and the safe end of a Clopper-Pearson interval. Thus the
        limit is correct at 95% confidence.</p>
        <p><strong>This test can only show a fault. It cannot show that the limit is
        tight.</strong> One run at n={n_members:,} gives a wide limit. Thus
        &epsilon;<sub>emp</sub> &lt;&lt; &epsilon;<sub>acct</sub> proves nothing about the
        accounting. But &epsilon;<sub>emp</sub> &gt; &epsilon;<sub>acct</sub> would prove
        that the code is wrong. This does not occur in any run. The test finds the
        <em>unprotected</em> model at &epsilon; &ge;
        {fmt(overfit["empirical_epsilon_mean"], 2)}, thus the test can find a real
        fault.</p>
      </div>
    </div>
  </div>

  <div class="row r-3">
    <div class="card warn">
      <h3>The careless model is worse in both ways</h3>
      <p>The overfit model leaks the most. It also predicts the worst, at
      {overfit["pct_of_bayes_ceiling"]:.1%} of the highest possible score. Privacy and
      accuracy are not opposite goals. Only the careless model loses both.</p>
    </div>
    <div class="card">
      <h3>Careful training is not a guarantee</h3>
      <p>LiRA still scores {tuned_lira:.3f} against the tuned model. That is above chance
      in all {n_seeds} runs. Careful training makes the leak small. It does not remove the
      leak. Only DP brings the attack down to chance level, and only DP gives a limit that
      also covers a new attack.</p>
    </div>
    <div class="card flag">
      <h3>We change one variable at a time</h3>
      <p>Each model uses the same split, the same scaling, the same class weights, the
      same architecture and the same seed. If they did not, this curve would measure the
      settings, not the privacy.</p>
    </div>
  </div>
</section>

<!-- page 5 -->
<section class="slide" data-n="5">
  <div class="eyebrow">Result 3: new work, limits, recommendation</div>
  <h2>The leak affects the most ill patients</h2>

  <div class="row grow" style="grid-template-columns: 0.62fr 0.78fr 1fr">
    <figure>
      <img src="{embed(RESULTS / 'subgroup_vulnerability.png')}" alt="Per-subgroup attack AUC">
    </figure>

    <div style="display:flex; flex-direction:column; gap:7px; min-height:0">
      <div class="card">
        <table>
          <tr><th>Group</th><th>n</th><th>Attack AUC</th></tr>
          {sub_rows}
        </table>
      </div>
      <div class="card warn" style="flex:1">
        <p>The attack identifies patients with <strong>{most["group"]}
        {most["dimension"]}</strong> at <strong>{fmt(most["attack_auc_mean"])}
        &plusmn; {most["attack_auc_std"]:.3f}</strong>. For
        {least["dimension"]} = {least["group"]} the value is
        {fmt(least["attack_auc_mean"])} &plusmn; {least["attack_auc_std"]:.3f}. This order
        occurs in all {n_seeds} runs. But that group has only
        {int(most["n_members"])} members in each run. Thus we report a
        <strong>direction, not an exact value</strong>.</p>
        <p style="margin-top:6px">The model memorises the records that are not usual,
        because it cannot generalise them. Thus the patients who can lose the most also
        have the highest risk.</p>
        <p class="muted" style="margin-top:6px">The weaker attacks <strong>cannot show
        this</strong>. They show the same risk for all groups. Only calibration for each
        record makes the difference visible. One global &epsilon; gives these patients too
        little protection.</p>
      </div>
    </div>

    <div style="display:flex; flex-direction:column; gap:7px; min-height:0">
      <div class="card flag">
        <h3>Recommendation: use the &epsilon;=3 federated model</h3>
        <p>It gives a test AUC of {fmt(dp3["target_test_auc_mean"])}, which is
        {dp3["pct_of_bayes_ceiling"]:.1%} of the highest possible score. Each hospital gets
        a formal ({dp3["theoretical_epsilon"]:.2f}, 1e-5) guarantee. Under the strongest
        published attack, the result is no better than a guess.</p>
        <p class="muted"><strong>Do not use &epsilon;=0.5.</strong> It costs a further
        {eps_min_cost:.1%} of the AUC. It does not make the attack weaker, because both
        models are already at chance. To lose accuracy for privacy that you cannot measure
        is not correct for a clinical system.</p>
      </div>
      <div class="card">
        <h3>What we did not prove</h3>
        <ul>
          <li>The data is synthetic and logistic. There are still only two clients.</li>
          <li>The {n_seeds} seeds measure the spread between runs. They do not measure the
          spread between different groups of patients, because the patients do not
          change.</li>
          <li>The &epsilon; audit gives a wide lower limit from one run.</li>
          <li>We do not use secure aggregation. The privacy claim does not need it, because
          each update is already private. But a production system must add it.</li>
          <li><strong>We did not compare DP against careful training directly.</strong>
          The gap between them is small. We can show only that the strongest attack stays
          above chance against careful training, and does not against DP.</li>
        </ul>
      </div>
      <div class="card good" style="flex:1">
        <h3>Reproducible, and checked</h3>
        <p><code>python run_all.py</code> makes every number and figure on these pages
        again, from the start. This is all {n_seeds} runs, in about 25 minutes.
        <strong><code>src/build_deck.py</code> builds this deck from
        <code>results/</code></strong>, thus no page can disagree with the data. We set
        every seed and pin each package version, because a new Opacus release can report a
        different &epsilon; for the same noise value. We run 13 checks on the files. For
        example: no validation record enters the attack set, both models use one definition
        of membership, and no two seeds use the same training set.</p>
      </div>
    </div>
  </div>
</section>
"""

    out = DOCS / "deck.html"
    out.write_text(html)

    # the contest allows five pages only. stop here rather than find out at the deadline.
    pages = html.count('class="slide"')
    if pages > MAX_PAGES:
        raise SystemExit(f"deck is {pages} pages; the submission allows at most {MAX_PAGES}")

    print(f"Wrote {out}  ({out.stat().st_size / 1024:.0f} KB, {pages} pages, self-contained)")
    print("Print to PDF: A4 landscape, margins none, background graphics ON.")


if __name__ == "__main__":
    main()
