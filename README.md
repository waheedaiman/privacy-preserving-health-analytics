# Privacy-Preserving Health Analytics

Two hospitals want one shared computer model. The model predicts which patients
will return to hospital within 30 days. Neither hospital is allowed to give its
patient records to the other, and UAE law does not permit it.

This project builds that model. Then it **attacks the model** to see what an
outsider can learn about the patients. Then it measures how well each defence
works, and what each defence costs in accuracy.

## The result in one paragraph

A model that is trained without care leaks information about its patients. An
attacker can work out which people were used to train it. That same careless
model is also the **least accurate** model that we built. A model that is trained
with care makes the leak small, but it does not remove it. The strongest attack
still beats chance against that model in all three runs. Only differential
privacy brings the attack down to chance level. Only it gives a promise that
holds against future attacks. At our scale, that promise costs **0.6% of the accuracy**.

---

## The words you need

You need six terms to read the rest of this page.

| Term | What it means |
|---|---|
| **Readmission model** | Software that gives each patient a risk score. The score is the chance that the patient returns to hospital within 30 days. |
| **Membership attack** | An attempt to find out if one named person's record was used to train the model. This is a privacy breach, because it shows that the person was in that hospital. |
| **Federated learning** | Each hospital trains on its own patients. The hospitals send only the trained settings, never the patient records. |
| **Differential privacy (DP)** | A method that adds controlled random noise during training. The noise limits how much any one patient can change the result. |
| **Epsilon** | The size of the privacy promise. A **smaller** epsilon gives **more** privacy and a little less accuracy. |
| **AUC** | A score between 0.5 and 1.0. For the model, 1.0 is perfect and 0.5 is a guess. For the attack, 0.5 means the attacker learns nothing. |

---

## What changed after the Stage 2 review

The jury gave one main criticism:

> "The main limitation is scale: synthetic logistic data, two clients of 400
> records and one seed per configuration. Repeated training runs and a richer
> dataset would firm up the privacy conclusions."

We made the data larger and we ran everything three times.

| Before | Now |
|---|---|
| 2,000 patients, 800 used for training | **6,000 patients, 2,400 used for training** |
| 10 medical measurements per patient | **13 measurements.** We added HbA1c, heart failure and kidney disease |
| 1 run | **3 runs.** Each run uses a different split of the patients |
| A range from one run | **An average and a spread across the 3 runs**, on every number and chart |

Three new facts came from this work.

- **The conclusions hold.** The spread between runs is very small, between 0.001
  and 0.007. The DP models are at chance level in all three runs. The careless
  model leaks in all three runs.
- **More data makes the leak smaller.** With 800 training records the attack
  scored 0.655. With 2,400 records it scores 0.586. Small groups of patients are
  the real risk.
- **More data also makes privacy cheaper.** At the old size, DP cost 2.5% of the
  accuracy. At the new size it costs 0.6%. This is a reason for many hospitals to
  join, not few.

---

## What we found

Each number is the average of 3 runs. The value after the ± sign is the spread
between those runs. A small spread means that the result is stable.

The best score that **any** model could reach on this data is **0.7844**. We can
calculate this exactly, because we made the data ourselves. We compare every
model against that limit. Thus nobody can say that we chose a weak model to
compare against.

| Model | Accuracy (AUC) | Percent of the best possible | Attack score (AUC) | Privacy promise |
|---|---|---|---|---|
| Trained without care | 0.711 ± 0.007 | 90.7% | **0.586 ± 0.001** | none |
| Trained with care | 0.776 ± 0.007 | 98.9% | 0.512 ± 0.003 | none |
| DP, epsilon = 8 | 0.773 ± 0.007 | 98.5% | 0.507 ± 0.006 | yes |
| DP, epsilon = 3 | 0.771 ± 0.007 | 98.3% | 0.506 ± 0.007 | yes |
| DP, epsilon = 1 | 0.766 ± 0.005 | 97.7% | 0.506 ± 0.007 | yes |
| DP, epsilon = 0.5 | 0.753 ± 0.005 | 95.9% | 0.506 ± 0.005 | yes |

Read the table like this. In the accuracy column, higher is better. In the attack
column, **0.5 is the safe value**, because it means the attacker does no better
than a guess. Only the first row is clearly above 0.5.

**Four results that we can defend.**

1. **Poor training loses privacy and accuracy together.** The careless model is
   the only model that leaks. It is also the least accurate model, at 90.7% of
   the best possible score against 98.9% for the careful model. Privacy and
   accuracy are not opposite goals.

2. **More data reduces the leak, but does not stop it.** The attack score fell
   from 0.655 to 0.586 when we added patients. But 0.586 is still far above the
   safe value of 0.5, and our audit still proves that a real leak exists.

3. **The common measure hides the worst attack.** The four best attacks look
   almost equal on the average score. But we must look at the attacks that are
   almost never wrong. There, the LiRA attack identifies 1.89% of the training
   patients, and the next best attack identifies 1.11%. LiRA finds **1.7 times
   more people** for the same number of false alarms.

4. **Privacy gets cheaper as the group of patients grows.** The cost of
   epsilon = 3 fell from 2.5% of the accuracy to 0.6%. The noise stays the same
   for each training step, but the amount of real signal grows.

**Some patients are at more risk than others.** The attack finds patients with 5
or more other illnesses most easily, at a score of 0.701. Next are patients over
75 years old at 0.661, then patients who did return to hospital at 0.658. For
most other patients the score is near 0.59. The model must memorise the records
that are not usual, because it cannot learn a general rule for them. Thus the
most ill patients carry the most risk. The group with 5 or more illnesses is
small. It has near 95 people in each run. Thus we give this as a direction, not
as an exact value.

### We tested our own privacy code

To use a privacy library is not proof that the privacy works. So we turn the
result of each attack into a measured epsilon value. Then we compare it with the
epsilon that the library promised.

```
model trained without care   no promise; the attack proves a leak at epsilon >= 2.36
model trained with care      no promise; the attack proves almost nothing, epsilon >= 0.01
DP, epsilon = 8              promised 7.99   measured 0.00   correct
DP, epsilon = 3              promised 2.99   measured 0.00   correct
DP, epsilon = 1              promised 0.99   measured 0.00   correct
DP, epsilon = 0.5            promised 0.50   measured 0.00   correct
```

This test can only find a fault. It cannot prove that the promise is exact. If a
measured value went **above** a promised value, the code would be wrong. This
never occurs. The test does find the unprotected model at epsilon >= 2.36, thus
we know that the test works.

---

## Why the law makes this design necessary

Federated learning is not a preference here. The law requires it.

| Law | What it requires | What we do |
|---|---|---|
| **Federal Law No. 2 of 2019** (UAE Health Data Law) | You must not store, process or transfer UAE health data outside the State | Patient records never leave their own hospital. Only protected settings move. |
| **Ministerial Resolution No. 51 of 2021** | Gives a small number of exceptions to that rule | We do not need an exception, because no data crosses a border. |
| **Federal Decree-Law No. 45 of 2021 (PDPL), Art. 20** | You must encrypt the data and remove the names | Removal of names is not enough. Our attack proves it. DP is stronger, because it limits what an attacker can learn even with extra information. |
| **PDPL Arts. 22 and 23** | Control transfers to other countries | Not used, because we transfer no personal data. |

The measured attack is the evidence for row three. The model itself can release
information. To remove the name from the record does not stop this.

---

## How to run it

```bash
nix-shell                            # NixOS only: makes a system library visible
pip install -r requirements.txt
python run_all.py                    # 3 runs, about 25 minutes
python run_all.py --fast             # skips the slow attack, about 4 minutes
python tests/test_pipeline.py        # 13 checks on the output files
```

You can also run one stage on its own, for example
`python src/train_baseline.py --seed 1`.

We set every random seed, so the numbers repeat exactly. We also fix the version
of each software package. A new release of that library can report a different
epsilon for the same settings.

**What each run changes.** The 6,000 patients stay the same, because a hospital
has the patients that it has. Each run changes which patients train the model.
Thus the answer that the attacker looks for changes in every run. The ± values
report the spread over those runs.

---

## What is in each file

| Path | What it does |
|---|---|
| `src/generate_data.py` | Makes the two hospital datasets and the splits |
| `src/train_baseline.py` | Trains the two ordinary models: careless and careful |
| `src/train_dp.py` | Trains the federated models that use differential privacy |
| `src/lira.py` | Runs LiRA, the strongest known attack |
| `src/mia_attack.py` | Runs all the attacks, tests the privacy code, draws the charts |
| `src/build_deck.py` | Builds the submission slides from the results |
| `data/splits_seed<n>.csv` | Says which patients train the model in each run |
| `data/oracle.csv` | The true answer. It is never given to a model |
| `runs/seed<n>/` | Everything that one run produced |
| `results/` | The averages over the runs, and the charts |

---

## What we did not prove

- The patient data is synthetic. Real medical data is harder to predict.
- There are only two hospitals, with 1,200 training records each. A real group
  would be larger.
- The 3 runs measure the spread between runs. They do not measure the spread
  between different groups of patients, because the patients do not change.
- The privacy test gives a wide lower limit, not an exact value.
- The strongest attack trains extra copies of the model, called shadow models.
  They show the attacker what a normal score looks like. We use 32 copies for each
  model in each run. Other researchers use 64 to 256. We chose more runs instead
  of more copies.
- We do not use secure aggregation. Our privacy claim does not need it, because
  each update is already protected. A production system must still add it.
- We did not compare DP against careful training directly. The gap between them
  is small. We can show only that the strongest attack stays above chance against
  careful training, and does not against DP.

---

## For technical reviewers

Four choices control whether the numbers mean anything.

**One split per run, written once.** `data/splits_seed<n>.csv` is the only record
of which patients train the model. Every trainer reads it. A test proves that no
two runs use the same split.

**Validation records are not in the attack set.** The model used those records to
decide when to stop, but never learned from them. They belong in neither group,
so we leave them out.

**Only one thing changes at a time.** The split, the scaling, the class weights,
the architecture and the seed are the same for every model. The ordinary models
differ only in regularisation. The DP models differ only in the privacy method.

**We rank the attacks on the average, not on one run.** To choose the best attack
in each run and then take the average would report the highest of several noisy
values. We take the average first, then rank.

---

## Sources

- Shokri et al., *Membership Inference Attacks Against Machine Learning Models*, IEEE S&P 2017
- Yeom et al., *Privacy Risk in Machine Learning*, CSF 2018
- Carlini et al., *Membership Inference Attacks From First Principles*, IEEE S&P 2022
- Jagielski et al., *Auditing Differentially Private Machine Learning*, NeurIPS 2020
- Kairouz et al., *The Composition Theorem for Differential Privacy*, ICML 2015
- Abadi et al., *Deep Learning with Differential Privacy*, CCS 2016
