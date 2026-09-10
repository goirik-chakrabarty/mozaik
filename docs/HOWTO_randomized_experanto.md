# HOWTO — run the **Randomized Experanto** experiment: simulation → Experanto export

`RandomizedExperanto` is the **production** stimulus-presentation protocol for the neural-foundation-model
data pipeline. It drives the MOZAIK V1 model (`SelfSustainedPushPull`) over an explicit, pre-computed list
of images and videos — a *chunk* — so a large stimulus set can be split into walltime-balanced chunks,
simulated independently, and then exported to Experanto format.

The stimulus/timing logic lives in this repo (`mozaik/mozaik/experiments/vision.py`); the driver, chunk
generation, and export live in the sibling **`mozaik-models/experanto/`** project. This HOWTO is the
general (any-dataset, any-chunk-count) runbook. For the tiny 3-trial smoke case with copy-paste commands
see the workspace-level `HOWTO_test3_sim_then_export.md`.

### The Experanto experiment family

`RandomizedExperanto` is one of three subclasses of **`PixelMovieExperantoBase`** (`vision.py`), which owns
the shared stimulus construction and timing (pre-blank → image → 49 ms post-blank; videos bare). The three
differ only in **how they enumerate the stimuli** to present:

| Class | Enumerates stimuli by | Status |
|---|---|---|
| **`RandomizedExperanto`** | an explicit, pre-computed **chunk JSON** (`chunk_dict_path`) | **production** — this HOWTO |
| `MeasurePixelMovieExperanto` | scanning a whole Experanto **screen directory** (optionally windowed) | ad-hoc measurement |
| `SingleMoviePixelMovieExperanto` | a single **movie file** presented as frame chunks | legacy, no live caller |

Only `RandomizedExperanto` is documented here. Sibling protocols elsewhere in `vision.py`
(`MeasureNaturalImages`, `MeasurePixelMovieFromFile`, …) are **not** part of the Experanto export pipeline
and are out of scope.

There are **two** ways to produce shards:

| | **Workflow 1 — canonical (multi-chunk)** | **Workflow 2 — inline (single-chunk)** |
|---|---|---|
| Shape | sim one chunk per job, **then** a separate export per trial | sim **and** export one chunk in one process |
| Use when | any real dataset (chunks `0..N-1` concatenated per trial) | a single-chunk run you want exported immediately |
| Entry | `export.py` driver (walks chunks) | `run.py --export` (exports the one chunk it simulated) |
| Output | shards under `OUTPUT_PREFIX{trial}/` | shard **next to the datastore** (`<datastore>/experanto/`) |

> **This cluster only:** a config-driven `cluster/submit.sh <conf>` wrapper exists **on disk** for
> convenience, but it is **cluster-specific and not tracked/pushed** (see the closing note). Everything
> below is the portable, in-tree path — a direct `run.py` / `export.py` invocation inside the container —
> which is what this branch (`csng-mozaik-update`) ships.

---

## TL;DR

```bash
# 0) Build the chunk lists once (offline, outside the container is fine — pure Python + pyyaml)
python mozaik-models/experanto/generate_chunks.py \
    --data-root <dataset>/screen/... --output-dir /data/mozaik_chunk \
    --n-trials 20 --n-chunks 12

# 1) SIMULATE each (trial, chunk): one run.py per chunk, TRIAL/CHUNK/CHUNK_DIR select the chunk JSON
#    (loop the array over TRIAL*N_CHUNKS + CHUNK; see "Workflow 1 - Step 1")

# 2) EXPORT each trial: concatenate its chunks into one Experanto shard
python -u export.py <trial> --n-chunks 12          # CHUNK_DIR / OUTPUT_PREFIX / DATASTORE_PREFIX via env
```

---

## Prerequisites

- **Repos / checkouts** (bound into the container by the invocation below, `$PWD`-relative):
  - `mozaik/` (this repo, on `csng-mozaik-update`, → `/mozaik`) — the experiment class + exporter library.
  - `mozaik-models/experanto/` (→ `/project`) — holds `run.py`, `export.py`, `generate_chunks.py`, `param/`.
  - `experanto/` sibling at `../../experanto` (→ `/experanto`).
  - `/mnt/vast-react/projects/neural_foundation_model` (→ `/data`) — dataset, chunks, and output.
- **SIF:** a `freeze_time`-capable image (pyNN 0.13.0). At time of writing that is
  `mozaik-sif/mozaik-opt-qpatch_2026-08-20.sif` — confirm the blessed image in
  `mozaik-models/experanto/experiments/LOG.md`. (Earlier `qpatch_2026-07-14` SIFs predate `freeze_time`
  and will fail on this branch.)
- **Input dataset:** an Experanto **screen** dataset (`screen/meta/*.yml` + `screen/data/*.npy`) — the
  stimuli the experiment reads. Selected by `BASE_PATH`.
- **Chunk files:** `{CHUNK_DIR}/{trial}_{chunk}.json` — built in Step 0.

---

## Step 0 — generate the chunk lists (`generate_chunks.py`)

Run once to turn a stimulus dataset into per-`(trial, chunk)` JSON lists:

```bash
python mozaik-models/experanto/generate_chunks.py \
    --data-root /data/<dataset>/screen/... \   # dir containing screen/meta/*.yml
    --output-dir /data/mozaik_chunk \
    --n-trials 20 --n-chunks 12 \
    --seed 42                                   # base seed (default 42)
```

- Scans `screen/meta/*.yml`, keeping `image` and `video` stimuli.
- **Per trial**, shuffles the full stimulus list with `Random(seed + trial)` — every trial sees the same
  stimuli in a different, reproducible order.
- **Balances** into `n_chunks` by greedy min-heap (each stimulus → the currently-cheapest chunk). Cost is a
  walltime estimate: images carry two blanks; **video cost is per-frame**
  (`video_base + num_frames × video_per_frame`), so one long video does not swamp a chunk.
- Emits `{trial}_{chunk}.json`, each record exactly `{modality, file, trial}` — the only fields
  `RandomizedExperanto.generate_stimuli` reads. It prints each chunk's stimulus count and **estimated
  walltime**; size `--n-chunks` so a chunk fits the job time limit.

The chunk carries **no seed** — per-trial noise is set at launch (see "Seeds").

---

## Workflow 1 — canonical (sim per chunk → export per trial)

### Step 1 — Simulation (one job per `(trial, chunk)`)

Each job simulates one chunk. Select it with `TRIAL` / `CHUNK` / `CHUNK_DIR`; vary `simulation_seed` per
trial for independent noise. Direct in-container invocation (portable; adapt the outer loop / array to your
scheduler):

```bash
cd /mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik
module load apptainer
SIF=$PWD/../mozaik-sif/mozaik-opt-qpatch_2026-08-20.sif

TRIAL=0 CHUNK=0                                   # ← the chunk to simulate
apptainer exec --cleanenv \
  --env PYTHONPATH=/mozaik --env OMPI_MCA_orte_tmpdir_base=/tmp \
  --env TRIAL=$TRIAL --env CHUNK=$CHUNK --env CHUNK_DIR=/data/mozaik_chunk \
  --env BASE_PATH=/data/<input-screen-dataset> \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  --bind "$PWD/mozaik:/mozaik" --bind "$PWD/../mozaik-models/experanto:/project" \
  --bind "$PWD/../../experanto:/experanto" \
  --bind /mnt/vast-react/projects/neural_foundation_model:/data \
  "$SIF" bash -lc '
    unset HTTP_PROXY HTTPS_PROXY FTP_PROXY http_proxy https_proxy ftp_proxy
    cd /project
    taskset -c 0-191 mpirun -n 12 --bind-to none --oversubscribe \
      -x OMP_NUM_THREADS -x MKL_NUM_THREADS -x OPENBLAS_NUM_THREADS -x PYTHONPATH \
      -x TRIAL -x CHUNK -x CHUNK_DIR -x BASE_PATH \
      python -u run.py nest 12 param/defaults \
        results_dir "'"'"'/data/<fresh-out-dir>/'"'"'" \
        simulation_seed '"$(( TRIAL*1000 + CHUNK + 1 ))"' \
        trial'"$TRIAL"'_chunk'"$CHUNK"'
  '
```

- **Output:** one datastore per chunk,
  `SelfSustainedPushPull_trial{T}_chunk{C}_____simulation_seed:{n}/`, under `results_dir`.
- **`results_dir` must be a quoted Python-string literal** — mozaik `eval()`s override values, so a bare
  `/data/...` fails (hence the `"'"'"'…'"'"'"` quoting through the nested shells).
- **`simulation_seed` must be nonzero** (NEST rejects `rng_seed=0`); vary it per trial for per-trial noise.
- To run **all** `(trial, chunk)` pairs, loop `idx = 0 .. n_trials*n_chunks-1` with
  `TRIAL = idx / N_CHUNKS`, `CHUNK = idx % N_CHUNKS` (a SLURM array is the natural fit).

### Step 2 — Export (one shard per trial, after that trial's chunks COMPLETE)

`export.py <trial…> --n-chunks N` concatenates chunks `0..N-1` for each trial into one shard. Env selects
the paths:

```bash
# inside the same container (cd /project), or via apptainer exec as above:
CHUNK_DIR=/data/mozaik_chunk \
OUTPUT_PREFIX=/data/mozaik_data/trial \
DATASTORE_PREFIX=/data/<the results_dir the sim wrote to> \
  python -u export.py 0 --n-chunks 12                 # export trial 0
#   split a long export across jobs:  --chunk-start 0 --chunk-end 6   then   --chunk-start 6 --chunk-end 12
#   screen only:  --screen-only        spikes only:  --spikes-only     images only:  --modality-filter image
```

| Env / flag | Role |
|---|---|
| `CHUNK_DIR` | Directory of chunk JSONs (screen timeline needs **all** `N` chunks even for a spike subset). |
| `OUTPUT_PREFIX` | Shard dir per trial = `{OUTPUT_PREFIX}{trial}` → `responses/` + `screen/`. |
| `DATASTORE_PREFIX` | Dir to resolve datastores in; each is found by glob `SelfSustainedPushPull_trial{T}_chunk{C}_____*`. |
| `SHEET_NAMES` | Comma-separated sheet subset; unset = **all recorded sheets** (multi-sheet default). |
| `--n-chunks` / `--chunk-start` / `--chunk-end` | Which chunks to fold in (resume/append across jobs). |
| `--batch-size` | Chunks held in memory before flushing. |

> **Datastore resolution:** the export globs `SelfSustainedPushPull_trial{T}_chunk{C}_____*` under
> `DATASTORE_PREFIX`. If that dir has **two** matches for one `(trial, chunk)` it raises `Ambiguous
> datastore` — point `DATASTORE_PREFIX` at a dir with exactly one datastore per `(trial, chunk)`.

**Output:** `{OUTPUT_PREFIX}{trial}/responses/{spikes.npy, meta.yml}` (flat float64 spike seconds + N+1 CSR
`spike_indices`) and `{OUTPUT_PREFIX}{trial}/screen/{combined_meta.json, timestamps.npy, meta/*, data/*}`.

---

## Workflow 2 — inline (one job: simulate + export a single chunk)

`run.py … <run_name> --export` runs the sim, then (rank 0 only, after `data_store.save()`) exports the
just-simulated chunk to a **full shard next to the datastore**, reusing the same exporter library. No
separate export job; single chunk only (multi-chunk datasets → Workflow 1).

Use the Step-1 invocation above and append `--export` to the `run.py` line:

```bash
      python -u run.py nest 12 param/defaults \
        results_dir "'"'"'/data/<fresh-out-dir>/'"'"'" \
        simulation_seed 1000 \
        trial0_chunk0_inline --export
```

- `--export` is stripped from `argv` before the mozaik CLI parses it, so it can sit at the end.
- The chunk it exports is the one `TRIAL`/`CHUNK`/`CHUNK_DIR` selected — same env as the sim.
- **Output:** `<datastore>/experanto/{responses,screen}` — identical shard format to Workflow 1.

---

## Environment knobs (the experiment driver)

`create_randomized_experanto(model)` (`mozaik-models/experanto/experiments.py`) reads:

| Env var | Default | Role |
|---|---|---|
| `TRIAL` | `0` | Trial index; selects the chunk JSON and (at launch) the noise seed. |
| `CHUNK` | `0` | Chunk index within the trial. Reads `{CHUNK_DIR}/{TRIAL}_{CHUNK}.json`. |
| `CHUNK_DIR` | `/data/mozaik_chunk` | Directory of chunk JSONs. |
| `BASE_PATH` | historical single-session dataset | Input Experanto **screen** dataset the stimuli are read from (e.g. a P4 subset from `materialize_subset_screen.py`). |
| `STIM_WIDTH` | `11` | Presented image's longest-axis extent in **degrees** (not the visual field). E2 Cadena runs use `6.7`. |

---

## The script chain

```
run.py                                   (mozaik-models/experanto — entry point)
  run_workflow("SelfSustainedPushPull", SelfSustainedPushPull, create_randomized_experanto)
      → build model
      → create_randomized_experanto: reads TRIAL/CHUNK/CHUNK_DIR/BASE_PATH/STIM_WIDTH
            → RandomizedExperanto(chunk_dict_path = {CHUNK_DIR}/{TRIAL}_{CHUNK}.json)   ← mozaik/experiments/vision.py
                  → generate_stimuli(): for each {file, trial} in the chunk,
                        _append_meta_stimulus(): image → pre-blank + image + 49 ms post-blank;  video → bare
      → present stimuli, record spikes → data_store.save()  (rank 0)
      → (if --export and rank 0) export_datastore_inline(...)                            ← Workflow 2 hook

export.py <trials> --n-chunks N          (mozaik-models/experanto — Workflow 1 driver)
      → run_experanto_export(...)         ← mozaik/mozaik/meta_workflow/experanto_export.py (generic driver)
            resolve datastore per (trial, chunk) by glob → load → exporter library
                  → mozaik/mozaik/tools/experanto_export.py  (MozaikTrialExporter + MozaikScreenExporter)
                        → Experanto shards under {OUTPUT_PREFIX}{trial}/
```

Both entry points call the **same** exporter library in the `mozaik` package — one export code path.

**Timing / sync invariant:** each image is `pre-blank → image (~497 ms) → 49 ms post-blank`; videos are
`num_frames × 35 ms`, bare. `POST_BLANK_MS = 49` is defined in `PixelMovieExperantoBase`
(`mozaik/experiments/vision.py`) and **mirrored** in the exporter — the spike and screen timelines share one
clock, so keep the two equal (`responses/meta.yml:end_time == screen/timestamps.npy[-1]`).

---

## Seeds (three-stream)

`param/defaults` uses `model_seed` / `simulation_seed` / `experiment_seed`:

| Seed | Holds | Varies |
|---|---|---|
| `model_seed=1023` | network identity (connectivity, positions, weights, sampling, stimulus order) | fixed across trials |
| `experiment_seed=0` | experiment-level RNG | fixed |
| `simulation_seed` | NEST kernel noise | **override per trial** (nonzero) for independent noise, same network |

Per-trial noise is set on the `run.py` CLI (`simulation_seed <n>`), **not** in the chunk JSON. The sim is
bit-reproducible under fixed seeds. The seed refactor changed noise bit-for-bit vs the old
`lgn_stepcurrentsource_noise_seed` scheme, so current runs will **not** byte-reproduce pre-refactor
datastores — expected, not a regression.

---

## Verify the export

- **PSTH notebook:** `mozaik-models/experanto/notebooks/verify_psth_export.ipynb` — point its config cell at
  a shard dir and Run All: stimulus-locking (§1–6) and export-vs-datastore PSTH parity (§7, build
  `datastore_psths.npz` first via `analysis/compute_psth_datastore.py`).
- **Structural / invariant checks:** compare `responses/spikes.npy`, `responses/meta.yml` CSR
  `spike_indices` (N+1), `screen/timestamps.npy`, `screen/combined_meta.json`, `screen/data/*.npy` against a
  reference; assert `responses/meta.yml:end_time == screen/timestamps.npy[-1]`.
- **Golden gate:** sanity-gate against `docs/plan/audit/golden/P1.json` before/after any sim/export change.

---

## Note — the `cluster/` wrapper is local-only

A `cluster/submit.sh <conf>` config-driven launcher (one `.conf` per experiment: `sim-{test3,prod}`,
`export-{test3,prod}`) exists **on disk** in this working copy and wraps exactly the invocations above
(`ARRAY` index → `TRIAL = idx / N_CHUNKS`, `CHUNK = idx % N_CHUNKS`). It is **specific to this cluster and
deliberately not tracked/pushed** (kept out via the repo's local `info/exclude`; the cluster runner code was
removed from the tree in `52b6a8e`). Treat it as a local convenience — the portable, reproducible launch
path is the direct `run.py` / `export.py` invocation documented here.
