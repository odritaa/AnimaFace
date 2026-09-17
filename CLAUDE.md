# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AnimaFace is the "Rostro" (face) module of a larger project called **Anima**
(a TFC / thesis project). It is a statistical pipeline with 5 steps, currently
being built in order:

1. **Espacio de rasgos faciales** — a low-dimensional face-shape space built
   with PCA over the Basel Face Model (BFM). **Done** — see
   `punto1_pca_facial/`.
2. **Puntaje poligénico** — a polygenic score computed from known genetic
   variants (Xiong et al., 2025). Not started.
3. **BRIM** — a model connecting the genetic score to the PCA components,
   validated with leave-one-out cross-validation. Not started.
4. **Calibración** — conformal prediction calibration of BRIM's output. Not
   started.
5. **Salida final** — a calibrated probability distribution over facial
   traits, with a confidence interval per region. Not started.

Each step is added as its own top-level folder, named `puntoN_<descripción>/`
(e.g. `punto1_pca_facial/`), containing its code plus a README documenting
the decisions made for that step. Steps are built and merged one at a time;
do not jump ahead to a later step unless explicitly asked to.

### Hard architectural rule: no LLM in steps 1–5

Steps 1 through 5 are pure statistics — trained or computed from the
project's own data, and validated with statistical methods (ΔR² with
bootstrap, empirical coverage). **No LLM may replace or intervene in any of
these steps.** If an LLM is added later, it lives strictly *outside* the
pipeline: it only reads the already-computed output of Step 5 (a structured
JSON: region, ΔR², confidence interval, whether it's significant) to phrase
it in natural language. It never computes, calibrates, or fits anything
itself.

**Consequence for any function you write in these steps:** keep outputs in
a clean, structured format (e.g. JSON-serializable dicts/arrays with clear
field names) so something external can read the result without coupling to
internal code — even though no consumer exists yet.

## Step 1 — `punto1_pca_facial/` (done)

See `punto1_pca_facial/README.md` for full details. Key decisions already
made and **not to be revisited without explicit instruction**:

- **Base file:** `model2019_bfm.h5` (Basel Face Model 2019, classic mask) —
  not the full-head or face12 variants, to avoid heuristically-modeled
  neck/back-of-head regions.
- **Components retained:** 31 (95% of total shape variance).
- **Scale normalization:** intentionally **not** applied (no additional
  Procrustes step on top of the BFM's own space). This is a documented,
  accepted limitation, not an open problem — it's written up in the
  thesis's Limitaciones section, and should stay that way unless the user
  asks to revisit it.

`espacio_rasgos_rostro.py` contains:
- `explorar_bfm(path)` / `cargar_bfm_real(path, grupo_shape)` — inspect and
  load the real `.h5` BFM file (dataset names depend on the BFM version, so
  always explore before assuming names).
- `demo_sintetica()` / `analizar_varianza()` — synthetic-data PCA demo used
  to validate the PCA logic before the real BFM file was available.
- `analizar_varianza_bfm(varianza)` — same variance analysis but for the
  eigenvalues already present in the BFM file (no PCA refit needed).
- `graficar_varianza(...)` — plots the cumulative variance curve.
- `reconstruir_cara(pesos, media, base_pca, varianza, n_componentes)` —
  reconstructs a face from a weight vector expressed in standard deviations
  per component; `pesos=0` returns exactly the mean face.
- `verificar_reconstruccion(...)` — sanity check that must pass before
  trusting the exported feature space in later pipeline steps: zero weights
  reproduce the mean exactly, and +2 SD on PC1 gives a displacement of a few
  millimeters (not something absurd).
- `exportar_espacio_rasgos(...)` — writes the lightweight `.npz` used by the
  rest of the pipeline (`media`, `base_pca` truncated to N components,
  `varianza`, `n_componentes`), so later steps don't need to load the full
  `.h5` file.

`espacio_rasgos_rostro.npz` is the already-exported Step 1 output:
`media` (142317,) = 47439 points × 3 (x, y, z), `base_pca` (142317, 31),
`varianza` (31,), `n_componentes` = 31.

## Commands

There is no build system, test suite, or dependency manifest in this repo
yet. To run the Step 1 script directly:

```bash
python punto1_pca_facial/espacio_rasgos_rostro.py
```

This runs the synthetic-data demo (`demo_sintetica` → `analizar_varianza` →
`graficar_varianza`) and prints instructions for running against the real
BFM file instead. It requires `numpy`, `h5py`, `scikit-learn`, and
`matplotlib`, none of which are pinned anywhere yet (no `requirements.txt`
exists) — install them manually if they're missing.
