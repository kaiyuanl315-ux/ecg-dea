# Exploratory Fusion-DL waveform attribution analysis

Plan fixed on 15 September 2026 before new prediction or attribution results.

## Existing study and question

Reuse the current 72-hour mortality cohort, patient partitions, labels, ten-member Fusion-DL weights and saved predictions. This is a post hoc exploratory supplement: how do ECG inputs contribute to the fixed ensemble score, conditional on the encounter's observed processed clinical inputs? No training, threshold updating, primary-result replacement, phenotype inference, or mechanistic claim is planned.

## Inputs and population

Use the current preprocessed archive directly, with ID-verified mappings to its waveform, tabular and label arrays. The sampling specification fixes 191 encounters from 190 patients: up to 32 encounters in each mortality-status by three-risk-stratum cell (all 31 mortality-positive low-risk encounters). Seed 20260915. Summaries will be reported by cell; this balanced sample does not estimate overall-population attribution. Select one display case nearest each sample cell's median saved score before examining attributions.

## Model and prediction verification

Strictly load all ten checkpoint states using the recovered matching model definitions; record checksums and versions. Use evaluation mode with batch normalization frozen and dropout off. Compare per-member and mean-sigmoid predictions against saved per-record scores, initially using the prior absolute 5e-5 plus relative 1e-4 tolerance. Evaluate all sampled cases on CPU and MPS to distinguish numerical-backend differences; also check exact clinical-threshold classifications. Any relaxed tolerance or unresolved difference must be explained and recorded, not silently treated as an exact match.

## Attribution target and computation

Target F(x,c) = mean_k sigmoid(f_k(x,c)), before logistic recalibration. Hold all twelve tabular inputs, including observation masks, fixed at c. Attribute only the ECG from b to x using Gauss-Legendre Integrated Gradients. Average signed member attributions before taking absolute values. Start with 128 integration nodes and retain per-member output and baseline predictions. Primary ECG baseline b is zero in the verified model-input scale. Parameters are frozen; input gradients remain enabled.

## Numerical and sensitivity checks

- Unit-check the integrator against an analytic linear example.
- Check completeness against F(x,c)-F(b,c) for every case. A candidate numerical tolerance is 0.0005 + 0.02 times the absolute output difference; report every residual. Refine any failing cases with 256, 512, 1024, and if necessary 2048 nodes, preserving the history. Do not drop difficult cases.
- Compare 128 and 256 nodes in two score-spread cases per cell selected without examining IG (12 cases), reporting signed-attribution differences, completeness and lead-share rank correlations. Further refinement is allowed if integration is unstable; the change will be recorded.
- In the same 12 cases, use an alternative baseline that holds each lead at its own temporal mean, and compare lead shares with zero-baseline IG. This checks baseline dependence, not biological validity. Baseline sensitivity is reported even if rankings differ.
- On a small prespecified subset, compare CPU and MPS attribution calculations if MPS is used for the final computation.

## Summaries and figure

For each case, sum absolute ensemble IG across time within each channel and divide by the sum across all channels/time points. Report group medians (and source-data means/IQRs), with no population inference or multiplicity claims. Verify channel names and input amplitude units against original ECG records before labeling the figure. Display the six preselected genuine 10-second waveforms in a fixed lead, with clearly distinguished signed attribution colors. Do not average waveforms across people or use signal-to-noise panels.

## Handoff

Deliver one Figure S8 with a precise legend, numerical QA, source arrays and reproducible scripts. Update a new manuscript-package version only after the provenance and numerical checks. Preserve prior package files and the primary statistics.

## Recorded diagnostic addition after initial sensitivity results

The 12-case alternative-reference analysis at 128 nodes found maximum lead-share changes of 12.275 and 6.759 percentage points in sample rows 0 and 100. To determine whether this large baseline dependence could reflect quadrature error, repeat the temporal-mean reference at 256 nodes for those two cases. Selection for this numerical diagnostic occurred after observing the baseline comparison; it was not part of the original frozen 12-case sampling rule. Keep all original results and report both the reference dependence and the diagnostic comparison. Do not replace the primary zero-baseline figure or select new display examples.
