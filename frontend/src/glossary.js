// The vocabulary is the real one. What replaces a translation layer is
// explanation in place: every technical term carries its definition where it
// appears, densely, because the fastest way to learn a word is to meet its
// meaning every time you meet the word.
//
// DENSE MEANS AT EVERY APPEARANCE, not first-appearance-only. A chemist does
// not read this app top to bottom; they open the tab that has the number they
// need. A definition that only appears on the screen they did not open is a
// definition that is not there.
//
// Every entry answers three things where it can: what it IS, what it is FOR,
// and what it means when it moves. The third is the one that gets skipped and
// the one that makes a term usable.
export const TERMS = {
  // ── the objective ────────────────────────────────────────────────────
  CRF: 'Chromatographic Response Function - the single number being maximised. ' +
       'CRF = n_clean_peaks x (1 - hump_time_fraction)^2. A high count of ' +
       'well-separated peaks is good; time under an unresolved hump is bad, and ' +
       'the square makes the penalty bite only once the hump takes a real share ' +
       'of the run.',
  'clean peak': 'A resolved peak the prominence search found, sitting outside any ' +
       'unresolved region. These are what the objective counts.',
  shoulder: 'A pick the second-derivative search found but the prominence search ' +
       'did not - partial separation, riding on the flank of a neighbour. Recorded, ' +
       'not counted by the objective.',
  'on-hump': 'A pick inside an unresolved region. Excluded from the clean count, ' +
       'because a peak on a raised floor is not separated from what is under it.',
  hump: 'UCM - unresolved complex mixture. A broad stretch where the signal never ' +
       'returns to baseline and many low-prominence peaks ride on a raised floor. ' +
       'The objective penalises the SHARE OF THE RUN spent under one.',
  'hump_time_fraction': 'The share of the run spent under a hump. Total hump time ' +
       'divided by total run time.',
  isocratic: 'A gradient that does not change: the same %B from start to finish. ' +
       'It is a legitimate method and deliberately kept inside the design space, ' +
       'so the optimiser is allowed to propose one.',

  // ── the loop ─────────────────────────────────────────────────────────
  'cold start': 'The spread-out methods run before any model exists. They are not ' +
       'chosen by the optimiser - they exist to give it a first look at the space.',
  optimise: 'The phase after the cold start, where the model - not a fixed list - ' +
       'chooses the next method.',
  'acquisition function': 'qLogNoisyExpectedImprovement. It scores every candidate ' +
       'method by how much improvement over the best result so far it expects, ' +
       'accounting for the fact that the best result is itself a noisy measurement. ' +
       'It is what turns a prediction into a decision about what to run next.',
  'expected improvement': 'How much better than the current best a candidate method ' +
       'is expected to score. The optimiser proposes wherever this is largest, ' +
       'which balances trying near the best result against trying where it knows ' +
       'least.',
  incumbent: 'The best method found so far. Everything the optimiser proposes is ' +
       'judged against it.',
  replicate: 'The same method run twice back to back. Two runs at identical ' +
       'settings differ only by noise, which is the only way to measure sigma ' +
       'directly - spatial scatter confounds it with the lengthscales.',
  'within-method SD': 'The spread between the replicates of ONE method. It is the ' +
       'campaign’s only direct measurement of run-to-run noise. Blank where a ' +
       'method was run once, because one run cannot have a spread.',
  sigma: 'The run-to-run noise of the score, in CRF units - how much the same ' +
       'method varies between identical runs. Measured from replicate pairs. Until ' +
       'a pair exists the app uses a placeholder of 1.0 CRF and says so.',
  tied: 'Two replicates scored EXACTLY the same. With an integer peak count that is ' +
       'a real outcome, not evidence of zero noise: it means sigma is smaller than ' +
       'one clean peak, so this pair could not resolve it.',
  dof: 'Degrees of freedom - how many independent pieces of information went into ' +
       'an estimate. Each replicate pair contributes one. A sigma with 1 dof is a ' +
       'number; with 8 dof it is a measurement.',
  'run budget': 'How many UNIQUE METHODS this campaign is allowed to try. One distinct ' +
       'set of parameters costs one, however many times it is injected: repeats and ' +
       'instrument checks spend bench time and no budget. Bench time therefore scales ' +
       'with the replicate count, so every screen that sets or reports a budget also ' +
       'states the injections it implies. When the budget is gone the loop stops ' +
       'proposing - nothing already recorded is affected, and you can extend it or ' +
       'close the campaign.',
  'finished campaign': 'A campaign ends one of two ways: the run budget is spent, or ' +
       'you judge the separation good enough and close it. Closing stops new methods ' +
       'being served; the workbook, the traces and the report all keep working, and ' +
       'it can be reopened if you need more runs.',

  // ── the instrument ───────────────────────────────────────────────────
  'reference run': 'One fixed method re-run on a schedule. Its movement can only be ' +
       'the instrument, so it separates instrument drift from chemistry.',
  anchor: 'The first reference run, measured before anything else. Every later ' +
       'reference is a difference from it, which is why it cannot be added later.',
  cadence: 'How often the instrument-check method comes round - one reference every ' +
       'N proposals. ENFORCED, not advised: when a reference is due the loop will ' +
       'not serve a design method until it has been run.',
  drift: 'Anything outside the four knobs that moved across the campaign - mobile ' +
       'phase, column condition, injected mass, ambient temperature.',
  'control chart': 'The reference score plotted against the position the instrument ' +
       'ran it at, with the anchor and the WATCH and HALT limits drawn on. It is ' +
       'the one chart whose movement can only be the instrument.',
  verdict: 'PASS, WATCH or HALT. PASS: the reference is holding. WATCH: it has ' +
       'moved enough to distrust the incumbent. HALT: it has moved enough that ' +
       'proposing is blocked until the instrument is serviced and re-anchored. No ' +
       'reference at all is a WATCH, not a PASS - unmeasured is not the same as fine.',
  monotone: 'Falling every time, with no recovery. Three consecutive falls is a ' +
       'HALT even if none of them is individually large, because a steady decline ' +
       'is a different thing from scatter.',
  'drift-adjusted': 'The score with the reference’s movement subtracted, so runs ' +
       'from different days can be compared. It assumes the drift is REVERSIBLE and ' +
       'nothing can check that - correcting an irreversibly degraded column gives ' +
       'numbers the instrument can no longer produce. It changes what is reported ' +
       'as best; it never changes what the model is fitted to.',
  'phantom incumbent': 'The raw and drift-adjusted winners are DIFFERENT methods. ' +
       'Part of the raw winner’s lead is WHEN it ran, not what it is - so the ' +
       'campaign may conclude it has converged against a decaying baseline and adopt ' +
       'the second-best chemistry. Re-run the adjusted winner back to back with the ' +
       'reference before trusting either.',
  'tailing factor': 'How asymmetric a peak is - how much it drags on its trailing ' +
       'edge. Rising tailing with steady widths points at surface chemistry (active ' +
       'sites, or a void) rather than at lost efficiency.',
  step: 'A single jump carrying most of the movement, rather than a gradual trend. ' +
       'A step is something that was CHANGED between two runs - a new mobile-phase ' +
       'batch, a swapped column. A slope is something wearing out.',

  // ── the model ────────────────────────────────────────────────────────
  GP: 'Gaussian process - the model behind the proposals. It predicts the score at ' +
       'any method it has not run, AND how unsure it is about that prediction. The ' +
       'second half is what makes it useful: the optimiser can tell the difference ' +
       'between "probably bad" and "no idea".',
  posterior: 'What the model believes AFTER seeing the runs so far - as opposed to ' +
       'the prior, which is what it assumed before any data.',
  prior: 'What the model assumes before it sees any data. Here the priors come from ' +
       'BoTorch and are deliberately left alone: they are what stops a fit on a few ' +
       'dozen points from chasing noise.',
  'posterior mean': 'What the model predicts this method would score.',
  'posterior SD': 'How unsure the model is about that prediction, in CRF units. ' +
       'A new measurement can reduce this; the noise term cannot be reduced.',
  kernel: 'The assumption about how smooth the response is - how much knowing the ' +
       'score at one method tells you about a nearby one. This campaign uses ' +
       'Matern-5/2, which assumes less smoothness than the alternative, because a ' +
       'chromatographic response can change sharply.',
  lengthscale: 'How far you have to move along one parameter before the score ' +
       'changes appreciably, in normalised units where the whole allowed range is ' +
       '0 to 1. SHORT means that parameter matters. Long means it barely moves the ' +
       'score inside the space being searched.',
  ARD: 'Automatic relevance determination - one lengthscale per parameter instead ' +
       'of one for all of them, so the model can discover that the score changes ' +
       'fast along temperature and slowly along duration.',
  collapsed: 'A lengthscale that hit its lower limit. It is not a measurement: it ' +
       'means the fit could not identify a scale along that parameter, usually ' +
       'because there are too few runs or the noise term is too small for the ' +
       'scatter in the data.',
  'evaluation grid': 'A fixed set of candidate methods the model’s uncertainty is ' +
       'averaged over. Fixed and cached on purpose: if the set moved between runs, ' +
       'the learning curve would be an artefact of resampling rather than a ' +
       'measure of learning.',
  'noise floor': 'The smallest run-to-run noise the model is allowed to assume, set ' +
       'from the measured replicate spread. Without it the model explains ' +
       'disagreement between replicates by inventing a very wiggly surface.',
  floor_z: 'The noise floor after conversion into the model’s own units - the ' +
       'measured sigma squared, divided by the spread of the scores so far. If it ' +
       'reaches 1 the replicate noise is as large as the whole campaign’s spread, ' +
       'the posterior is nearly flat, and that is the finding rather than a fault.',
  pooled: 'Combined across several methods. The pooled sigma uses every replicate ' +
       'pair in the campaign rather than one, which is why it gets more trustworthy ' +
       'as the campaign goes on.',

  // ── the picker ───────────────────────────────────────────────────────
  picker: 'The peak picker: the code that turns a raw chromatogram into the two ' +
       'numbers the objective eats - how many clean peaks, and what share of the ' +
       'run is under a hump. Its settings ESTIMATE what is really there; they do ' +
       'not define it, which is why correcting a failed estimate is legitimate.',
  'signal-to-noise gate': 'The prominence threshold, as a multiple of the estimated ' +
       'detector noise. It decides what counts as a peak at all.',
  prominence: 'How far a peak rises above the higher of the two valleys beside it - ' +
       'how much it stands out from its surroundings, rather than how tall it is. ' +
       'A tall peak on a raised shoulder can have little prominence.',
  'hump floor ratio': 'How elevated the signal lower envelope must stay relative to ' +
       'the signal before a region qualifies as unresolved.',
  baseline: 'The drifting zero the signal is measured against, fitted and subtracted ' +
       'before anything is counted. Everything downstream depends on it: a baseline ' +
       'that follows the signal too closely quietly removes the very structure ' +
       'being looked for.',
  arPLS: 'Asymmetric least squares - the default baseline fitter. It fits a smooth ' +
       'line that sits under the peaks. Its stiffness is adjustable; a flexible ' +
       'baseline can bend into a broad hump and absorb it entirely.',
  tuned: 'The picker settings were changed for this one run, because the automatic ' +
       'estimate was wrong on this chromatogram. Allowed and recorded on the row. ' +
       'It is a different act from drawing a hump region by hand.',
  stamped: 'Written onto the run’s own row in the workbook, so the settings that ' +
       'produced a score travel with it. It is what lets a trace be re-measured ' +
       'later exactly as it was measured then.',
  're-score': 'Re-measuring every stored trace under changed picker settings, so the ' +
       'CRF column keeps meaning one thing. It is the only sanctioned way to change ' +
       'the settings mid-campaign; changing them without it leaves the sheet a ' +
       'mixture of two rulers.',
  material: 'A metadata field that says what the runs are measurements OF - sample, ' +
       'column, mobile phase, instrument. Correcting how one is written down is ' +
       'routine; changing which column was actually on the instrument means the ' +
       'earlier and later runs are not the same experiment.',
}

// ── version 2: the model choices made at creation ─────────────────────
Object.assign(TERMS, {
  surrogate: 'The model that stands in for the instrument between runs: it predicts ' +
       'the score at any method it has not run, and how sure it is. Here it is a ' +
       'Gaussian process; the choice is WHICH covariance kernel it uses, made once ' +
       'when the campaign is created and locked afterwards.',
  'Matern-5/2': 'The recommended kernel. It assumes the score is smooth but not ' +
       'infinitely so - twice differentiable - so a response that turns over a ' +
       'short interval does not force the fit to collapse a lengthscale.',
  'Matern-3/2': 'One notch rougher than Matern-5/2: once differentiable, so the ' +
       'model tolerates kinks and learns local structure quickly, at the cost of ' +
       'trusting each run less far from where it was made.',
  RBF: 'The squared-exponential kernel, BoTorch’s own default. It assumes the ' +
       'score varies smoothly everywhere; where the response has an edge it must ' +
       'either smooth it away or collapse a lengthscale to chase it.',
  qLogNEI: 'Log noisy expected improvement - the recommended acquisition. Expected ' +
       'improvement over the best result so far, treating that result as the noisy ' +
       'measurement it is, computed in log space so it keeps a gradient under a ' +
       'flat posterior.',
  qLogEI: 'Log expected improvement against the best OBSERVED score. Simpler than ' +
       'the noisy form, and vulnerable to a lucky high reading becoming a bar the ' +
       'model can never clear.',
  UCB: 'Upper confidence bound: propose wherever posterior mean plus beta times ' +
       'the posterior spread is highest. beta is a dial the analyst sets - high to ' +
       'explore, low to exploit - and nothing measured chooses it.',
  beta: 'The exploration weight in UCB. 2.0 is a conventional balance; below 1 the ' +
       'search sits on the best known region, above 4 it roams the box.',
  LogPI: 'Log probability of improvement: propose wherever ANY improvement over the ' +
       'best observed score is most probable, however small. Greedy - it hugs the ' +
       'incumbent and under-explores.',
  locked: 'Set once, at creation, and never changed underneath a running campaign. ' +
       'A campaign fitted under one kernel and refitted under another is not the ' +
       'same campaign, so the choice is written to the Config sheet and stays.',
  'restarts': 'How many starting points the acquisition optimiser climbs from. More ' +
       'restarts find the true maximum of the acquisition more reliably and cost ' +
       'seconds, not runs.',
  'raw samples': 'How many candidate methods are scored before the best few become ' +
       'restart points. Cheap; it only costs compute.',
})
