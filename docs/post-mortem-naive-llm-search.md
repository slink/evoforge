# Post-Mortem: Why Naive LLM Evolutionary Proof Search Didn't Work

**Date:** 2026-03-06
**System:** evoforge v1, Lean 4 backend
**Target:** `norm_le_one` — prove that positive definite functions have norm ≤ 1

## What We Built

An evolutionary engine that uses LLMs to mutate tactic proofs. The loop: seed a population of 30 proof attempts, evaluate each step-by-step in a Lean REPL, select the fittest, mutate them (4 cheap structural operators + 2 LLM-powered operators), repeat. Three-tier verification catches false positives: REPL step-by-step → REPL cmd → full `lake env lean` compilation.

We expected the LLM to gradually improve proofs over generations, guided by fitness feedback and search memory. The evolutionary approach should explore diverse proof strategies while selection pressure drives toward completeness.

## What Actually Happened

```
10 generations, 476 evaluations, 14 minutes
Best fitness: 0.7 (a false positive — failed lake verification)
Zero verified proofs
Diversity: constant at 1.007 for all 10 generations
```

The system found proofs that *looked* promising — `suffices ‖φ ξ‖ ^ 2 ≤ 1` appeared in ~10 different variants — but none survived full verification. The population was technically diverse by our entropy metric (different first tactics and proof lengths) but semantically stagnant: every generation produced minor variations of the same few broken strategies.

## Root Causes

### Three Silent Bugs

**1. Crossover was dead code.** Two of our six mutation operators — `LLMCrossover` and `SplicePrefixes` — were no-ops. The engine never set `guidance_individual` on the mutation context, so crossover always fell back to plain mutation. `SplicePrefixes` tried to parse the search memory text (a human-readable string like "Successful patterns: intro x...") as a tactic sequence, which obviously returned `None`. One-third of our mutation budget was wasted on operators that returned the parent genome unchanged.

**2. Adaptive weights never updated.** The mutation ensemble tracked operator statistics (`update_stats`), but the engine never called it. Config said `schedule = "phased"` but all six operators stayed at uniform 1/6 weight forever — including the two broken ones. The system couldn't learn that `TacticSwap` (randomly swapping adjacent tactics) almost never produces valid Lean proofs.

**3. Reflection was thrown away.** Every 10 generations, the engine asked the LLM to analyze the population and suggest strategies. The LLM returned structured JSON with `strategies_to_try`, `strategies_to_avoid`, and `suggested_temperature`. The engine logged "Reflection response received (1908 chars)" and discarded it. 1908 characters of potentially useful strategic guidance, straight to `/dev/null`.

### The Fitness Function Was Lying

Fitness was `steps_succeeded / total_steps`. This rewarded length, not progress. A 10-step proof where 7 tactics individually parse but close zero goals scores 0.7. A 2-step proof where both tactics close the actual goal scores 1.0 — but a 2-step proof where the first tactic creates 3 sub-goals and the second fails scores 0.5, *even though the first approach was more promising*.

The system learned to evolve long sequences of individually-valid tactics (`intro x`, `simp`, `norm_num`, `ring`, `push_neg`...) that don't actually prove anything. The REPL accepts each tactic, reports the proof state changed, and the evaluator counts it as progress. But the *goals* weren't being reduced — just shuffled.

### The Architecture Was Wrong for the Problem

This is the deepest issue, and it's what the literature should have told us from the start.

**Theorem proving is a tree search problem, not a sequence optimization problem.**

When a proof gets stuck at step 4, the right move is to try 5 different tactics at step 4 and explore each branch. Evolution's approach — generate 30 complete linear sequences, evaluate them, mutate the whole sequence — is an absurdly inefficient way to explore alternatives at a specific proof state. It's like trying to solve a maze by generating random complete paths from start to finish, rather than walking to the first fork and trying each direction.

Every successful LLM theorem prover in the literature uses tree search in some form:
- **HTPS** (Lample et al. 2022): hyper-tree proof search, best-first expansion
- **AlphaProof** (DeepMind 2024): MCTS over proof states with a value network
- **COPRA** (2024): stateful interaction, trying alternatives at each step
- **Goedel-Prover-V2** (2025): Monte Carlo tree self-refinement

We were doing FunSearch-style evolutionary search (which works great for *program synthesis* where the fitness landscape is smoother) on a problem where the fitness landscape is essentially a cliff: proofs either work or they don't, and partial credit for "steps that parse" doesn't correlate with "proof strategies that are close to correct."

### Cheap Operators Were Noise

`TacticSwap` randomly swaps adjacent tactics. `TacticReorder` shuffles a window of 3 tactics. In Lean, tactic order matters — `intro x` must come before `simp [x]`. These operators almost never produce valid proofs. They burn evaluation budget generating garbage that the REPL rejects at step 1, giving fitness 0.0 with no gradient signal. Unlike genetic programming on arithmetic expressions (where swapping subtrees often produces valid programs), tactic proofs have strong sequential dependencies.

## What We Did About It

**Fixed the bugs:** Wired crossover, ensemble stats, and reflection parsing. These alone should significantly improve search quality by making all six operators functional, letting the system learn which operators work, and feeding LLM analysis back into the search.

**Fixed the fitness function:** New formula weights 60% on goal reduction and 40% on step completion. Proofs that close goals now rank higher than proofs that just accumulate valid-but-useless tactics.

**Added tree search:** A best-first `ProofTreeSearch` that expands the most promising proof states, asking the LLM for N candidate tactics at each node. It runs as a hybrid alongside evolution — evolution finds promising proof prefixes, tree search efficiently explores branches from those prefixes.

## Lessons

1. **Test your operators actually run.** We had unit tests for each operator in isolation, but no integration test verifying the engine actually invoked them with correct arguments. The bugs hid because the system *appeared* to work — it just worked badly.

2. **Match your search algorithm to your fitness landscape.** Evolution works when fitness is smooth and incremental improvements are common. Theorem proving has a jagged, cliff-like landscape. Tree search is the right tool because it can efficiently backtrack from dead ends.

3. **Your fitness function IS your specification.** If you reward "steps that parse," you get "long sequences of parsing steps." The system optimized exactly what we told it to optimize — we just told it to optimize the wrong thing.

4. **Read the literature before building.** Every paper on LLM-guided theorem proving uses tree search. We built an evolutionary system because that was the project's framing, but should have added tree search from day one as the primary proof exploration strategy, with evolution as a meta-level strategy for discovering proof *approaches*.
