# Why the Benders Loop Will Hit Feasibility Cuts (plain-language)

_Companion to `../doc/task1_allow_enforce.md`. This is the intuition, not the formal
argument. It explains what "the loop will hit feasibility cuts" means, why it is
true for **this** model, and what it implies for the implementation._

## The setup, in one picture

The loop has two players passing a problem back and forth:

- **The quantum side (master)** picks the *discrete* choices `z` — for each time
  slot: charge or discharge? import or export? which SoC band? serve the outage or
  not? It picks a full combination of on/off switches.
- **The classical side (Gurobi)** takes that fixed combination and figures out the
  *actual power numbers* `x` — how many kW to import, to charge, etc. — that are
  cheapest **while respecting physics**.

The point of Benders is that after Gurobi solves, it hands a **note back to the
master**: "here's what that choice actually costs / here's what you got wrong."
That note is a **cut**. Next round the master is smarter. Cuts are how the two
sides communicate.

## Two kinds of note (cut)

There are exactly two things Gurobi can report back:

1. **"That switch combo works — and here's its true cost."** → an **optimality
   cut**. A *price tag*. "This configuration is legal, but it's more expensive than
   you thought, so stop underestimating it." The master keeps the option but revises
   its cost upward.

2. **"That switch combo is *impossible*. No set of power numbers satisfies physics
   here."** → a **feasibility cut**. A *ban*, not a price tag. "Never propose this
   combination again."

So **"the loop will hit feasibility cuts"** means: **the quantum side will sometimes
propose switch-combinations that are physically impossible, and Gurobi will bounce
them back with a ban rather than a price.**

## Why "impossible" can even happen — the intuition

The master only checks the **cheap, obvious rules** when it proposes a `z`: "don't
charge and discharge at once," "pick exactly one SoC band," etc. Those are baked
into the Grover mixer — combinatorial hygiene, checkable without touching physics.

But the master **cannot see the physics**. It doesn't know whether the battery has
enough energy, or whether the numbers can be made to balance. Only Gurobi knows. So
the master can hand over a combo that *looks* fine on paper but is physically dead on
arrival. That gap — "structurally legal but physically impossible" — is exactly what
a feasibility cut catches.

The plan (`QC_Ansatz_07-02.md`) hoped this would basically never happen, via the
escape hatch: "worst case, set all power flows to zero — doing nothing is always
legal." If that held, every combo would have the do-nothing fallback, nothing would
be impossible, and you'd only ever get price-tag cuts. **For this model the
do-nothing fallback does not exist.** That is the finding.

## Why this model *does* hit feasibility cuts — three reasons

**Reason 1 — the "serve the outage" switch is a promise, not a permission.**
When the grid is down and the master flips `y_t = 1`, it *promises* "the battery
alone fully covers the load this slot." Hard equality. If the battery is nearly
empty or throttled, it physically **cannot** deliver that. Promise broken →
impossible → feasibility cut. (The one the plan already knew about.)

**Reason 2 — the SoC-band switch is also a promise.**
"Slot 5 is in the *high* band" promises "charge level above 900 kWh at slot 5." But
charge level isn't a free dial — it's the running total of all charging/discharging
before it, from a fixed start. If there isn't enough time or power to climb above 900
by slot 5, that band is a promise the trajectory can't keep → impossible →
feasibility cut. The master picks the band blindly; only physics knows if it's
reachable.

**Reason 3 — the subtle one — "do nothing" is actually illegal here.**
Whenever the grid is up there's a **hard balancing equation**: everything flowing in
must exactly equal the load. The master's switches are "*at most one* direction on"
— nothing forces at least one source **on**. So the master can legally flip
*everything off*. But if the building needs 200 kW and the sun isn't providing it,
"everything off" gives 0 kW into a 200 kW demand. `0 = 200` is false. Impossible →
feasibility cut.

The deep reason is one word: the balance is an **equals** (`=`), not an
**at-least** (`≥`). "Do nothing" only works when the leftover can quietly vanish. An
equals-constraint lets nothing vanish — every kW must be accounted for — so idleness
during a shortfall is forbidden.

## What this means for the implementation

1. **You cannot skip the infeasible branch in Schritt 3.** Gurobi *will* sometimes
   return INFEASIBLE. Your code must expect it, read the **Farkas certificate** (the
   proof of impossibility), and turn it into a feasibility cut. Optimality-cuts-only
   will crash or misbehave the first time the master proposes a dead combo.

2. **Two cut-machines, not one.** Optimality cuts (price tags → sharpen `H_C`'s
   diagonal) *and* feasibility cuts (bans → delete that state from the mixer, or add a
   large penalty in `H_C`). Both paths must exist.

3. **Extra iterations, not lost correctness.** Feasibility cuts aren't a bug — they're
   the loop working. Each one spends a round teaching the master that a combo is
   impossible instead of improving cost. Expect more iterations; still converges to the
   right answer, just chattier.

4. **A calmer loop is a modeling change, not an algorithm change.** You could make
   "do nothing" legal again by forcing exactly one source on, or by adding a small
   curtailment/slack continuous variable so the balance becomes a `≥` that absorbs
   slack. Then Reason 3 disappears (1 and 2 stay). Optional; **not** recommended for
   the PoC — see the note below.

5. **It confirms the model is combinatorially "interesting."** A model where every
   combo trivially works is a boring test — the discrete choices barely matter. Genuine
   impossible combinations mean the master's job is real, a good property for
   demonstrating the mechanism.

## Note on keeping the equality (recommended)

Upholding the hard power-balance equality is the right default:

- It is the **faithful physics** — `P^imp`/`P^exp` already *are* the grid slack;
  there is no extra unmodeled place for power to go.
- Relaxing it lets the LP **silently shed power**, which changes what is optimized and
  can **mask real infeasibility** — the very signal Benders needs.
- The equality's duals are the **true shadow prices** of the balance (energy /
  demand-charge sensitivity), which the optimality cuts depend on.
- The infeasibility it produces is **handled by design** (feasibility cuts), not a
  defect to paper over.

Only relax if, at larger scale, the feasibility-cut volume becomes a **measured**
performance problem — a future-work knob, not a PoC concern.

**Bottom line:** "the loop will hit feasibility cuts" = the quantum side will keep
proposing switch-settings that are physically impossible, Gurobi will keep bouncing
them back as bans, and the implementation must be built to receive and encode those
bans — because in this model, unlike the plan's optimistic default, "just do nothing"
is not a legal fallback.
