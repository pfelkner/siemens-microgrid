# QC-Ansatz: Hybrid Quantum–Classical Microgrid Dispatch

_Stand: 05.07.2026_

## Überblick

Wir lösen das Microgrid-Dispatch-Problem hybrid, indem wir die Variablen nach Typ aufteilen:

- **Diskrete Entscheidungen** (Laden/Entladen, Import/Export, served-or-not `y_t`, Asset an/aus), im Folgenden `z` → **Qubits**, gelöst per GM-QAOA (QAOA mit Grover-Mixer).
- **Kontinuierliche Größen** (Leistungs-Setpoints, committed peak), im Folgenden `x` → bleiben **klassisch**, gelöst per klassischem Solver (Gurobi). Sie werden **nicht diskretisiert**.

Statt beides gemeinsam zu lösen, optimieren wir **abwechselnd**: bei fixem `x` die diskreten Variablen (Quantum), bei fixer diskreter Konfiguration die kontinuierlichen Variablen (klassisch), und iterieren.

Damit dieses Ping-Pong garantiert zur **optimalen** Lösung konvergiert (und nicht nur zu einem lokalen Optimum), strukturieren wir es als **Benders-Dekomposition**: der diskrete Teil ist das _Master-Problem_ (per QAOA), der kontinuierliche Teil das _Subproblem_ (per Gurobi). Das Subproblem liefert nach jedem Durchlauf nicht nur `x*`, sondern einen **Benders-Cut** — eine zusätzliche Nebenbedingung, die das Master-Problem für die nächste Runde verschärft. Unter der Annahme, dass beide Subprobleme je Runde optimal gelöst werden, konvergiert das in endlich vielen Iterationen zum globalen Optimum. (Vorbehalt zum QAOA-Teil siehe Abschnitt „Benders-Struktur".)

**Grundvoraussetzung:** Bei fixierter diskreter Konfiguration muss der Raum der kontinuierlichen Variablen konvex sein. Das ist erfüllt, solange das kontinuierliche Subproblem ein LP ist (lineare Constraints: Leistungsbilanz, SoC-Dynamik, Box-Bounds, Grid-Box, Demand Charge) — dann ist der Feasible-Bereich ein konvexes Polytop. Wichtig: es darf sich kein Produkt zweier kontinuierlicher Variablen einschleichen (z. B. nichtlineare Effizienzkurve), sonst kippt die Konvexität. Diese Konvexität ist doppelt tragend: Sie sorgt nicht nur dafür, dass Gurobi das Subproblem exakt löst, sondern ist auch die Voraussetzung dafür, dass dessen Dual-Werte **gültige** Benders-Cuts liefern (starke LP-Dualität). Bei einem nichtkonvexen Subproblem wären die Cuts keine korrekten unteren Schranken mehr und die Konvergenzgarantie fiele.

**Notation:** `x` = kontinuierliche Entscheidungsvariablen, `z` = diskrete Konfiguration (Bitstring aus der QAOA; die served-bits `y_t` sind eine Komponente davon).

---

## Der Loop

### Schritt 1 — Initialisierung

Starte mit einer feasiblen Konfiguration kontinuierlicher Variablen `x` (z. B. random, aber feasibel).

### Schritt 2 — Diskrete Variablen (Quantum)

Gib `x` in die GM-QAOA. Diese besteht aus zwei Bausteinen:

- **Grover-Mixer:** kodiert alle _strukturell_ feasiblen Konfigurationen der diskreten Variablen. Für den PoC brute-force gebaut (jede mögliche Belegung prüfen: feasibel → 1, sonst → 0) und **einmalig** erstellt. „Strukturell feasibel" meint hier die `x`-unabhängigen kombinatorischen Regeln (Lade/Entlade-XOR, Import/Export-XOR, served-bit-Konsistenz).
- **Cost-Hamiltonian `H_C`:** diagonal; der Eintrag für eine Konfiguration `z` ist ihre **direkte** Kosten (die Kostenanteile, die nur von `z` abhängen) **plus** das aktuelle untere Modell der kontinuierlichen Restkosten aus den bisher gesammelten Benders-Cuts (siehe unten). In der ersten Runde gibt es noch keine Cuts; die Restkosten werden mit dem beim Start-`x` erreichten Wert initialisiert. Ab dann wird `H_C` je Runde durch den neuen Cut verschärft — der Mixer bleibt gleich.

Die QAOA liefert einen Lösungsraum möglicher diskreter Konfigurationen. Wir samplen die beste daraus — der Bitstring `z` ist die Belegung der diskreten Variablen.

### Schritt 3 — Kontinuierliche Variablen (klassisch)

Fixiere die diskrete Konfiguration `z` aus Schritt 2. Löse damit das kontinuierliche Subproblem exakt per Gurobi → optimales `x*` für genau diese Konfiguration. Da das ein restringiertes LP ist, bleibt die Lösung per Konstruktion im feasiblen Bereich. Neben `x*` lesen wir die **Dual-/Schattenpreise** des LP aus — daraus wird der Benders-Cut gebildet:

- Ist das Subproblem **feasibel**, liefern die Duals einen **Optimality-Cut**: eine untere Schranke an die kontinuierlichen Restkosten als Funktion von `z`.
- Ist es **infeasibel**, liefert der Farkas-Zertifikat (unbeschränkter Dual-Strahl) einen **Feasibility-Cut**: er schließt genau die `z` aus, die keine feasible Fortsetzung haben.

### Schritt 4 — Loop

Füge den Cut aus Schritt 3 dem Master hinzu (er verschärft `H_C` bzw. den feasiblen Bereich, siehe „Benders-Struktur") und gehe zurück zu Schritt 2. Wiederholen, bis die untere Schranke des Masters und der beste gefundene Zielwert zusammenfallen (Optimalität nachgewiesen) oder das Iterationslimit erreicht ist.

---

## Benders-Struktur (das ist es, was die Konvergenz sichert)

Der Grund, das Ping-Pong als Benders-Dekomposition zu bauen: Eine naive Alternierung (Master minimiert nur die Kosten _beim aktuellen_ `x`) hat keine Konvergenzgarantie und kann zykeln. Benders baut stattdessen über die Iterationen ein **immer genaueres unteres Modell** der kontinuierlichen Restkosten `q(z)` (= optimaler Subproblem-Wert bei fixem `z`) auf und lässt den Master über dieses Modell optimieren. Das garantiert Konvergenz zum globalen Optimum.

**Wie die Cuts in den QAOA-Master eingehen** — im brute-force-PoC angenehm direkt:

- **Optimality-Cut** (feasibles Subproblem): eine affine untere Schranke an `q(z)`. Da wir den Cost-Hamiltonian ohnehin per Hand über alle `z` aufbauen, ist der Diagonaleintrag für `z` einfach: direkte Kosten von `z` plus das Maximum über alle bisher gesammelten Cuts, ausgewertet an `z`. Die Hilfsgröße `η` aus der Benders-Theorie braucht also **kein eigenes Qubit** — sie fällt als punktweises Maximum affiner Cuts in die Diagonale.
- **Feasibility-Cut** (infeasibles Subproblem): schließt die betroffenen `z` aus → im PoC entweder aus dem feasible-State-Vektor / Mixer streichen oder in `H_C` mit einer hohen Strafe belegen.

**Der Feasibility-Cut ist zugleich die Absicherung gegen den einen Bruchpunkt des Loops:** Was, wenn eine strukturell feasible Konfiguration `z` bei den aktuellen Daten _keine_ feasible kontinuierliche Fortsetzung hat? Dann meldet Gurobi in Schritt 3 infeasibel — und genau dieser Fall erzeugt automatisch einen Feasibility-Cut, der `z` künftig ausschließt. Ob dieser Fall überhaupt auftritt, hängt daran, ob die Binärvariablen kontinuierliche Aktivität **erlauben** oder **erzwingen**:

- **Erlauben** (z. B. `charge ≤ M·b`: `b=1` erlaubt Laden ≥ 0, erzwingt aber nichts): „alle Flüsse ≈ 0 setzen" ist fast immer feasibel → jede Konfiguration hat eine Fortsetzung → es entstehen gar keine Feasibility-Cuts. Normalfall bei üblichen Dispatch-Formulierungen.
- **Erzwingen** (z. B. served-bit `y_t=1` erzwingt gedeckte Last): manche Konfigurationen können infeasibel sein → dann greifen die Feasibility-Cuts.

Es lohnt sich trotzdem, das vorab einmal zu prüfen (erlaubt oder erzwingt), weil es bestimmt, ob im Loop überhaupt Feasibility-Cuts anfallen oder nur Optimality-Cuts.

**Vorbehalt:** Die Konvergenzgarantie gilt unter der Annahme, dass _beide_ Subprobleme je Runde optimal gelöst werden. Für das kontinuierliche LP stimmt das (Gurobi). Für das diskrete Master ist der Solver aber die **QAOA — ein Heuristik-Sampler**, der die Master-Optimalität nicht zertifiziert. Die Garantie überträgt sich also nur so weit, wie die QAOA das Master-Optimum je Runde tatsächlich findet. Bei den kleinen PoC-Instanzen ist das unkritisch (man kann den Lösungsraum praktisch erschöpfend sampeln); für größere Instanzen bleibt es ein offener Punkt.

**Anmerkung:** Jede Iteration bringt eine zusätzliche Nebenbedingung (den Cut) ins Master. Diese sauber in Mixer/Cost-Hamiltonian zu kodieren, ist im Allgemeinen Aufwand — aber da wir die effiziente Umsetzung der Nebenbedingungen ohnehin bewusst ignorieren (brute-force-Konstruktion, siehe „Aufgaben"), fällt der Extra-Cut für den Rahmen des Praktikums nicht ins Gewicht: es heißt schlicht, den Diagonal-Cost bzw. den feasiblen Zustandsvektor je Runde neu zu berechnen.

---

## Unsere Aufgaben

1. **Erlaubt/erzwingt-Check (zuerst):** Für jede diskrete Binärvariable klären, ob sie kontinuierliche Aktivität erlaubt oder erzwingt → daraus folgt, ob im Loop überhaupt Feasibility-Cuts anfallen (nur Optimality-Cuts, falls recourse vollständig).
   → **ERLEDIGT** — Ergebnis in `siemens-microgrid/doc/task1_allow_enforce.md`. Kernbefund: recourse ist **nicht** vollständig → es fallen **Feasibility-Cuts** an (nicht nur Optimality-Cuts). Details unter „Umsetzungsstand".
2. **Instanz definieren:** PoC klein halten (`T = 2–3`, diskrete Variablen ≤ ~8×3, je nach Hardware / cuQuantum). Die strukturelle diskrete Feasibility exakt benennen (das, was der Mixer kodiert).
   → **ERLEDIGT** — `qc/instance.py`: `Instance`-Dataclass (`T`, `n_bits` frei konfigurierbar), Bit-Layout `ROLES = (ch, dis, imp, exp, b_low, b_mid, b_high, y)`, vektorisiertes `structurally_feasible()` (XOR-Regeln, SoC-Band One-Hot, Outage-Pinning). Getestet in `tests/test_qc_instance.py`.
3. **Start-`x`:** Skript, das eine feasible Konfiguration kontinuierlicher Variablen liefert.
   → **ERLEDIGT** — `siemens-microgrid/subproblem/feasible_start_x.py` (Sampler) + CLI `scenario_runner.py`. Details unter „Umsetzungsstand".
4. **Grover-Mixer:** feasible-State-Vektor brute-force bauen (feasibel → 1, sonst 0), Erstellung parallelisieren. 0-Einträge streichen → Mixer wird auf dem Rest trivial all-to-all. Mixer- und Cost-Operator als NumPy-Matrizen.
   → **ERLEDIGT** — `qc/grover_mixer.py` + `qc/instance.py`. Feasible Set via Cartesian Product über Per-Slot-Mengen (O(|F|) statt O(2^{8T})). Mixer als Rank-1-Update in `qc/qaoa.py` — nie als vollständige Matrix materialisiert. Dense-Matrizen in `qc/dense.py` nur für Tests.
5. **Cost-Hamiltonian:** `H_C` diagonal aufbauen als direkte Kosten pro `z` plus punktweises Maximum der bisher gesammelten Optimality-Cuts; je Runde neu berechnen, wenn ein Cut dazukommt.
   → **ERLEDIGT** — `qc/instance.py:direct_costs()` berechnet den z-abhängigen Teil (Resiliency-Bonus) vektorisiert über die feasiblen States. Cut-Integration in `H_C`: punktweises Maximum der Optimality-Cuts über die Bit-Matrix in der Diagonale, implementiert in `qc/benders.py`.
6. **QAOA-Durchlauf:** per NumPy-Matrixmultiplikation; QAOA-Winkel für den PoC fest (Ramp-Winkel, kleines `p`) → Lösungsraum → beste Konfiguration `z` samplen.
   → **ERLEDIGT** — `qc/qaoa.py`: `gm_qaoa()` (Subspace-Evolution, Ramp-Winkel, `p=6` kalibriert) + `sample_best()`. CLI-Demo `qc/run_poc.py`: QAOA-Verteilung vs. exaktes Optimum, Round-1-Ansicht des Benders-Masters (nur direkte Kosten, noch keine Cuts). Tests in `tests/test_qc_*.py`.
7. **Klassischer Solver (Subproblem):** Gurobi-Modell für die kontinuierlichen Variablen bei fixem `z`; Rückgabe von `x*` **und den Dual-Werten**.
   → **ERLEDIGT** — `subproblem/subproblem.py`, `solve_subproblem(instance)`; `rhs_affine`-Export ergänzt (RHS als affine Funktion der Master-Bits, selbstgeprüft gegen Gurobi-`Constr.RHS` bei jedem Solve). Formelle Abnahme erfolgt.
8. **Cut-Bildung + -Integration:** aus den Duals den Benders-Cut konstruieren (Optimality-Cut bei feasiblem, Feasibility-Cut bei infeasiblem Subproblem) und ins Master einarbeiten (Diagonale von `H_C` bzw. feasiblen Zustandsvektor updaten).
   → **ERLEDIGT** — `qc/benders.py`: `optimality_cut` (verankerte Cuts `q(z) ≥ q̄ + w·(z − z̄)` aus Duals × `rhs_affine`), `feasibility_cut` (vorzeichennormalisierte Farkas-Cuts, filtern alle States mit demselben Unlösbarkeits-Beweis aus dem feasiblen Zustandsvektor), vektorisierte Auswertung über die Bit-Matrix; Adapter `to_slot_configs` / `build_sub_instance`.
9. **Loop:** die Schritte zu einer Schleife zusammenfügen; Abbruch, wenn untere Master-Schranke und bester Zielwert zusammenfallen (Optimalität) oder Iterationslimit erreicht.
   → **ERLEDIGT** — `qc/benders.py::benders_loop` + CLI `qc/run_loop.py`: ein `z` pro Runde (best-of-shots), Kosten = direkte Kosten + punktweises Maximum der Optimality-Cuts, LB = exaktes Minimum über die verbliebene Enumeration (−∞ bis zum ersten Optimality-Cut), UB = bestes direct+Q, Terminierung nach Gap/max_rounds/infeasibel. `brute_force_optimum` als exakte Referenz.
10. **Simulation Runs + Plots:** 1–2 Runs; Approximation Ratio und Time-to-Solution plotten.
11. **Vergleich klassisch:** gegen die vollständig klassische Lösung (Gurobi-MILP als Ground Truth) prüfen — Ergebnisse sollten feasibel und einigermaßen gut sein.

---

## Umsetzungsstand (Stand: 05.07.2026)

Klassischer Teil: `subproblem/` (Aufgaben 1, 3, 7). Quantum-Teil: `qc/` (Aufgaben 2, 4–6, 8, 9).
Erledigt: **1–9** (vollständig).
Offen: **10/11** (Plots, Vergleich gegen Gurobi-MILP).

### Aufgabe 1 — Erlaubt/erzwingt-Check (erledigt)

Ergebnis: `doc/task1_allow_enforce.md` (+ Klartext-Erklärung `subproblem/why_feasibility_cuts.md`).
Klassifikation der Binärvariablen (Quellen: `classical_solver.py`, `notes/presentations/reductions.tex`):

- **Erlauben** (einseitiges Gate `flow ≤ M·b`, „Fluss = 0" immer zulässig): `b^ch, b^dis, b^imp, b^exp`.
- **Erzwingen** (fixieren eine kontinuierliche Größe, können infeasibel sein): die SoC-Band-Bits
  `b^low/b^mid/b^high` (zwingen `E` in ein Intervall) und `y_t` (served → erzwingt exakt gedeckte Outage-Last).

**Kernbefund (relevant für den Loop):** Der recourse ist **nicht vollständig** → es fallen
**Feasibility-Cuts** an, nicht nur Optimality-Cuts. Grund über `y_t` und die Bänder hinaus:
die Online-Leistungsbilanz ist eine **Gleichung**, die Aktivitäts-Gates aber nur „höchstens eins"
(`≤ 1`). Der Master kann daher in einem Defizit-Slot „alles aus" wählen → das LP ist infeasibel.
**Konsequenz:** Schritt 3 **muss** den Gurobi-infeasibel-Zweig (Farkas → Feasibility-Cut) behandeln;
die Annahme „nur Optimality-Cuts" trifft für dieses Modell **nicht** zu.

### Aufgabe 3 — Start-`x` (erledigt)

Ergebnis: `subproblem/feasible_start_x.py`, Funktion `feasible_configs(instance, n, seed)`.

- **Vorgehen:** fixes `z` einsetzen → das LP auf die **freien Batterieleistungen** pro Slot reduzieren
  (SoC-Trajektorie und Netzflüsse sind daraus abgeleitet, nicht frei), das reduzierte Polytop
  `G·u ≤ h` aufbauen und diverse feasible Punkte per **Chebyshev-Zentrum + Hit-and-Run** ziehen.
  Jeder Sample wird gegen Balance/Bounds/Band/Throttle **verifiziert**.
- Dient zugleich als **Feasibility-Orakel**: hat ein `z` keine kontinuierliche Fortsetzung, wirft es
  `Infeasible` — genau der Feasibility-Cut-Fall aus Aufgabe 1.
- **Verifiziert** am Referenzfall `reference-t3` (die T=3-Instanz aus `doc/conversation.md`): alle
  Samples liegen im handgerechneten reduzierten Polytop (`θ_0 ≤ 72`, SoC-Boden bindet vor der Batterie).
- `T` ist **nicht** fest verdrahtet (`T = len(pv)`), `Δt = Params.dt` — Instanzgröße frei konfigurierbar.

**Bedienung — `subproblem/scenario_runner.py` (CLI, kein Code-Editieren nötig):**

- Szenario wählen: `--scenario NAME` (Registry: `reference-t3`, `night-deficit`, `outage-served`,
  `outage-infeasible`), `--file x.json` (Slot-Schema mit Defaults, Beispiel `sample_t3.json`),
  oder `--csv all_data.csv --slots N` (diskretes `z` heuristisch gefüllt).
- `--solve` löst zusätzlich das Subproblem (Aufgabe 7), `--dump out.json|.csv` exportiert **alle**
  feasiblen Samples, `--save` legt ein editierbares Szenario-Template ab, `--list`, `--selftest`.

> **Hinweis zur Abgrenzung:** Die gesampelten/exportierten Konfigurationen sind feasible **kontinuierliche
> `x`** (Start-`x`, Schritt 1) — Saat für die Loop-Initialisierung bzw. den Startwert der Restkosten in
> `H_C`. Sie sind **nicht** die diskreten Zustände des Grover-Mixers (Aufgabe 4); der Mixer enumeriert `z`.

### Aufgabe 7 — Klassischer Solver (Subproblem) — erledigt

Ergebnis: `subproblem/subproblem.py`, `solve_subproblem(instance)`. Baut das fixe-`z`-LP in Gurobi und
gibt `x*`, `Q(z)` **und die Duals** zurück; bei infeasiblem LP das **Farkas-Zertifikat** (für den
Feasibility-Cut). `z` geht **nur über die RHS** ein (fester Constraint-Satz) → die Duals liefern in `z`
**affine** Benders-Cuts, wie sie Aufgabe 8 braucht. Die Online-Bilanz bleibt bewusst harte **Gleichung**
(Begründung in `why_feasibility_cuts.md`). **Verifikation** an `reference-t3`: das LP-Optimum trifft die
handgerechnete Ecke (Batterieleistungen 72/200/162, `Q(z) = 1929.25`); der Infeasibel-Zweig liefert ein
Farkas-Zertifikat. `rhs_affine`-Export ergänzt (Selbstcheck gegen Gurobi-`Constr.RHS` bei jedem Solve).
Formelle Abnahme erfolgt.

### Aufgaben 8 + 9 — Cut-Bildung und Loop (erledigt)

Ergebnis: `qc/benders.py` (Cuts + Loop) + CLI `qc/run_loop.py`.

- **Ein `z` pro Runde** (best-of-shots aus dem QAOA-Sample); Kosten = direkte Kosten + punktweises Maximum der bisher gesammelten Optimality-Cuts.
- **Verankerte Optimality-Cuts** `q(z) ≥ q̄ + w·(z − z̄)`: Gewicht `w` aus Duals × `rhs_affine` (keine Bound-Duals nötig, da `z` nur über die RHS eingeht).
- **Vorzeichennormalisierte Farkas-Feasibility-Cuts:** filtern alle States mit demselben Unlösbarkeits-Beweis aus dem feasiblen Zustandsvektor; Auswertung vektorisiert über die Bit-Matrix.
- **LB** = exaktes Minimum über die verbliebene Enumeration (−∞ bis zum ersten Optimality-Cut); **UB** = bestes direct+Q. Terminierung nach Gap / max_rounds / infeasibel.
- `brute_force_optimum` als exakte Referenz für Tests.
- `subproblem/` ist jetzt ein Package (`__init__.py`); Imports package-absolut; Scripts via `uv run python -m subproblem.scenario_runner`.
- **Notebook-Visualisierung** (`visualize.ipynb`): Zelle 17 nutzt den echten Cut; neue Loop-Sektion mit Rundentabelle, UB/LB-Konvergenzkurve, Verteilungs-Slider, |F|-Bars und Finale vs. Ground Truth.
- **Verifiziert:** End-to-end-Test gegen Brute-Force-Optimum (`tests/test_qc_benders.py`). Im Demo-Fenster (T=2, Outage): 5 Farkas-Cuts eliminieren u. a. alle served-States; Optimality-Cuts schließen den Gap in Runde 8 exakt.

### Nächste Schritte

Aufgabe **10/11** (Approximation Ratio + Time-to-Solution plotten, Vergleich gegen Gurobi-MILP).

---

## Ergebnis / Deliverable

1–2 Plots (Approximation Ratio + Time-to-Solution) aus wenigen Simulation Runs. Für das Paper reicht der **Proof-of-Concept**, dass der Loop funktioniert und feasible, brauchbare Lösungen liefert. Echte (nicht brute-force) Grover-Mixer- und Cost-Hamiltonian-Konstruktionen sind Future Work.

---

## Einschränkungen

- **Konvergenzgarantie nur so stark wie der Master-Solver.** Die Benders-Struktur liefert auf Dekompositionsebene eine Garantie zum globalen Optimum unter der Annahme, dass beide Subprobleme je Runde optimal gelöst werden. Das kontinuierliche LP erfüllt das (Gurobi), das diskrete Master aber nur, soweit die QAOA (Heuristik-Sampler) das Master-Optimum je Runde tatsächlich findet. Bei den kleinen PoC-Instanzen praktisch unkritisch (nahezu erschöpfendes Sampling möglich); für größere Instanzen offen. Die Garantie an diesen Vorbehalt koppeln, keine unbedingte Optimalität behaupten.
- **Kein Quantenvorteil-Claim.** Für dieses Problem schlägt Quantum den klassischen Solver nicht. Die Time-to-Solution auf dem Simulator misst simulierte Schaltkreis- plus klassische Solverkosten, nicht eine echte QPU-Laufzeit — entsprechend framen.
- **PoC-Maßstab.** Das Qubitbudget skaliert mit der Zahl diskreter Entscheidungen; deshalb kleine Instanzen, Statevector-Simulator / NVIDIA cuQuantum.

---

## Offene Fragen

- **Reicht Single-Asset für den PoC?** Für den reinen Mechanismus-Test ja — die Pipeline läuft vollständig. Zu beachten: Single-Asset ist _komplexitätsmäßig_ trivial (die XOR-Binärvariablen sind am Optimum determiniert, die LP-Relaxierung ist tight), taugt also **nicht** als Beleg für Quantenpotenzial. Echte kombinatorische Härte entsteht erst durch Kopplung mehrerer Assets (gemeinsame Demand Charge, geteilte Generatoren). Für den PoC unkritisch, für jede Potenzial-Aussage relevant.
- **QAOA-Winkel:** feste Ramp-Winkel bei kleinem `p` sollten für den PoC reichen — verifizieren, ob minimales Winkel-Tuning nötig wird.
- **Cut-Encoding im QAOA-Master:** Für den PoC bauen wir die Cuts brute-force in Diagonale/Zustandsvektor ein. Wie man Benders-Cuts _effizient_ (nicht brute-force) in Mixer bzw. Cost-Hamiltonian kodiert, ist offen — für den Praktikumsrahmen bewusst ausgeklammert, aber der interessante Punkt für Future Work.
- **Überträgt sich die Benders-Garantie mit einem heuristischen Master?** Formal gilt sie nur bei exaktem Master-Solve. Ab welcher Instanzgröße die QAOA das Master-Optimum nicht mehr zuverlässig trifft — und was das für die Konvergenz bedeutet — wäre einen eigenen Blick wert.