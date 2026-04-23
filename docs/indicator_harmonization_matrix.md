# Indicator Harmonization Matrix For `infraScanRoad` and `infraScanRail`

This note is centered on your thesis question:

How far should `infraScanRoad` and `infraScanRail` use the same approaches, calculations, and valuation factors so that outputs are comparable, while still respecting the mode-specific logic from NIBA and NISTRA?

The answer is:

- keep mode-specific physical impact calculations where road and rail genuinely differ
- harmonize valuation factors, discounting, time horizon, aggregation, and ranking logic wherever the impact concept is the same
- use `eniba.xlsm` and `enistra_2022_de.xlsm` to infer both the common valuation logic and the mode-specific calculation logic

## 1. What the Excel files can tell us

### NIBA workbook

The workbook is formula-readable and relatively compact. The key evidence is in `Berechnungen` and `Umrechnungsfaktoren`.

Examples:

- air pollution is monetized from traffic quantities times unit values
- noise is monetized from traffic quantities times unit values
- climate is monetized from traffic quantities times unit values
- travel time gains are monetized using explicit time-value parameters
- accidents are monetized using explicit accident cost factors
- the workbook uses a discount rate of `0.02`

### NISTRA workbook

The workbook is also formula-readable, but the logic is distributed across indicator modules.

Examples:

- indicator families are explicit: `DK*`, `VQ*`, `SI*`, `UW*`, `QI*`
- `Bewertungssätze KNA` makes the valuation side explicit
- `Zusammenfassung KNA` exposes sensitivity assumptions including:
  - `Diskontsatz`
  - `Verkehrswachstum`
  - `Sensitivität Zeitwert`
  - `Klima-Kostensatz`
  - `VOSL`
- `Gewichtung und Annahmen KWA` exposes MCDA weighting using `SUMPRODUCT(...)`

So yes: the Excel files are useful not only for outputs, but also for reconstructing the intended calculation logic.

## 2. Immediate harmonization issue already visible in InfraScan

There are already important factor mismatches between road and rail.

### Discount rate

- NIBA workbook: `0.02`
- NISTRA workbook: `0.02`
- `infraScanRail`: `discount_rate = 0.03` in [cost_parameters.py](/Users/laura/Desktop/infraScan_lkuehner/infraScan/infraScanRail/cost_parameters.py:21)
- `infraScanRoad`: no equally explicit centralized discounting factor visible in the same way from the inspected files

Recommendation:

- if you want comparability and alignment with both reference tools, the common benchmark discount rate should be `0.02`
- if you keep `0.03` in rail, this should be a sensitivity case, not the main case

### Value of travel time savings

- NIBA workbook explicitly contains time valuation factors in `Umrechnungsfaktoren`
- `infraScanRoad`: `VTTS = 32.2` CHF/h in [settings.py](/Users/laura/Desktop/infraScan_lkuehner/infraScan/infraScanRoad/settings.py:62)
- `infraScanRail`: `VTTS = 14.8` CHF/h in [cost_parameters.py](/Users/laura/Desktop/infraScan_lkuehner/infraScan/infraScanRail/cost_parameters.py:3)

Recommendation:

- do not assume one common VTTS if the user groups differ
- instead align to a common source and common year / price basis
- then use mode-specific VTTS only if justified by NIBA / NISTRA or by user composition

### Time horizon

- both current models use long horizons around `50` years in their monetization parameters
- NIBA and NISTRA both use explicit long-term annualization / comparison-year logic

Recommendation:

- keep one common main valuation horizon for both models
- document any extra rail capacity phasing separately

## 3. Priority indicators: detailed harmonization

These are the four highest-value indicators to align first.

### 3.1 Owner: Costs

#### NIBA logic

NIBA has explicit infrastructure and operating cost indicators, including:

- `10.3 Betriebskosten Infrastruktur`
- investment inputs in `Infrastruktur`

The workbook structure suggests direct calculation from infrastructure quantities and cost factors, then annualization / conversion.

#### NISTRA logic

NISTRA uses:

- `DK1` Baukosten
- `DK2` Ersatzinvestitionen
- `DK3` Landkosten
- `DK4` Betriebs- und Unterhaltskosten Strasse

This is the clearest indication that owner cost should be treated as a core harmonized indicator family.

#### Current InfraScanRoad

- construction costs in [scoring.py](/Users/laura/Desktop/infraScan_lkuehner/infraScan/infraScanRoad/scoring.py:63)
- maintenance costs in [scoring.py](/Users/laura/Desktop/infraScan_lkuehner/infraScan/infraScanRoad/scoring.py:129)
- coefficients in [settings.py](/Users/laura/Desktop/infraScan_lkuehner/infraScan/infraScanRoad/settings.py:49)

Road is already highly explicit:

- `c_openhighway = 15200`
- `c_tunnel = 416000`
- `c_bridge = 63900`
- `ramp = 102000000`
- maintenance coefficients are also explicit

#### Current InfraScanRail

- construction and maintenance costs in [scoring.py](/Users/laura/Desktop/infraScan_lkuehner/infraScan/infraScanRail/scoring.py:211)
- coefficients in [cost_parameters.py](/Users/laura/Desktop/infraScan_lkuehner/infraScan/infraScanRail/cost_parameters.py:5)

Rail is also explicit:

- `track_cost_per_meter`
- `tunnel_cost_per_meter`
- `bridge_cost_per_meter`
- maintenance as a fraction of construction cost
- capacity intervention logic for additional owner cost

#### Harmonization recommendation

Alignment level:

- same indicator family: yes
- same raw formula: no
- same valuation principle: yes

Recommendation:

- keep raw engineering cost generation mode-specific
- harmonize:
  - price basis year
  - discounting
  - residual-value treatment
  - treatment of replacements and maintenance
  - optimism bias
- create a common owner-cost breakdown:
  - initial investment
  - replacement / renewal
  - maintenance and operations
  - residual value
  - total discounted owner cost

#### Thesis note

This is a case where comparability should come from the accounting framework, not from identical physical cost formulas.

### 3.2 User: Value of travel time saving

#### NIBA logic

NIBA directly monetizes travel time gains in `Berechnungen`, for example:

- `11.1 Reisezeitgewinn Stammverkehr Personenverkehr`

and uses explicit values in `Umrechnungsfaktoren`, including:

- time value for travel time change
- time value for adaptation time
- time value for reliability

#### NISTRA logic

NISTRA uses:

- `VQ1n Reisezeit Stammverkehr`
- `VQ2n Zuverlässigkeit`

and exposes time-value sensitivity in `Zusammenfassung KNA`.

This indicates that travel time and reliability should be separate but coordinated user-benefit indicators.

#### Current InfraScanRoad

- travel time monetization in [scoring.py](/Users/laura/Desktop/infraScan_lkuehner/infraScan/infraScanRoad/scoring.py:2937)
- `VTTS = 32.2` in [settings.py](/Users/laura/Desktop/infraScan_lkuehner/infraScan/infraScanRoad/settings.py:62)

Current road monetization uses:

- `mon_factor = VTTS * 2.5 * 250 * duration`

So the road model currently uses a workday / peak-period scaling assumption rather than the `365` annualization seen in rail.

#### Current InfraScanRail

- travel time monetization in [scoring.py](/Users/laura/Desktop/infraScan_lkuehner/infraScan/infraScanRail/scoring.py:1692)
- alternative monetization in [TT_Delay.py](/Users/laura/Desktop/infraScan_lkuehner/infraScan/infraScanRail/TT_Delay.py:291)
- `VTTS = 14.8` in [cost_parameters.py](/Users/laura/Desktop/infraScan_lkuehner/infraScan/infraScanRail/cost_parameters.py:3)

Current rail monetization uses:

- `mon_factor = VTTS * 365 * duration`

#### Harmonization recommendation

Alignment level:

- same indicator family: yes
- same raw travel-time computation: no
- same valuation logic: yes, as far as possible

Recommendation:

- keep network/travel-time generation mode-specific
- harmonize:
  - unit of travel time difference
  - annualization logic
  - valuation-year price basis
  - user segmentation logic
- decide explicitly whether road and rail should both use:
  - a full-year factor, or
  - a workday / peak-period factor

Right now road and rail are not directly comparable because their annualization differs.

#### Thesis note

This is one of the clearest harmonization priorities.

### 3.3 User / Direct affected public: Reliability and resilience-related user effects

#### NIBA logic

NIBA explicitly contains a reliability time-value parameter in `Umrechnungsfaktoren`.

#### NISTRA logic

NISTRA explicitly contains:

- `VQ2n Zuverlässigkeit`
- `VQ2w Zuverlässigkeit`

So reliability is clearly a first-class indicator in the reference tools.

#### Current InfraScanRoad

From the inspected files, road is strong on travel time and accessibility, but no equally explicit reliability monetization is visible yet.

#### Current InfraScanRail

Rail already contains partial reliability/comfort logic:

- comfort-weighted transfer time in [TT_Delay.py](/Users/laura/Desktop/infraScan_lkuehner/infraScan/infraScanRail/TT_Delay.py:97)
- capacity constraints as a system-quality feature throughout the capacity pipeline

#### Harmonization recommendation

Alignment level:

- same indicator family: yes
- same raw formula: probably no
- same valuation principle: yes

Recommendation:

- define one common reliability concept first:
  - travel-time variance
  - delay exposure
  - robustness / redundancy
  - missed-connection penalties
- then use mode-specific raw models to generate the metric
- monetize or score it using one common valuation logic where possible

#### Thesis note

Reliability is likely one of the biggest current gaps in direct comparability.

### 3.4 Direct affected public: Externalities

#### NIBA logic

NIBA directly monetizes:

- air pollution
- noise
- climate
- accidents

in `Berechnungen`, using unit values from `Umrechnungsfaktoren`.

#### NISTRA logic

NISTRA explicitly separates these as indicator families:

- `SI1n` accidents
- `UW1n_Luft` air pollution
- `UW1n_Lärm` noise
- `UW6` climate-related indicator

The valuation layer is exposed through `Bewertungssätze KNA`.

#### Current InfraScanRoad

Road already has a strong externality layer:

- externalities in [scoring.py](/Users/laura/Desktop/infraScan_lkuehner/infraScan/infraScanRoad/scoring.py:234)
- noise in [scoring.py](/Users/laura/Desktop/infraScan_lkuehner/infraScan/infraScanRoad/scoring.py:311)
- explicit parameters in [settings.py](/Users/laura/Desktop/infraScan_lkuehner/infraScan/infraScanRoad/settings.py:66)

Road explicitly monetizes:

- climate
- habitat loss
- fragmentation
- land reallocation
- noise

Accidents were not found as a similarly developed explicit valuation block in the inspected road files.

#### Current InfraScanRail

Rail is strong on cost and travel time, but from the inspected files the externality layer is less explicit than road.

This suggests that:

- rail may currently underrepresent environmental and accident externalities compared with both reference tools
- road may currently represent some environmental externalities more strongly than rail

#### Harmonization recommendation

Alignment level:

- same indicator families: yes
- same raw exposure model: usually no
- same valuation factors: yes, where units are made compatible

Recommendation:

- define a common externality menu:
  - climate
  - air pollution
  - noise
  - accidents
  - habitat / fragmentation / land take
- then decide for each:
  - whether both models can calculate it
  - whether the same unit value can be used

#### Thesis note

This is the second strongest harmonization priority after travel time.

## 4. Matrix based on your stakeholder list

| Stakeholder | Indicator | NIBA / NISTRA relevance | Current road status | Current rail status | Recommended harmonization |
| --- | --- | --- | --- | --- | --- |
| All | Time Horizon | explicit long-horizon valuation logic | present | present | use one common main horizon |
| All | Discount rate | explicit in both workbooks, benchmark `0.02` | not yet surfaced as clearly in inspected road files | `0.03` currently | use `0.02` as main benchmark, test alternatives in sensitivity |
| All | Optimism Bias | not explicit in inspected workbook snippets but consistent with project appraisal logic | not explicit | not explicit | add one common uplift rule for owner costs |
| Owner | Costs | core in both tools | strong | strong | harmonize accounting and discounting, not engineering formula |
| User | Costs | partly covered via operating costs and generalized cost concepts | partial | partial | distinguish operator cost from user generalized cost |
| User | VTTS | core in both tools | strong | strong | harmonize annualization, price basis, segmentation |
| User | Network Access | indirect in both tools | strong accessibility logic | strong catchment/access logic | define one common access metric, keep mode-specific raw model |
| User | Network redundancy | partly linked to reliability / resilience | weak explicit treatment | partly via capacity and graph structure | define as a scored indicator first |
| User | Utilization | relevant but not explicit in workbook snippets | weak | strong via capacity / flows | use common reporting unit, mode-specific raw calculation |
| User | Comfort | weakly explicit in NIBA/NISTRA, often embedded in generalized travel cost | weak | partial via comfort-weighted transfer time | make explicit only if needed |
| User | Reliability | explicit in both tools | weak explicit treatment | partial | high-priority harmonization gap |
| User | Modal shift | implicit as a mechanism behind impacts | scenarios exist | scenarios exist | treat as explanatory mechanism, not necessarily a top-level welfare indicator |
| Direct affected public | Externality | core in both tools | strong | partial | high-priority harmonization |
| Indirect affected public | Accessibility to opportunities | closer to wider economic / regional accessibility | partial | partial | define a common accessibility-opportunity metric if in scope |
| Indirect affected public | Number of accidents | explicit in both tools | currently weak / not clearly implemented | not clearly implemented in inspected files | add as common indicator if data permit |

## 5. What should be harmonized at which level

### Harmonize fully

- discount rate
- price basis year
- valuation horizon
- annualization convention
- NPV logic
- residual-value logic
- scenario naming and year structure
- CBA aggregation rules
- MCDA weighting method

### Harmonize at factor level, but not raw-calculation level

- VTTS
- reliability value
- accident unit values
- air / climate / noise unit values
- habitat / land-take valuation factors

### Keep mode-specific

- network generation
- travel-time assignment
- capacity logic
- access/catchment mechanics
- engineering cost geometry
- exposure calculation for some externalities

## 6. Best next step for the thesis

The next useful task is not software architecture. It is methodological mapping.

Build a compact table with these columns:

- `indicator`
- `stakeholder`
- `road raw calculation`
- `rail raw calculation`
- `NIBA reference`
- `NISTRA reference`
- `common factor possible?`
- `common aggregation possible?`
- `recommended harmonized main case`
- `recommended sensitivity cases`

The first four indicators to populate should be:

1. owner costs
2. VTTS
3. reliability
4. externalities

That would give you the methodological backbone for a strong comparability chapter.
