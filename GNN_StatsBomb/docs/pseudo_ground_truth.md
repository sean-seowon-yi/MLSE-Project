# Pseudo Ground Truth — Player Similarity Pairs

## Purpose

This document defines a set of player similarity pairs used as **directional
sanity checks** for the GNN player similarity system.  After running Phase 6
(inference), the automated evaluation checks whether the model's
nearest-neighbour rankings agree with these externally motivated pairs.

These are **not** strict benchmarks.  See [Caveats](#caveats) below.

Pairs are implemented in `src/phase6_inference/ground_truth.py`.  Run **`--mode ground_truth`** or the ground-truth step inside **`full_eval`**; results are written under `evaluations/{tag}/ground_truth/` (not in this document).

**Last updated:** March 2026

## Dataset Context

The model is trained on StatsBomb open data restricted to matches with 360
freeze-frame files.  The dataset covers:


| Competition            | Season  | Matches | Players   |
| ---------------------- | ------- | ------- | --------- |
| Bundesliga             | 2023/24 | 34      | 372       |
| FIFA World Cup         | 2022    | 64      | 678       |
| UEFA Women's Euro      | 2022    | 28      | 298       |
| UEFA Women's Euro      | 2025    | 31      | 313       |
| UEFA Euro              | 2020    | 51      | 490       |
| UEFA Euro              | 2024    | 51      | 491       |
| FIFA Women's World Cup | 2023    | 64      | 616       |
| **Total**              |         | **323** | **2 408** |


Players must have ≥ 50 possessions to receive an embedding
(`InferenceConfig.min_samples_per_player = 50`).

## Caveats

1. **Context mismatch (most important).**  Many publicly cited player
  comparisons originate from analyses of full club-season data.  Our dataset
   is overwhelmingly international tournaments plus one Bundesliga season.
   A player may behave very differently in these contexts — Robertson at
   Scotland is not Robertson at Liverpool.  Agreement is encouraging;
   disagreement does not necessarily indicate a model flaw.
2. **Sample size.**  Some players have relatively few possessions (e.g.
  Alexander-Arnold: 87, Yamal: 252).  Embeddings from small samples are
   noisier.
3. **Different model.**  Our GNN architecture, features, and similarity
  metric differ from any external system.  Exact rank reproduction is not
   expected even with identical data.
4. **Gender split.**  Several pairs are drawn from women's competitions only.
  The model trains on all data together with gender-aware retrieval, so these
   pairs validate within the women's sub-population.
5. **Directional, not metric.**  We check "does Player B appear in Player A's
  top-k?" and report rank + cosine similarity.  We do not expect exact
   similarity scores from external systems.

---

## Tier 1 — Strong directional expectations

Pairs with broad consensus from football analytics, media, and domain
experts.  We expect the partner to appear within the top-20, ideally top-10.

### Pair 1: Luka Modrić ↔ Toni Kroos


|                 | Player A                      | Player B                        |
| --------------- | ----------------------------- | ------------------------------- |
| **Name**        | Luka Modrić                   | Toni Kroos                      |
| **ID**          | 5463                          | 5574                            |
| **Position**    | Right Center Midfield         | Left Defensive Midfield         |
| **Possessions** | 828                           | 584                             |
| **Context**     | Croatia (WC 2022 + Euro 2024) | Germany (Euro 2020 + Euro 2024) |


**Why this pair:** Modrić and Kroos formed Real Madrid's midfield partnership
for over a decade.  Both are tempo-controlling deep-lying playmakers renowned
for press-resistance, metronomic passing accuracy, and exceptional spatial
awareness.  They dictate play through intelligent positioning rather than
athleticism.  Their statistical profiles on progressive passes received,
pass completion under pressure, and ball retention overlap extensively.

**Sources:** Managing Madrid tactical analysis; FBref statistical similarity
model; universally accepted comparison in football analytics.

### Pair 2: Virgil van Dijk ↔ Rúben Dias


|                 | Player A              | Player B             |
| --------------- | --------------------- | -------------------- |
| **Name**        | Virgil van Dijk       | Rúben Dias           |
| **ID**          | 3669                  | 5206                 |
| **Position**    | Left Center Back      | Left Center Back     |
| **Possessions** | 562                   | 618                  |
| **Context**     | Netherlands (WC 2022) | Portugal (Euro 2024) |


**Why this pair:** Both are dominant ball-playing centre-backs who defend
primarily through positioning and anticipation rather than reckless
challenges.  They are each their national team's primary ball-progressor from
the back, with similar defensive action profiles (high tackle success rate,
low fouling) and progressive passing volume.

**Sources:** Opta Analyst player comparison; The Athletic PL analytics.

### Pair 3: Jordi Alba ↔ Andrew Robertson


|                 | Player A           | Player B         |
| --------------- | ------------------ | ---------------- |
| **Name**        | Jordi Alba         | Andrew Robertson |
| **ID**          | 5211               | 3655             |
| **Position**    | Left Back          | Left Wing Back   |
| **Possessions** | 570                | 255              |
| **Context**     | Spain (Euros + WC) | Scotland (Euros) |


**Why this pair:** StatsBomb's Alba-replacement article identifies Robertson
as the strongest like-for-like option.  Both are overlapping left-backs known
for high-volume attacking output — crosses into the box, progressive carries
along the touchline, and underlapping runs.  Their underlying action
distributions in the final third are closely aligned.

**Sources:** StatsBomb public blog (Alba replacement analysis).

**Caveat:** Robertson plays for Scotland in our data, not Liverpool — a
different tactical system than the StatsBomb analysis.

### Pair 4: Trent Alexander-Arnold ↔ Achraf Hakimi


|                 | Player A                 | Player B          |
| --------------- | ------------------------ | ----------------- |
| **Name**        | Trent Alexander-Arnold   | Achraf Hakimi     |
| **ID**          | 3664                     | 5245              |
| **Position**    | Right Defensive Midfield | Right Back        |
| **Possessions** | 87                       | 333               |
| **Context**     | England (Euros)          | Morocco (WC 2022) |


**Why this pair:** StatsBomb's recruitment example flags Hakimi as a very
similar profile to TAA.  Both redefine the full-back role with elite long
passing range, progressive carries, and chance-creation from deep.  Their
expected-assist and key-pass profiles from the right-back zone are closely
matched.

**Sources:** StatsBomb recruitment-search blog post.

**Caveat:** TAA has only 87 possessions in the 360 data — borderline for
reliable embedding.  Different competition contexts (Euros vs World Cup).

### Pair 5: Aitana Bonmatí ↔ Alexia Putellas


|                 | Player A                           | Player B                           |
| --------------- | ---------------------------------- | ---------------------------------- |
| **Name**        | Aitana Bonmatí                     | Alexia Putellas                    |
| **ID**          | 15284                              | 10143                              |
| **Position**    | Right Center Midfield              | Left Center Midfield               |
| **Possessions** | 854                                | 465                                |
| **Context**     | Spain Women (Euro 2022 + WWC 2023) | Spain Women (Euro 2022 + WWC 2023) |


**Why this pair:** Barcelona and Spain's midfield engine.  They are
consecutive Ballon d'Or Féminin winners with overlapping profiles in ball
retention under pressure (~88% pass completion under pressure for Bonmatí),
progressive passing, between-the-lines movement, and spatial intelligence.
Guardiola described Bonmatí as "like the women's Iniesta", and Putellas is
compared to Modrić in Total Football Analysis scouting reports — both
pointing to the same playmaking archetype.

**Sources:** Pep Guardiola (quoted by 90min, Goal.com); Total Football
Analysis scouting report on Putellas; AP News.

---

## Tier 2 — Good directional expectations

Pairs with solid supporting evidence from analytics outlets or broadly
shared media comparisons.  We expect the partner within the top-50.

### Pair 6: Bukayo Saka ↔ Ousmane Dembélé


|                 | Player A                  | Player B                 |
| --------------- | ------------------------- | ------------------------ |
| **Name**        | Bukayo Saka               | Ousmane Dembélé          |
| **ID**          | 22084                     | 5477                     |
| **Position**    | Right Wing                | Right Wing               |
| **Possessions** | 518                       | 407                      |
| **Context**     | England (WC 2022 + Euros) | France (WC 2022 + Euros) |


**Why this pair:** Both are right-wing dribblers with high take-on rates and
creative final-third output.  FBref's statistical similarity model ranks
Dembélé among Saka's closest matches across European football.  Both cut
inside from the right half-space and generate chances through 1v1 dribbling
and through-ball delivery.

**Sources:** FBref statistical similarity scores; Eurosport/TNT Sports
comparison of inverted-winger archetypes.

### Pair 7: Harry Kane ↔ Robert Lewandowski


|                 | Player A                             | Player B            |
| --------------- | ------------------------------------ | ------------------- |
| **Name**        | Harry Kane                           | Robert Lewandowski  |
| **ID**          | 10955                                | 5668                |
| **Position**    | Center Forward                       | Left Center Forward |
| **Possessions** | 676                                  | 335                 |
| **Context**     | England + Bayern (Euros + WC + BuLi) | Poland (WC 2022)    |


**Why this pair:** Universally compared as the top two pure #9 strikers of
their generation.  Both combine clinical finishing (top-5 xG conversion in
Europe) with deep link-up play — they drop between the lines to receive,
play one-touch combinations, and make intelligent off-the-ball runs into
the box.  Their StatsBomb shot-profile and touch-map data overlap closely.

**Sources:** BBC Sport; Opta Analyst; StatsBomb shot-profile data.

### Pair 8: Florian Wirtz ↔ Kevin De Bruyne


|                 | Player A                   | Player B                      |
| --------------- | -------------------------- | ----------------------------- |
| **Name**        | Florian Wirtz              | Kevin De Bruyne               |
| **ID**          | 40724                      | 3089                          |
| **Position**    | Left Attacking Midfield    | Center Attacking Midfield     |
| **Possessions** | 1 820                      | 487                           |
| **Context**     | Germany (Euro 2024 + BuLi) | Belgium (WC 2022 + Euro 2024) |


**Why this pair:** Yahoo Sports' Euro 2024 preview described Wirtz as
"Germany have their own Kevin De Bruyne."  Both are creative attacking
midfielders who combine elite through-ball delivery with significant
goalscoring threat (Wirtz: 18G + 20A in 2023–24; De Bruyne routinely
posts similar dual-threat numbers).  Their progressive-pass and shot-
creation profiles are closely aligned.

**Sources:** Yahoo Sports; Reuters Euro 2024 analysis.

### Pair 9: Wendie Renard ↔ Millie Bright


|                 | Player A                            | Player B                             |
| --------------- | ----------------------------------- | ------------------------------------ |
| **Name**        | Wendie Renard                       | Millie Bright                        |
| **ID**          | 10125                               | 4642                                 |
| **Position**    | Left Center Back                    | Right Center Back                    |
| **Possessions** | 479                                 | 639                                  |
| **Context**     | France Women (Euro 2022 + WWC 2023) | England Women (Euro 2022 + WWC 2023) |


**Why this pair:** Both are tall (180 cm+), commanding centre-backs who
dominate aerially and progress the ball from deep.  They are widely regarded
as the two best CBs in women's football.  Their defensive action profiles
(aerial duels won, interceptions, ball recoveries) and progressive passing
rates from the back are closely matched.

**Sources:** FIFA.com player profiles; Her Football Hub scouting reports.

### Pair 10: Lauren Hemp ↔ María Caldentey


|                 | Player A                             | Player B                           |
| --------------- | ------------------------------------ | ---------------------------------- |
| **Name**        | Lauren Hemp                          | María Caldentey                    |
| **ID**          | 15555                                | 10161                              |
| **Position**    | Left Wing                            | Left Wing                          |
| **Possessions** | 787                                  | 840                                |
| **Context**     | England Women (Euro 2022 + WWC 2023) | Spain Women (Euro 2022 + WWC 2023) |


**Why this pair:** Both are creative left-wing forwards with high
progressive-carry volume and chance-creation rates.  They fill a similar
tactical role for their respective national teams — providing width on the
left, cutting inside for shots or threading through-balls to the striker.
Their carry and chance-creation per-90 numbers are closely aligned.

**Sources:** FBref statistical profiles; PlanetFootball positional analysis.

### Pair 11: Toni Kroos ↔ Enzo Fernandez


|                 | Player A                        | Player B                  |
| --------------- | ------------------------------- | ------------------------- |
| **Name**        | Toni Kroos                      | Enzo Fernandez            |
| **ID**          | 5574                            | 38718                     |
| **Position**    | Left Defensive Midfield         | Center Defensive Midfield |
| **Possessions** | 584                             | 351                       |
| **Context**     | Germany (Euro 2020 + Euro 2024) | Argentina (WC 2022)       |


**Why this pair:** StatsBomb's Kroos-replacement article notes that Enzo
Fernandez appears in the raw top-5 results.  Both are tempo-controlling
midfielders who sit deep, circulate possession with metronomic accuracy,
and occasionally deliver incisive long-range through-balls.  Enzo was
widely seen as the heir to Kroos's style when he transferred to Chelsea.

**Sources:** StatsBomb public blog (Kroos replacement analysis).

### Pair 12: Jude Bellingham ↔ Antoine Griezmann


|                 | Player A                      | Player B                     |
| --------------- | ----------------------------- | ---------------------------- |
| **Name**        | Jude Bellingham               | Antoine Griezmann            |
| **ID**          | 30714                         | 5487                         |
| **Position**    | Center Attacking Midfield     | Center Attacking Midfield    |
| **Possessions** | 607                           | 690                          |
| **Context**     | England (WC 2022 + Euro 2024) | France (WC 2022 + Euro 2024) |


**Why this pair:** Both operate as goal-threat attacking midfielders who
arrive late into the box with exceptional timing.  Bellingham's breakout
"goalscoring midfielder" profile was compared to Frank Lampard by Lampard
himself (BBC Sport), and Griezmann occupies a similar hybrid creator-scorer
role at the top of midfield.  Opta analytics highlight their shared tendency
to generate xG from central zones 14–18 metres from goal.

**Sources:** BBC Sport (Lampard interview); Opta Analyst creative-midfielder
profiles.

---

## Tier 3 — Conditional / weaker expectations

Pairs with looser evidence — media comparisons rather than direct analytics
endorsements, or structural caveats that weaken the expectation.  Any
above-average similarity is a positive signal.

### Pair 13: Jamal Musiala ↔ Phil Foden


|                 | Player A                             | Player B                  |
| --------------- | ------------------------------------ | ------------------------- |
| **Name**        | Jamal Musiala                        | Phil Foden                |
| **ID**          | 39565                                | 4354                      |
| **Position**    | Left Wing                            | Left Wing                 |
| **Possessions** | 442                                  | 477                       |
| **Context**     | Germany (WC 2022 + Euro 2024 + BuLi) | England (WC 2022 + Euros) |


**Why this pair:** Both are versatile left-side attackers with exceptional
close-control dribbling who can operate in tight interlinear spaces.
talkSPORT and Yahoo Sports have directly compared them as the best young
attacking talents in Europe, citing similar dribbling success rates and
progressive-action profiles.

**Caveat:** Their dribbling styles differ — Musiala relies more on stepovers
and body feints, Foden on positional intelligence and first-touch
manipulation.  This stylistic nuance may or may not surface in event data.

**Sources:** talkSPORT; Yahoo Sports; Musiala self-comparison interview
(Bavarian Football Works).

### Pair 14: Lamine Yamal ↔ Nico Williams


|                 | Player A          | Player B          |
| --------------- | ----------------- | ----------------- |
| **Name**        | Lamine Yamal      | Nico Williams     |
| **ID**          | 316046            | 68574             |
| **Position**    | Right Wing        | Left Wing         |
| **Possessions** | 252               | 317               |
| **Context**     | Spain (Euro 2024) | Spain (Euro 2024) |


**Why this pair:** Spain's Euro 2024 wide-forward duo.  Both are young,
explosive inverted wingers with high take-on rates and direct goal threat.
Goal.com's Euro 2024 preview and FBref scouting similarity scores show
overlapping profiles in progressive carries, shot-creating actions, and
dribble success rate.

**Caveat:** They play on opposite sides (Yamal RW, Williams LW), so their
on-ball action angles differ.  Both have relatively few possessions (252 and
317).  Played in the same tournament and same team — the model may or may
not capture intra-team stylistic similarity.

**Sources:** Goal.com Euro 2024 preview; FBref scouting similarity.

### Pair 15: Manuel Neuer ↔ Gianluigi Donnarumma


|                 | Player A                        | Player B             |
| --------------- | ------------------------------- | -------------------- |
| **Name**        | Manuel Neuer                    | Gianluigi Donnarumma |
| **ID**          | 5570                            | 7036                 |
| **Position**    | Goalkeeper                      | Goalkeeper           |
| **Possessions** | 339                             | 296                  |
| **Context**     | Germany (Euro 2020 + Euro 2024) | Italy (Euro 2020)    |


**Why this pair:** Both are imposing, distribution-oriented goalkeepers who
command a high defensive line.  They were consecutive Euro Championship
goalkeeper-of-the-tournament winners (Donnarumma won the Euro 2020 POTM;
Neuer was in the Euro 2020 squad and a Euro 2024 starter).  Both build play
from the back with above-average pass completion for goalkeepers and
regularly sweep outside the box.

**Caveat:** Neuer is a more aggressive sweeper-keeper who comes significantly
further off his line; Donnarumma is more line-oriented.  This behavioural
difference may reduce their embedding similarity despite shared
distributional patterns.

**Sources:** UEFA.com; FIFA.com goalkeeper profiles.

---

## Quick-Reference Table


| Tier | Player A               | ID     | Player B             | ID    | Poss A | Poss B |
| ---- | ---------------------- | ------ | -------------------- | ----- | ------ | ------ |
| 1    | Luka Modrić            | 5463   | Toni Kroos           | 5574  | 828    | 584    |
| 1    | Virgil van Dijk        | 3669   | Rúben Dias           | 5206  | 562    | 618    |
| 1    | Jordi Alba             | 5211   | Andrew Robertson     | 3655  | 570    | 255    |
| 1    | Trent Alexander-Arnold | 3664   | Achraf Hakimi        | 5245  | 87     | 333    |
| 1    | Aitana Bonmatí         | 15284  | Alexia Putellas      | 10143 | 854    | 465    |
| 2    | Bukayo Saka            | 22084  | Ousmane Dembélé      | 5477  | 518    | 407    |
| 2    | Harry Kane             | 10955  | Robert Lewandowski   | 5668  | 676    | 335    |
| 2    | Florian Wirtz          | 40724  | Kevin De Bruyne      | 3089  | 1 820  | 487    |
| 2    | Wendie Renard          | 10125  | Millie Bright        | 4642  | 479    | 639    |
| 2    | Lauren Hemp            | 15555  | María Caldentey      | 10161 | 787    | 840    |
| 2    | Toni Kroos             | 5574   | Enzo Fernandez       | 38718 | 584    | 351    |
| 2    | Jude Bellingham        | 30714  | Antoine Griezmann    | 5487  | 607    | 690    |
| 3    | Jamal Musiala          | 39565  | Phil Foden           | 4354  | 442    | 477    |
| 3    | Lamine Yamal           | 316046 | Nico Williams        | 68574 | 252    | 317    |
| 3    | Manuel Neuer           | 5570   | Gianluigi Donnarumma | 7036  | 339    | 296    |


## Pair Selection Criteria

The pairs were selected based on the following principles:

1. **Position diversity** — the 15 pairs span all four position groups
  (Goalkeeper, Defender, Midfielder, Forward) so the evaluation covers the
   full embedding space.
2. **Gender diversity** — 4 pairs are from women's football (Pairs 5, 9, 10,
  and partially within Pair 11), ensuring the evaluation covers both the
   male and female sub-populations.
3. **Evidence quality** — Tier 1 pairs have direct StatsBomb endorsement or
  unanimous domain-expert consensus.  Tier 2 pairs have analytics-outlet
   citations or widely accepted media comparisons.  Tier 3 pairs have
   reasonable but less definitive support.
4. **Data sufficiency** — most players have 300+ possessions, providing
  enough data for a stable embedding.  The few exceptions (TAA: 87,
   Yamal: 252, Robertson: 255) are noted as caveats.
5. **Playing-style similarity, not positional similarity** — pairs were
  chosen because the players make similar decisions in similar situations
   (e.g. both are tempo-controlling deep playmakers), not merely because
   they share a position label.

