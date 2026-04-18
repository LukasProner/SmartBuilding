# Q_learning1804_05.py – Podrobná dokumentácia

## 1. Prehľad a motivácia

Program `Q_learning1804_05.py` implementuje **tabuľkový Q-learning** (tabular Q-learning) nad prostredím CityLearn, kde agent riadi energetické zásobníky troch budov. Cieľom je experimentálne porovnať **12 rôznych reward funkcií**, z ktorých každá vedie agenta k odlišnej stratégii riadenia spotreby elektriny.

### Prečo 12 reward funkcií?

V reinforcement learningu je reward funkcia **jediný komunikačný kanál**, cez ktorý hovoríme agentovi, čo chceme dosiahnuť. Rôzne formulácie rewardu môžu viesť k radikálne odlišnému správaniu — aj keď agent, prostredie a stavový priestor zostanú identické. Program umožňuje systematicky skúmať, aký vplyv má dizajn rewardu na:

- celkový grid import (kWh)
- úspory oproti fixnej stratégii (%)
- diskomfort obyvateľov budovy (prehriatie)
- tvar profilu spotreby (špičky, plynulosť, nočné nabíjanie)

### Pozícia v rámci série experimentov

| Súbor | Čo pridáva |
|---|---|
| Q_learning1804_01 | Základný Q-learning, 1 reward |
| Q_learning1804_02 | Rozšírenie o viac pozorovaní |
| Q_learning1804_03 | 3 reward funkcie (Weather, Energy, Pricing) |
| Q_learning1804_04 | +1 reward (ComfortAware), diskomfort metrika |
| **Q_learning1804_05** | **+8 nových reward funkcií = 12 celkom**, refaktorovaná experimentálna slučka |

---

## 2. Prostredie CityLearn

### 2.1 Dataset

Používa sa `citylearn_challenge_2023_phase_1` – reálne dáta spotreby budov. Schema sa nachádza v:

```
data/datasets/citylearn_challenge_2023_phase_1/schema.json
```

### 2.2 Budovy

Program štandardne pracuje s tromi budovami: `Building_1`, `Building_2`, `Building_3`. Každá budova má vlastné zásobníky (DHW, batéria) a chladiaci systém. Agent rozhoduje **nezávisle pre každú budovu** (`central_agent=False`), čo znamená, že:

- reward funkcia vracia **zoznam** odmeien (jedna pre každú budovu)
- Q-tabuľka je zdieľaná, ale stavový index sa počíta pre každú budovu zvlášť
- akcie sa vyberajú per-budova

### 2.3 Pozorovania (observations)

Agent má prístup k 11 surových pozorovaniam z prostredia:

| Pozorovanie | Význam |
|---|---|
| `outdoor_dry_bulb_temperature_predicted_1` | Predpoveď vonkajšej teploty na 1h dopredu |
| `outdoor_dry_bulb_temperature_predicted_2` | Predpoveď vonkajšej teploty na 2h dopredu |
| `diffuse_solar_irradiance_predicted_1` | Difúzne slnečné žiarenie (predpoveď 1h) |
| `diffuse_solar_irradiance_predicted_2` | Difúzne slnečné žiarenie (predpoveď 2h) |
| `direct_solar_irradiance_predicted_1` | Priame slnečné žiarenie (predpoveď 1h) |
| `direct_solar_irradiance_predicted_2` | Priame slnečné žiarenie (predpoveď 2h) |
| `electricity_pricing` | Aktuálna cena elektrickej energie |
| `electricity_pricing_predicted_1` | Predpovedaná cena na 1h dopredu |
| `dhw_storage_soc` | Stav nabitia zásobníka teplej úžitkovej vody (0–1) |
| `electrical_storage_soc` | Stav nabitia batérie (0–1) |
| `occupant_count` | Počet obyvateľov prítomných v budove |

**Dôležité:** Reward funkcie majú prístup k **plným observation dicts** z prostredia (vrátane napr. `net_electricity_consumption`, `indoor_dry_bulb_temperature_cooling_delta`), nie iba k aktívnym pozorovaniam. Preto môžu využívať informácie, ktoré agent "nevidí" pri rozhodovaní, ale slúžia ako spätná väzba pre vyhodnotenie kvality akcie.

### 2.4 Akcie

Agent ovláda 3 zariadenia v každej budove:

| Akcia | Rozsah | Diskretizácia | Význam |
|---|---|---|---|
| `dhw_storage` | [-1, 1] | 3 biny: {-1.0, 0.0, 1.0} | Nabíjanie/vybíjanie zásobníka teplej vody |
| `electrical_storage` | [-1, 1] | 3 biny: {-1.0, 0.0, 1.0} | Nabíjanie/vybíjanie batérie |
| `cooling_device` | [0, 1] | 5 binov: {0.0, 0.25, 0.5, 0.75, 1.0} | Výkon chladiaceho zariadenia |

Celkový počet **joint akcií** = 3 × 3 × 5 = **45**.

---

## 3. Feature Engineering – Stavový priestor

### 3.1 Prečo inžiniering rysov?

Surových 11 pozorovaní s plnými rozsahmi by vyžadovalo obrovský počet diskrétnych stavov. Namiesto toho sa 11 surových pozorovaní transformuje na **9 inžinierovaných rysov** (engineered features), ktoré zachytávajú kľúčové vzory s menšou kardinalitou.

### 3.2 Transformácie

| Inžinierovaný rys | Výpočet | Biny | Význam |
|---|---|---|---|
| `temp_pred_1` | `t1` (priamo) | 3 | Aká bude teplota o hodinu |
| `temp_trend_12` | `t2 - t1` | 3 | Trend teploty: otepľuje/ochladuje sa? |
| `solar_pred_1_total` | `diffuse_1 + direct_1` | 3 | Celkové slnečné žiarenie o hodinu |
| `solar_trend_12_total` | `(d2+r2) - (d1+r1)` | 3 | Trend žiarenia: pribúda/ubúda? |
| `price_now` | `p0` (priamo) | 3 | Aktuálna cenová hladina |
| `price_trend_01` | `p1 - p0` | 3 | Pôjde cena hore alebo dole? |
| `dhw_storage_soc` | priamo | 3 | Stav zásobníka vody |
| `electrical_storage_soc` | priamo | 3 | Stav batérie |
| `occupancy_present` | `1 ak count > 0, inak 0` | 2 | Je budova obsadená? |

### 3.3 Veľkosť stavového priestoru

$3 \times 3 \times 3 \times 3 \times 3 \times 3 \times 3 \times 3 \times 2 = 13{,}122$ stavov

### 3.4 Veľkosť Q-tabuľky

$13{,}122 \text{ stavov} \times 45 \text{ akcií} = 590{,}490$ buniek

Pri `float32` = $590{,}490 \times 4 = 2{,}361{,}960$ bytov ≈ **2.25 MB** na Q-tabuľku.

### 3.5 Diskretizácia – ako fungujú biny

Trieda `ObservationDiscretizer` používa **rovnomerné delenie** (equal-width binning). Pre každý rys sa z min/max hodnôt pozorovaného priestoru vytvorí `n+1` hraničných bodov, z ktorých sa vnútorné `n-1` hraníc použije pre `numpy.digitize()`. Príklad pre 3 biny:

```
min ────┤──── edge1 ────┤──── edge2 ────┤──── max
  bin 0         bin 1          bin 2
```

Funkcia `encode()` vypočíta inžinierované rysy zo surového observation vektora, digitalizuje každý rys do jeho bin indexu, a potom pomocou `numpy.ravel_multi_index()` prevedie n-ticu indexov na jediné celé číslo – **flat state index**.

---

## 4. Akčný priestor – MultiActionDiscretizer

### 4.1 Kartézsky súčin

Pre každú dimenziu akcie sa vytvorí lineárna mriežka hodnôt (napr. pre `electrical_storage` s 3 binmi: `[-1.0, 0.0, 1.0]`). Potom sa vytvorí **kartézsky súčin** všetkých dimenzií:

```python
itertools.product([-1,0,1], [-1,0,1], [0, 0.25, 0.5, 0.75, 1.0])
→ 45 n-tíc
```

Každá n-tica je jedna "joint akcia". Agent si vo svojej Q-tabuľke vyberá index 0–44 a `decode_one()` ho preloží späť na konkrétne hodnoty akcií.

### 4.2 Prečo 5 binov pre chladenie a len 3 pre zásobníky?

Chladenie je **jednousmerné** (0 = vypnuté, 1 = maximum) a jemnejšie rozlíšenie výkonu má väčší vplyv na komfort a spotrebu. Zásobníky majú jasné režimy: nabiť / nič nerobiť / vybiť – tri stavy postačujú.

---

## 5. Q-Learning agent – OwnAdaptiveTabularQLearning

### 5.1 Základný algoritmus

Štandardný Q-learning s TD(0) updateom:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \big[ r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \big]$$

kde:
- $\alpha = 0.15$ – learning rate
- $\gamma = 0.95$ – discount factor
- $r$ – okamžitá odmena
- $s'$ – nasledujúci stav

### 5.2 Epsilon-greedy explorácia

Agent začína s $\varepsilon = 1.0$ (100% náhodné akcie) a exponenciálne znižuje:

$$\varepsilon_{ep} = \varepsilon_0 \cdot e^{-0.03 \cdot ep}$$

s dolným limitom $\varepsilon_{min} = 0.05$. To zabezpečuje, že:
- Na začiatku agent objaví rôzne časti stavového priestoru
- Postupne prechádza na využívanie naučenej politiky
- Vždy zostáva 5% explorácia pre prípad, že sa prostredie zmení

### 5.3 Adaptívny mechanizmus epsilon boostu

Toto je **vlastné rozšírenie** oproti štandardnému Q-learningu. Myšlienka:

1. Každých 5 epizód sa vypočíta kĺzavý priemer odmeny (rolling-5)
2. Ak sa tento priemer **nezlepšil** aspoň o `adaptive_min_improvement = 0.01` po dobu `adaptive_patience = 8` po sebe nasledujúcich epizód → epsilon sa **zvýši** o `adaptive_epsilon_boost = 0.08` (max na 0.35)
3. Tým sa agent "prinúti" opäť viac objavovať, keď sa učenie zastavilo

**Prečo to funguje?** Tabuľkový Q-learning môže uviaznuť v lokálnych minimách, kde agent opakovane navštevuje tie isté stavy a robí tie isté akcie. Dočasné zvýšenie explorácie ho vyvedie z tejto slučky.

### 5.4 Hyperparametre – výber a zdôvodnenie

| Parameter | Hodnota | Zdôvodnenie |
|---|---|---|
| `learning_rate` | 0.15 | Kompromis: dostatočne rýchle učenie vs. stabilita. Pri 0.3+ sa Q-hodnoty rozkmitávajú, pri 0.05 sa učenie príliš spomalí |
| `discount_factor` | 0.95 | Budúce odmeny sú dôležité (energetické rozhodnutia majú dlhodobé dôsledky), ale nie nekonečne |
| `epsilon_decay` | 0.03 | Pri 200 epizódach: ep. 50 → ε≈0.22, ep. 100 → ε≈0.05. Agent má ~100 epizód na prieskum |
| `minimum_epsilon` | 0.05 | Trvalá explorácia zabráni úplnému uviaznutiu |
| `adaptive_patience` | 8 | Dá agentovi dosť epizód na prekonanie štatistického šumu pred boost-om |
| `adaptive_epsilon_boost` | 0.08 | Dostatočne veľký na obnovenie prieskumu, ale nie tak veľký aby zničil naučené |
| `random_seed` | 7 | Reprodukovateľnosť – rovnaký seed → rovnaké výsledky |

---

## 6. Reward funkcie – podrobný rozbor

Všetky reward funkcie dedia z `citylearn.reward_function.RewardFunction`. Metóda `calculate()` prijíma zoznam observation dicts (jeden per budova) a vracia zoznam odmien. Pri `central_agent=False` vracia per-budova odmeny.

### 6.1 WeatherOccupancyReward

**Vzorec:**
$$r = -(\text{grid\_import} \times \text{occ\_factor} \times \text{weather\_factor})$$

kde:
- $\text{occ\_factor} = 1 + 1.5 \times \mathbb{1}[\text{occupant\_count} > 0]$
- $\text{weather\_factor} = 1 + 0.05 \times \max(\overline{T}_{forecast} - 24, 0)$

**Logika:** Penalizácia grid importu sa **zosilňuje**, keď sú ľudia v budove (vtedy je spotreba dôležitejšia z pohľadu komfortu) a keď je horúco (chladenie je najnáročnejšie pri vysokých teplotách). Práh 24°C bol zvolený ako typická hranica, kedy sa začína chladiť.

**Kedy je vhodný:** Keď chceme, aby agent prioritizoval úspory počas obsadených hodín a horúcich dní.

### 6.2 GridImportOnlyReward (Energy-only)

**Vzorec:**
$$r = -\max(\text{net\_electricity\_consumption}, 0)$$

**Logika:** Najjednoduchšia možná penalizácia – čistý lineárny trest za každý kWh importovaný zo siete. Export sa ignoruje (clipuje na 0). Slúži ako **baseline**, voči ktorému sa porovnávajú sofistikovanejšie reward funkcie.

**Kedy je vhodný:** Ako referencia. Ukazuje, čo dokáže agent, keď sa snaží len minimalizovať import bez akéhokoľvek ďalšieho signálu.

### 6.3 PricingAwareReward

**Vzorec:**
$$r = -(\text{grid\_import} \times (1 + 2.0 \times \text{price}))$$

**Logika:** Penalizácia grid importu je **násobená cenou**. Keď je elektrina drahá, rovnaký odber stojí viac odmeny. Agent by sa mal naučiť:
- Nabíjať zásobníky keď je elektrina lacná
- Vybíjať keď je drahá

**Prečo `price_weight = 2.0`?** Cena z CityLearn datasetu je normalizovaná a jej variabilita je relatívne nízka. Váha 2.0 zabezpečuje, že cenový signál je dostatočne silný na to, aby agent vôbec rozlišoval medzi lacnými a drahými hodinami.

### 6.4 ComfortAwareReward

**Vzorec:**
$$r = -\Big(\text{grid\_import} \times \text{occ\_factor} \times \text{weather\_factor}
+ 6.0 \times \text{cooling\_delta} \times \text{discomfort\_factor}\Big)$$

kde:
- $\text{cooling\_delta}$ = `indoor_dry_bulb_temperature_cooling_delta` – miera prehriatia budovy
- $\text{discomfort\_factor} = 1 + 2.0 \times \mathbb{1}[\text{occupied}]$

**Logika:** Rozširuje WeatherOccupancyReward o **explicitnú penalizáciu za diskomfort**. Cooling delta je kladná, keď je vnútorná teplota vyššia, než by bolo pohodlné. Keď sú v budove ľudia, diskomfortná penalizácia sa zdvojnásobí.

**Prečo `discomfort_weight = 6.0`?** Pri nižších váhach agent ignoruje komfort a len minimalizuje import. Hodnota 6.0 bola zvolená empiricky – pri nej agent začne aktívne chladiť počas obsadených hodín, aj za cenu vyššieho importu.

**Trade-off:** Tento reward explicitne obetuje energetické úspory za komfort. V porovnávacej tabuľke by mal mať **vyšší grid import** ale **nižší discomfort_proportion** oproti energy-only.

### 6.5 PeakShavingReward

**Vzorec:**
$$r = -(\text{grid\_import})^2$$

**Logika:** Kvadratická penalizácia dramaticky zvyšuje cenu veľkých odberov:

| Import (kWh) | Lineárna penalizácia | Kvadratická penalizácia |
|---|---|---|
| 1 | -1 | -1 |
| 2 | -2 | -4 |
| 5 | -5 | -25 |
| 10 | -10 | -100 |

Agent sa naučí **rozkladať spotrebu v čase** namiesto toho, aby mal veľké špičky. Pre distribučnú sieť je to žiaduce, pretože špičky vyžadujú drahú infraštruktúru.

**Očakávané správanie:** Nižšie maximum odberu, vyrovnanejší profil, ale nie nevyhnutne najnižší celkový import (agent môže mať vyšší priemerný odber, len bez špičiek).

### 6.6 SolarAlignmentReward

**Vzorec:**
$$r = -\Big(\text{grid\_import} \times (1 - 0.3 \times \text{normalized\_solar})\Big)$$

kde:
$$\text{normalized\_solar} = \min\Big(\frac{\text{diffuse}_1 + \text{direct}_1}{800}, 1\Big)$$

**Logika:** Keď je veľa solárnej energie (blízko 800 W/m²), multiplikátor klesá na 0.7, čím sa znižuje penalizácia za import. Keď slnko nesvieti, multiplikátor je 1.0 – plná penalizácia.

**Prečo to dáva zmysel?** Ak budova má solárne panely, import počas slnečných hodín je čiastočne pokrytý lokálnou výrobou. Agent by mal:
- V slnečných hodinách využívať elektrinu (alebo nabíjať zásobníky)
- V noci a zamračených dňoch šetriť a vybíjať zásobníky

**Konštanta 800:** Typická maximálna irradiácia pri jasnom počasí v miernom pásme. Slúži na normalizáciu do [0, 1].

**`solar_weight = 0.3`:** Ak by bol 1.0, agent by mal v slnečných hodinách nulový trest za import, čo by bolo príliš agresívne. 0.3 je konzervatívne zníženie.

### 6.7 StorageManagementReward

**Vzorec:**
$$r = -(\text{grid\_import} + 2.0 \times \text{soc\_waste})$$

kde:
$$\text{soc\_waste} = \begin{cases}
\max(0.5 - \text{avg\_soc}, 0) & \text{ak } \text{price} > 0.3 \\
0.5 \times \max(\text{avg\_soc} - 0.8, 0) & \text{ak } \text{price} \le 0.3
\end{cases}$$

a $\text{avg\_soc} = (\text{dhw\_soc} + \text{elec\_soc}) / 2$.

**Logika:** Priamo penalizuje **zlé stavy zásobníkov** v závislosti od ceny:

1. **Drahá hodina + nízky SOC → penalta**: Agent nemal zásoby keď ich potreboval. Prah SOC 0.5 = zásobníky by mali byť aspoň na polovicu
2. **Lacná hodina + preplnený SOC → penalta**: Agent nemôže nabiť lacno, pretože zásobníky sú už plné. Prah 0.8 = nad 80% je "zbytočne plné"

**Asymetria váh (0.5× pre plný SOC):** Je horšie nemať zásoby keď sú potrebné než mať ich trochu navyše. Preto je penalta za "premoc nabíjanie" polovičná.

**Prah ceny 0.3:** Empirický odhad "drahej" hodiny z distribúcie cien v datasete.

### 6.8 RampingPenaltyReward

**Vzorec:**
$$r = -(\text{grid\_import} + 0.5 \times |\text{grid\_import}_t - \text{grid\_import}_{t-1}|)$$

**Logika:** Okrem samotného importu penalizuje aj **veľké skoky** v odbere medzi po sebe nasledujúcimi krokmi. Pre distribučnú sieť je plynulý odber jednoduchší na reguláciu než prudké výkyvy.

**Implementačná poznámka:** Trieda si v `_prev_imports` dict uchováva posledný import pre každú budovu (index `i`). Na začiatku prvého kroku sa za "predchádzajúci" import berie aktuálny (aby prvý krok nebol umelo penalizovaný).

**`ramping_weight = 0.5`:** Import zostáva dominantným členom, ramping je doplnkový cieľ. Vyššie váhy by mohli viesť k tomu, že agent radšej importuje konštantne veľa, len aby nemal výkyvy.

### 6.9 TimeOfUseReward

**Vzorec:**
$$r = -(\text{grid\_import} \times \text{multiplier})$$

kde:
$$\text{multiplier} = \begin{cases}
3.0 & \text{ak } 17 \le h \le 21 \text{ (špička)} \\
0.5 & \text{ak } h \ge 22 \text{ alebo } h \le 6 \text{ (noc)} \\
1.0 & \text{inak (off-peak)}
\end{cases}$$

**Logika:** **Nezávisí na electricity_pricing** – je to čisto pevný časový rozvrh, ktorý simuluje Time-of-Use tarif. Agent sa učí:
- Vyhnúť sa importu v špičkových hodinách (17:00–21:00) → 3× trest
- Nabíjať zásobníky v noci (22:00–06:00) → len 0.5× trest

**Prečo nepoužiť priamo `electricity_pricing`?** Na to je PricingAwareReward. TimeOfUseReward testuje, či čisto fixný rozvrh (bez real-time cien) môže byť rovnako účinný. Niektoré energetické trhy používajú práve TOU tarify.

**Implementačná poznámka:** Hodina dňa sa počíta ako `self._step % 24`. Counter `_step` sa inkrementuje pri každom volaní `calculate()`. CityLearn dataset má hodinové kroky, takže `step % 24` korektne mapuje na hodinu dňa.

### 6.10 SelfSufficiencyReward

**Vzorec:**
$$r = \begin{cases}
-\text{net} & \text{ak } \text{net} > 0 \text{ (import)} \\
0.5 \times |\text{net}| & \text{ak } \text{net} \le 0 \text{ (export)}
\end{cases}$$

**Logika:** Na rozdiel od všetkých ostatných reward funkcií, táto **odmeňuje export** (záporný net_electricity_consumption). Keď budova vyrába viac než spotrebuje, agent dostáva pozitívnu odmenu.

**`export_bonus = 0.5`:** Export je odmenený menej než import je penalizovaný (0.5 vs 1.0). To zabezpečuje, že agent neoptimalizuje čisto na maximalizáciu exportu (čo by znamenalo nedostatok energie pre budovu).

**Očakávané správanie:** Agent sa snaží:
- Minimalizovať import (ako všetci)
- ALE navyše aktívne vybíjať zásobníky keď nie je potreba, čím "dodáva" do siete

### 6.11 CombinedMultiObjectiveReward

**Vzorec:**
$$r = -\Big(\text{grid\_import}
+ 1.0 \times \text{grid\_import} \times \text{price}
+ 3.0 \times \text{cooling\_delta}
+ 0.1 \times \text{grid\_import}^2\Big)$$

**Logika:** Kombinuje **štyri ciele** do jednej reward funkcie:

1. **Lineárny import** – základná penalizácia
2. **Pricing** – import × cena (cenový arbitráž)
3. **Komfort** – cooling_delta (prehrievanie)
4. **Peak shaving** – import² (vyhladzovanie špičiek)

**Prečo rôzne váhy?**
- `price_weight = 1.0` – cenový signál rovnocenný s importom
- `comfort_weight = 3.0` – komfort je 3× dôležitejší než kWh (ľudia > elektrina)
- `peak_weight = 0.1` – peak shaving je jemný doplnok, nie dominantný cieľ

**Trade-off:** Multi-objektívny reward je najkomplexnejší signál pre agenta. Riziko: agent nemusí vedieť, na čo sa má zamerať, keď sú ciele v konflikte (napr. lacná elektrina v noci vs. komfort cez deň). Výhodou je, že ak váhy sú dobre nastavené, agent nájde **kompromisné riešenie** lepšie než akýkoľvek single-objective variant.

### 6.12 NightPrechargeReward

**Vzorec:**
$$r = \begin{cases}
-0.4 \times \text{grid\_import} + 1.5 \times \text{avg\_soc} & \text{ak } 0 \le h \le 6 \text{ (noc)} \\
-2.0 \times \text{grid\_import} + 0.75 \times \text{avg\_soc} & \text{ak } 17 \le h \le 21 \text{ (špička)} \\
-\text{grid\_import} & \text{inak}
\end{cases}$$

**Logika:** Explicitne kóduje stratégiu **"nabij v noci, spotrebuj v špičke"**:

- **Noc (0–6h):** Import je penalizovaný len na 40% (nabíjanie je lacné). Navyše agent dostáva **bonus za plné zásobníky** – čím vyšší SOC, tým väčšia odmena.
- **Špička (17–21h):** Import je penalizovaný 2×. Agent stále dostáva bonus za SOC (ale len na 50% sily), čo ho motivuje mať zásoby aj počas špičky.
- **Mimo špičky:** Štandardná lineárna penalizácia.

**`soc_bonus = 1.5`:** Relatívne vysoká hodnota – v noci sa bonus za plný SOC (avg_soc=1.0 → +1.5) môže prekryť penaltu za import (napr. import 3 kWh → -1.2). To je zámer – agent by mal aktívne nabíjať v noci aj za cenu krátkodobého importu.

**Filozofia:** Tento reward je "najexplicitnejší" – priamo hovorí agentovi stratégiu. Výhoda: rýchle naučenie. Nevýhoda: ak stratégia nie je optimálna, agent nemá priestor objaviť lepšiu.

---

## 7. Referenčná stratégia – FixedPolicy

```python
FixedPolicy(dhw_action=0.0, electrical_action=0.0, cooling_action=0.5)
```

Fixná politika robí vždy to isté:
- **DHW = 0.0** – zásobník teplej vody sa nepoužíva (žiadne nabíjanie/vybíjanie)
- **Batéria = 0.0** – elektrická batéria sa nepoužíva
- **Chladenie = 0.5** – stredný výkon chladiaceho zariadenia

Slúži ako **baseline** – všetky Q-learning varianty sa porovnávajú voči nej cez metriku `savings_vs_fixed_pct`.

**Prečo tieto konkrétne hodnoty?** Cooling 0.5 reprezentuje "rozumné predvolené nastavenie", ktoré by bežný správca budovy mohol použiť. Nulové zásobníkové akcie znamenajú, že fixná stratégia nevyužíva flexibilitu – presne to, čo Q-learning má zlepšiť.

---

## 8. Tréningová slučka

### 8.1 Priebeh jednej epizódy

```
1. env.reset() → počiatočné pozorovania
2. agent.reset() → vyčistenie predchádzajúcich stavov
3. CYKLUS kým nie je terminated:
   a. agent.predict(obs) → akcie (ε-greedy)
   b. env.step(akcie) → nové obs, rewards, terminated
   c. agent.update(rewards, nové_obs, terminated) → TD update Q-tabuľky
4. agent.finish_episode(reward_history) → epsilon decay + adaptívny boost
```

### 8.2 Tréningové parametre

- **200 epizód** (default) pre každý reward variant
- Každá epizóda = jeden úplný prechod cez dataset (~720 hodinových krokov = 30 dní)
- Progress sa vypisuje každých `episodes // 20` epizód (t.j. ~každých 10 epizód pri 200)

### 8.3 Detekcia stability

Funkcia `estimate_stability_episode()` hľadá bod, kde sa učenie "stabilizovalo":

1. Používa **kĺzavé okno** 10 epizód
2. Porovnáva priemer posledných 10 epizód s priemerom predchádzajúcich 10
3. Ak je relatívna zmena ≤ 3% **a** smerodajná odchýlka ≤ 4.5% priemeru → stabilné

**Prečo je to dôležité?** Stability episode hovorí, koľko tréningových dát agent potrebuje. Ak sa jeden reward variant stabilizuje na ep. 50 a iný na ep. 150, druhý je buď náročnejší na učenie alebo osciluje.

---

## 9. Vyhodnotenie a metriky

### 9.1 Zbierané metriky

| Metrika | Popis |
|---|---|
| `total_grid_import_kwh` | Celkový import zo siete za evalúaciu |
| `total_net_consumption_kwh` | Čistá spotreba (import – export) |
| `discomfort_proportion` | Podiel krokov s diskomfortom (prehrievanie) |
| `cumulative_reward` | Celková odmena za evalúaciu (závisí od reward funkcie!) |
| `savings_vs_fixed_pct` | Percentuálna úspora grid importu oproti fixnej stratégii |
| `training_seconds` | Čas tréningu |
| `stability_episode` | Kedy sa tréning stabilizoval |
| `last_10_episode_reward_mean` | Priemerná odmena z posledných 10 epizód |

**Dôležitá poznámka o `cumulative_reward`:** Táto metrika NIE je porovnateľná medzi rôznymi reward funkciami! Každá reward funkcia má inú škálu. Porovnávať sa dajú len `grid_import_kwh`, `savings_vs_fixed_pct` a `discomfort_proportion`.

### 9.2 Evalúacia po tréningu

Po skončení tréningu sa agent spustí ešte raz s `deterministic=True` (bez explorácie). Tento "čistý" beh sa používa na porovnanie – agent vždy vyberie akciu s najvyššou Q-hodnotou.

---

## 10. Výstupné súbory

Všetky výstupy sa ukladajú do `outputs_q_learning1804_05/`:

| Súbor | Formát | Popis |
|---|---|---|
| `summary_results.csv` | CSV | Zhrnutie všetkých 13 politík (1 fixed + 12 Q-learning) |
| `policy_comparison.png` | PNG | 4-panelový porovnávací graf všetkých politík |
| `trajectory_fixed_strategy.csv` | CSV | Krokový záznam fixnej politiky |
| `kpis_fixed.csv` | CSV | KPI metriky z CityLearn evaluácie (fixed) |
| `trajectory_q_learning_{key}.csv` | CSV | Krokový záznam pre daný reward variant |
| `learning_trace_{key}.csv` | CSV | Priebeh učenia (episode_reward, epsilon per epizóda) |
| `q_table_{key}.npy` | NumPy | Natrénovaná Q-tabuľka (590,490 float32 hodnôt) |
| `kpis_{key}.csv` | CSV | KPI metriky z CityLearn evaluácie |
| `reward_time_and_learning_comparison_{key}.png` | PNG | 4-panelový graf: step reward, cumulative reward, learning progress, cumulative import |

kde `{key}` ∈ {`weather`, `energy`, `pricing`, `comfort`, `peak`, `solar`, `storage`, `ramping`, `tou`, `selfsuff`, `combined`, `nightpre`}.

**Celkovo:** 1 summary CSV + 1 porovnávací graf + 2 fixné CSV + 12 × (trajectory CSV + learning trace CSV + Q-table NPY + KPIs CSV + graf PNG) = **63 súborov**.

---

## 11. Grafy

### 11.1 policy_comparison.png

Obsahuje 4 sub-grafy:

1. **Total grid import** (bar chart) – absolútny import pre každú politiku
2. **Savings vs fixed strategy** (bar chart) – percentuálna úspora
3. **Discomfort proportion** (bar chart) – podiel diskomfortu
4. **Average grid import over 14-day profile** (line chart) – priemerný hodinový profil importu

### 11.2 Per-reward porovnávacie grafy

Každý reward variant dostáva vlastný 4-panelový graf:

1. **Step reward** – okamžitá odmena v čase (fixed vs Q-learning)
2. **Cumulative reward** – kumulovaná odmena
3. **Learning progress** – kĺzavý priemer odmeny cez epizódy (krivka učenia)
4. **Cumulative grid import** – celkový import v čase

### 11.3 Farebná paleta

13 farieb bolo vybraných tak, aby boli vizuálne rozlíšiteľné aj pri väčšom počte kriviek:

| Politika | Farba | Hex |
|---|---|---|
| Fixed | Sivá | #9aa0a6 |
| Weather | Tmavomodrá | #1d3557 |
| Energy | Teal | #2a9d8f |
| Pricing | Koralová | #e76f51 |
| Comfort | Fialová | #6d597a |
| PeakShaving | Petrol | #264653 |
| Solar | Zlatá | #e9c46a |
| Storage | Piesková | #f4a261 |
| Ramping | Olivová | #606c38 |
| TimeOfUse | Hnedá | #bc6c25 |
| SelfSufficiency | Polnočná modrá | #023047 |
| Combined | Červená | #d62828 |
| NightPrecharge | Nebeská modrá | #219ebc |

---

## 12. Architektúra kódu

### 12.1 Diagram závislostí

```
main()
 └─ run_experiment()
     ├─ make_env() → CityLearnEnv
     ├─ FixedPolicy → run_policy()
     ├─ pre každý REWARD_CONFIG:
     │   ├─ make_env(reward_function=cls)
     │   ├─ OwnAdaptiveTabularQLearning(env)
     │   │   ├─ ObservationDiscretizer(env)
     │   │   └─ MultiActionDiscretizer(env)
     │   ├─ train_q_learning(agent, env)
     │   └─ run_policy(agent, env)
     ├─ build_results_frame()
     ├─ save_policy_comparison_figure()
     └─ save_time_and_learning_comparison() × 12
```

### 12.2 Refaktoring oproti predchádzajúcim verziám

V Q_learning1804_03 a 1804_04 bol každý reward variant implementovaný ako separátny blok copy-paste kódu v `run_experiment()`. Toto sa stávalo neudržateľným pri 12 variantoch. Preto:

- Vytvorený zoznam `REWARD_CONFIGS` s n-ticami `(kľúč, názov, trieda)`
- `run_experiment()` iteruje cez `REWARD_CONFIGS` v jednom for-cykle
- Výhodou je, že pridanie nového variantu vyžaduje len: 1) definovať triedu, 2) pridať riadok do `REWARD_CONFIGS`
- Zakomentovaním riadku v `REWARD_CONFIGS` sa variant preskočí (bez mazania kódu)

---

## 13. Spustenie programu

### 13.1 Základné spustenie

```bash
python Q_learning1804_05.py
```

### 13.2 S vlastnými parametrami

```bash
python Q_learning1804_05.py \
  --episodes 100 \
  --buildings Building_1 Building_2 \
  --baseline-cooling 0.7 \
  --random-seed 42 \
  --output-dir outputs_q_learning1804_05_custom
```

### 13.3 CLI argumenty

| Argument | Default | Popis |
|---|---|---|
| `--schema` | `data/.../schema.json` | Cesta k CityLearn schema |
| `--buildings` | Building_1, 2, 3 | Zoznam budov |
| `--episodes` | 50 | Počet tréningových epizód |
| `--baseline-cooling` | 0.5 | Cooling akcia fixnej stratégie |
| `--random-seed` | 7 | Random seed pre reprodukovateľnosť |
| `--comparison-horizon` | 719 | Počet krokov pre vizuálne porovnanie |
| `--output-dir` | `outputs_q_learning1804_05` | Výstupný priečinok |

---

## 14. Očakávané výsledky a interpretácia

### 14.1 Čo porovnávať

1. **Grid import** – Ktorý reward vedie k najnižšiemu importu? Pravdepodobne Energy-only alebo PeakShaving.
2. **Savings** – Koľko percent ušetrí Q-learning oproti fixnej stratégii?
3. **Discomfort** – ComfortAware a Combined by mali mať najnižší diskomfort, ale na úkor vyššieho importu.
4. **Profil odberu** – PeakShaving a Ramping by mali mať najplochejší profil, NightPrecharge najvýraznejšie nočné špičky.

### 14.2 Predpokladané trade-offy

| Reward | Import ↓ | Discomfort ↓ | Flat profil | Nočné nabíjanie |
|---|---|---|---|---|
| Energy-only | ★★★ | ★ | ★ | ★ |
| PeakShaving | ★★ | ★ | ★★★ | ★ |
| Comfort | ★ | ★★★ | ★ | ★ |
| NightPrecharge | ★★ | ★ | ★ | ★★★ |
| Combined | ★★ | ★★ | ★★ | ★★ |
| Solar | ★★ | ★ | ★ | ★★ |

### 14.3 Prečo niektoré reward funkcie nemusia fungovať dobre

- **TimeOfUse** – Pevný rozvrh nemusí zodpovedať skutočným cenovým špičkám v datasete
- **SelfSufficiency** – Ak budovy nemajú solárne panely, export je obmedzený a bonus je nedosiahnuteľný
- **RampingPenalty** – Ak optimálna stratégia vyžaduje rýchle prepnutie (napr. z nabíjania na vybíjanie), ramping penalty to penalizuje
- **StorageManagement** – Prahy (0.3 pre cenu, 0.5/0.8 pre SOC) nemusia zodpovedať distribúcii hodnôt v datasete

---

## 15. Obmedzenia a možné rozšírenia

### 15.1 Obmedzenia

1. **Tabuľkový Q-learning** – Stavy sú diskretizované, čo stráca informáciu. Deep Q-learning by mohol pracovať s kontinuálnymi stavmi.
2. **Zdieľaná Q-tabuľka** – Všetky 3 budovy zdieľajú jednu Q-tabuľku, čo predpokladá, že majú podobné správanie.
3. **Single-step TD** – Nepoužíva sa eligibility traces ani n-step returns, čo spomaľuje propagáciu odmeny.
4. **Žiadna validácia hyper-parametrov** – Váhy v reward funkciách boli nastavené intuitívne/empiricky, nie systematickým grid searchom.
5. **Step counter v reward funkciách** – TimeOfUse a NightPrecharge používajú interný `_step` counter na výpočet hodiny dňa. Ak sa prostredie resetuje inak než na začiatok dňa, counter sa rozsynchonizuje. V CityLearn datasete to nie je problém (reset na začiatok periódy).

### 15.2 Možné rozšírenia

- **Per-building Q-tabuľky** – Každá budova má vlastnú tabuľku, umožňuje špecializáciu
- **Váhový sweep** – Systematické testovanie rôznych váh pre Combined reward
- **Transfer learning** – Použiť Q-tabuľku natrénovanú jedným rewardom ako inicializáciu pre iný
- **Multi-agent** – Budovy spolu kooperujú cez centrálny reward
