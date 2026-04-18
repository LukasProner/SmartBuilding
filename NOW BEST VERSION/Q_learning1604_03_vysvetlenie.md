# Vysvetlenie: Q_learning1604_03.py

Tento dokument vysvetľuje každú dôležitú časť skriptu `Q_learning1604_03.py` od začiatku do konca. Cieľom je porozumieť **prečo** bola každá vec urobená tak, ako bola — nielen **čo** robí.

---

## 1. Čo skript celkovo robí

Skript trénuje **tabulkový Q-learning agenta**, ktorý sa naučí riadiť tri zariadenia v jednej budove:

- **DHW zásobník** (dhw = Domestic Hot Water — zásobník teplej vody)
- **Elektrická batéria** (electrical storage)
- **Chladiace zariadenie** (cooling device)

Cieľom je **minimalizovať odber elektrickej energie zo siete** (grid import), pričom berie do úvahy obsadenosť budovy a predpoveď počasia.

Po tréningu skript porovná naučeného agenta s **fixnou stratégiou** — teda politikou, ktorá vždy robí rovnakú akciu bez ohľadu na stav.

---

## 2. Prostredie: CityLearn

### Čo je CityLearn?

CityLearn je open-source Python knižnica navrhnutá špeciálne pre výskum riadenia budov pomocou posilňovacieho učenia (Reinforcement Learning). Simuluje jednu alebo viacero budov s reálnymi dátami o spotrebe, počasí, obsadenosti atď.

Dokumentácia: https://intelligent-environments-lab.github.io/CityLearn/

### Dataset

```python
DEFAULT_SCHEMA = PROJECT_ROOT / 'data' / 'datasets' / 'citylearn_challenge_2023_phase_1' / 'schema.json'
```

Používa sa dataset z **CityLearn Challenge 2023 Phase 1**. Schema je JSON súbor, ktorý popisuje:
- ktoré budovy sú dostupné,
- aké dáta (počasie, ceny, obsadenosť...) obsahuje každý časový krok,
- aké zariadenia má každá budova.

Jeden **časový krok = 1 hodina**. Celý dataset pokrýva **1 rok (8760 hodín)**.

### Prečo iba jedna budova?

```python
DEFAULT_BUILDING = 'Building_1'
```

Tabulkový Q-learning má omezenú škálovateľnosť — stavový a akčný priestor rastie exponenciálne s počtom budov. Pre jednu budovu je to ešte zvládnuteľné (2880 stavov × 45 akcií). Pre viac budov by bolo potrebné použiť neurónové siete (deep RL).

---

## 3. Observácie (stav prostredia)

### Čo sú observácie v CityLearn?

Na každom časovom kroku CityLearn vráti agentovi vektor čísel — pozorovania (`observations`). Z nich agent spoznáva aktuálny stav sveta a rozhoduje o akcii.

CityLearn umožňuje vybrať si iba tie observácie, ktoré chceš — cez parameter `active_observations`. Tým sa zmenší vstupný priestor a agent sa učí rýchlejšie.

### Vybrané observácie

```python
ACTIVE_OBSERVATIONS = [
    'outdoor_dry_bulb_temperature_predicted_1',  # predpoveď teploty za 1 hodinu
    'outdoor_dry_bulb_temperature_predicted_2',  # predpoveď teploty za 2 hodiny
    'outdoor_dry_bulb_temperature_predicted_3',  # predpoveď teploty za 3 hodiny
    'dhw_storage_soc',                           # nabitie zásobníka teplej vody (0–1)
    'electrical_storage_soc',                    # nabitie elektrickej batérie (0–1)
    'occupant_count',                            # počet ľudí v budove
]
```

**Prečo práve tieto?**

- **Teplotné predpovede**: Chladiace zariadenie potrebuje vedieť, ako bude vonku teplo. Ak bude o 2 hodiny veľká horúčava, má zmysel začať chladiť skôr — to je tzv. *pre-cooling*. Tri hodiny dopredu dávajú agentovi časový horizont na plánovanie.
- **SOC zásobníka teplej vody (`dhw_storage_soc`)**: Agent musí vedieť, koľko teplej vody je v zásobníku, aby sa rozhodol, či ho ohrievať alebo nechať. Rozsah je 0 (prázdny) až 1 (plný).
- **SOC elektrickej batérie (`electrical_storage_soc`)**: Podobne — agent musí vedieť, koľko energie je v batérii, aby vedel, či ju nabíjať alebo vybíjať.
- **Počet ľudí (`occupant_count`)**: Viac ľudí = väčšia spotreba tepla, teplej vody aj chladu. Táto informácia ovplyvňuje reward funkciu a teda aj to, ako veľmi záleží na optimalizácii v danom čase.

### Prečo NIE sú zahrnuté iné observácie (napr. hodina, slnečné žiarenie)?

Skript záámerne vynechal hodinu dňa, aby ukázal, že samo-organized správanie môže vyplynúť z fyzikálnych signálov (teplotné predpovede). Vynechanie slnečného žiarenia je kompromis — s 6 observáciami je stav ešte diskretizovateľný do rozumnej veľkosti.

---

## 4. Diskretizácia stavového priestoru

### Prečo je to potrebné?

Q-learning v klasickej tabulkovej forme potrebuje **konečný, diskrétny** stavový priestor. Observácie sú však spojité čísla (napr. teplota = 18.7°C). Preto ich musíme zaokrúhliť do "košov" (bins).

### Ako to funguje — trieda `ObservationDiscretizer`

```python
class ObservationDiscretizer:
    def __init__(self, env: CityLearnEnv, bin_counts: dict[str, int]):
        ...
        for low, high, count in zip(env.observation_space[0].low, env.observation_space[0].high, self.bin_counts):
            self.edges.append(np.linspace(float(low), float(high), count + 1)[1:-1])
```

Pre každú observáciu sa zoberú jej minimálna a maximálna hodnota z `observation_space` prostredia a rovnomerne sa rozdelí na `n` košov pomocou `np.linspace`. Hranice košov sa uložia ako `edges`.

**Príklad pre teplotu (5 košov):**
- Rozsah: -5°C až 45°C
- Hrany: [-5, 5, 15, 25, 35, 45] → 5 intervalov
- Ak je teplota 22°C, padne do koša č. 3

### Koľko košov pre každú observáciu?

```python
OBSERVATION_BIN_SIZES = {
    'outdoor_dry_bulb_temperature_predicted_1': 5,
    'outdoor_dry_bulb_temperature_predicted_2': 4,
    'outdoor_dry_bulb_temperature_predicted_3': 4,
    'dhw_storage_soc': 3,
    'electrical_storage_soc': 3,
    'occupant_count': 4,
}
```

- Najbližšia predpoveď teploty dostala **5 košov** — je najpresnejšia a najdôležitejšia pre okamžité rozhodnutie.
- Vzdialenejšie predpovede majú **4 koše** — menej presné, stačí hrubšia granularita.
- SOC zásobníkov sú **3 koše** — nízky / stredný / vysoký. Viac detailov by neprinieslo veľký úžitok.
- Obsadenosť **4 koše** — nikto, málo, stredne, veľa.

### Celková veľkosť stavového priestoru

$$5 \times 4 \times 4 \times 3 \times 3 \times 4 = 2880 \text{ stavov}$$

To je zvládnuteľný počet pre tabulkový Q-learning.

### Kódovanie stavu na jedno číslo — `encode()`

```python
def encode(self, observation: list[float]) -> int:
    digits = [int(np.digitize(float(v), e, right=False)) for v, e in zip(observation, self.edges)]
    return int(np.ravel_multi_index(tuple(digits), self.state_shape))
```

- `np.digitize` zaradí každú hodnotu do správneho koša.
- `np.ravel_multi_index` skonvertuje viacrozmerné indexy (napr. (3, 2, 1, 0, 2, 1)) na jedno celé číslo — index riadku v Q-tabuľke.

Je to ako preložiť GPS súradnice (6D) na číslo domu v katalógu.

---

## 5. Akcie

### Čo agent môže robiť?

```python
ACTIVE_ACTIONS = ['dhw_storage', 'electrical_storage', 'cooling_device']
```

V CityLearn sú akcie spojité čísla v rozsahu, ktorý závisí od zariadenia. Pre zásobníky je to typicky `[-1, 1]` (záporné = vybíjaj, kladné = nabíjaj). Pre chladenie `[0, 1]` (výkon chladenia od 0 do 100 %).

### Diskretizácia akčného priestoru — trieda `MultiActionDiscretizer`

Agent nemôže pracovať so spojitými akciami v tabulkovej forme. Preto sa každá dimenzia akcie rozdelí na niekoľko úrovní:

```python
ACTION_BIN_COUNTS = [3, 3, 5]
# dhw_storage:        3 hodnoty → [-1.0, 0.0, 1.0]   (vybíjaj / stoj / nabíjaj)
# electrical_storage: 3 hodnoty → [-1.0, 0.0, 1.0]   (vybíjaj / stoj / nabíjaj)
# cooling_device:     5 hodnoty → [0.0, 0.25, 0.5, 0.75, 1.0] (výkon chladenia)
```

### Joint action space — kartézsky súčin

Kľúčový koncept: agent nevyberá akciu pre každé zariadenie zvlášť. Vyberá **jednu spojenú akciu (joint action)** — kombináciu pre všetky tri zariadenia naraz.

```python
self.joint_actions = list(iterproduct(*self.value_grids))
```

Príklad joint akcií:
```
(-1.0, -1.0,  0.00)  → vybíjaj DHW, vybíjaj batériu, nechlaď
(-1.0,  0.0,  0.50)  → vybíjaj DHW, nič s batériou, polovičné chladenie
( 1.0,  1.0,  1.00)  → nabíjaj DHW, nabíjaj batériu, plné chladenie
...celkovo 3 × 3 × 5 = 45 kombinácií
```

Každá z 45 joint akcií je jeden **stĺpec v Q-tabuľke**.

---

## 6. Q-tabuľka

Q-tabuľka je **srdce tabulkového Q-learningu**. Je to dvojrozmerné pole (matica):

```
Q-tabuľka: [2880 stavov × 45 akcií] = 129 600 buniek
```

Každá bunka `Q[s, a]` obsahuje **odhadovanú celkovú odmenu**, ktorú agent získa, ak je v stave `s`, vykoná akciu `a` a potom bude postupovať optimálne.

Na začiatku sú všetky hodnoty 0. Počas tréningu sa postupne aktualizujú na základe skúseností.

```python
self.q_table = np.zeros(
    (self.observation_discretizer.state_count, self.action_discretizer.action_count),
    dtype=np.float32,
)
```

Prečo `float32` a nie `float64`? Ušetrí sa pamäť — pre 129 600 buniek je to iba 0,5 MB.

---

## 7. Reward funkcia (odmena)

### Prečo je reward dôležitý?

Reward je **jediný signál**, z ktorého sa agent učí. Definuje, čo je "dobré" a čo "zlé" správanie. Ak reward nie je dobre navrhnutý, agent sa naučí nesprávne veci.

### WeatherOccupancyReward

```python
class WeatherOccupancyReward(RewardFunction):
    def calculate(self, observations: list[dict]) -> list[float]:
        grid_import = max(observation['net_electricity_consumption'], 0.0)
        occupancy_factor = 1.0 + 1.5 * min(occupant_count / 5.0, 1.0)
        weather_factor  = 1.0 + 0.05 * max(mean_forecast - 24.0, 0.0)
        reward = -(grid_import * occupancy_factor * weather_factor)
```

V skratke: **reward = záporný odber zo siete, váhovaný obsadenosťou a teplotou**.

#### Čo je `net_electricity_consumption`?

V CityLearn toto je **čistá spotreba budovy zo siete** — teda koľko kWh v danej hodine budova odobrala od distribučnej siete (po zohľadnení vlastnej výroby slnečnej energie a batérie).

Používa sa `max(..., 0.0)` pretože záporná hodnota by znamenala export do siete (solárny prebytok). Exportovanie nie je penalizované — agent ho jednoducho ignoruje v odhadovanej strate.

#### `occupancy_factor` — prečo obsadenosť?

$$\text{occupancy\_factor} = 1 + 1.5 \times \min\left(\frac{\text{occupant\_count}}{5}, 1\right)$$

- Ak je budova prázdna (0 ľudí): faktor = 1.0 → normálna penalizácia
- Ak je budova plná (5+ ľudí): faktor = 1.0 + 1.5 = 2.5 → 2,5× väčšia penalizácia

**Dôvod:** Keď je veľa ľudí, je dôležitejšie zabezpečiť tepelný komfort — chladenie, teplá voda. Agent sa tak naučí optimalizovať energiu práve vtedy, keď na tom záleží.

Delenie 5-timi normalizuje obsadenosť na rozsah [0, 1]. `min(..., 1)` zaistí, že aj pri 10 ľuďoch faktor neprekročí 2.5.

#### `weather_factor` — prečo počasie?

$$\text{weather\_factor} = 1 + 0.05 \times \max(\bar{T}_{\text{forecast}} - 24, 0)$$

- Ak je predpoveď pod 24°C: faktor = 1.0 → normálna penalizácia
- Ak je predpoveď 34°C: faktor = 1.0 + 0.05 × 10 = 1.5 → 1,5× väčšia penalizácia

**Dôvod:** V horúcom počasí je spotreba na chladenie kritickejšia, energetické úspory majú väčší dopad na tepelný komfort aj na sieť (špičkové zaťaženie).

Hodnota 24°C je "prahová" pohodlná teplota — pod ňou sa chladenie takmer nepotrebuje.

#### Prečo je reward záporný?

Q-learning maximalizuje kumulatívnu odmenu. Keďže chceme **minimalizovať** spotrebu, odmena musí byť záporná — čím menej energy agent spotrebuje, tým menej záporná (teda lepšia) odmena dostane.

Ideálna odmena = 0 (nulová spotreba zo siete). Typická odmena = -5 až -20 za hodinu.

#### Ako sa reward integruje do CityLearn?

```python
class WeatherOccupancyReward(RewardFunction):
    super().__init__(env_metadata, **kwargs)
```

CityLearn má vlastný systém reward funkcií — triedu `RewardFunction`. Dedením z nej a implementáciou metódy `calculate()` skript "zaregistruje" svoju custom reward funkciu priamo do prostredia. CityLearn ju potom automaticky zavolá po každom kroku (`env.step()`).

To je elegantné riešenie — reward je oddelený od agenta a môže byť jednoducho vymenený za inú triedu.

---

## 8. Fixná referenčná politika

```python
class FixedPolicy:
    def __init__(self, dhw_action=0.0, electrical_action=0.0, cooling_action=0.5):
        self.actions = [dhw_action, electrical_action, cooling_action]

    def predict(self, observations, deterministic=None):
        return [list(self.actions) for _ in observations]
```

Fixná politika vždy vracia rovnaké akcie — bez ohľadu na stav:
- DHW zásobník: nič nerob (0.0)
- Batéria: nič nerob (0.0)
- Chladenie: 50% výkon (0.5)

**Prečo táto referencia?** Je to najjednoduchšia mysliteľná "stratégia". Akýkoľvek inteligentný agent by mal vedieť fungovať lepšie. Porovnanie s fixnou politikou teda ukazuje, o koľko percent Q-learning zlepšuje energetickú efektivitu.

---

## 9. Q-learning agent — `OwnAdaptiveTabularQLearning`

### 9.1 Štruktúra agenta

Agent drží:
- `q_table` — samotná Q-tabuľka
- `observation_discretizer` — konvertuje pozorovania → index stavu
- `action_discretizer` — konvertuje index akcie → skutočné hodnoty akcií
- `epsilon` — miera náhodného skúmania (explorácia)
- `learning_rate` (α) — ako rýchlo agent aktualizuje svoje odhady
- `discount_factor` (γ) — ako veľmi záleží na budúcich odmenách

### 9.2 Výber akcie — `predict()`

```python
def predict(self, observations, deterministic=False):
    state_index = self.observation_discretizer.encode(observations[0])
    if deterministic or self.random_state.rand() > self.epsilon:
        action_index = int(np.argmax(self.q_table[state_index]))
    else:
        action_index = int(self.random_state.randint(self.action_discretizer.action_count))
    return self.action_discretizer.decode(action_index)
```

Toto je **ε-greedy stratégia**:
- S pravdepodobnosťou `epsilon` → vyber **náhodnú akciu** (explorácia — skúšanie nových vecí)
- S pravdepodobnosťou `1 - epsilon` → vyber **najlepšiu akciu podľa Q-tabuľky** (exploitácia — využívanie naučeného)

Na začiatku tréningu je `epsilon = 1.0` (agent robí úplne náhodné akcie). Postupne klesá, agent sa čoraz viac spolieha na naučenú politiku.

`np.argmax(self.q_table[state_index])` vyberie index joint akcie s najvyššou hodnotou Q pre daný stav.

### 9.3 Aktualizácia Q-tabuľky — `update()` — TD(0) pravidlo

Toto je jadro Q-learningu:

```python
def update(self, reward, next_observations, terminated):
    next_state_index = self.observation_discretizer.encode(next_observations[0])
    best_next_value = 0.0 if terminated else float(np.max(self.q_table[next_state_index]))
    td_target = float(reward) + self.discount_factor * best_next_value
    td_error = td_target - float(self.q_table[self.last_state_index, self.last_action_index])
    self.q_table[self.last_state_index, self.last_action_index] += self.learning_rate * td_error
```

Vzorec v matematike:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \underbrace{\left[r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a)\right]}_{\text{TD chyba}}$$

Kde:
- $s$ = aktuálny stav, $a$ = vykonaná akcia
- $r$ = odmena za tento krok
- $s'$ = nový stav po akcii
- $\alpha$ = learning rate (= 0.15)
- $\gamma$ = discount factor (= 0.95)

**Čo to znamená intuitívne?**

Agent hovorí: "Myslel som, že z tohto stavu s touto akciou dostanem hodnotu `Q(s,a)`. Ale teraz som dostal odmenu `r` a vidím, že z ďalšieho stavu môžem dostať ešte `best_next_value`. Môj odhad bol o `td_error` nesprávny. Opravím ho o `learning_rate * td_error`."

Ak je `terminated = True` (koniec epizódy), `best_next_value = 0` — žiadna budúcnosť.

**Prečo `gamma = 0.95`?**

Discount factor hovorí, koľko sa cení budúca odmena oproti okamžitej:
- $\gamma = 1.0$ → budúce odmeny sú rovnako dôležité ako okamžité
- $\gamma = 0.0$ → záleží iba na okamžitej odmene

$\gamma = 0.95$ znamená, že odmena za 10 hodín je vážená $0.95^{10} \approx 0.60$ — teda asi 60 % okamžitej. Rozumný kompromis pre hodinové rozhodovanie.

### 9.4 Adaptívna explorácia — `finish_episode()`

```python
def finish_episode(self, reward_history):
    self.epsilon = max(self.minimum_epsilon,
                       self.epsilon_init * np.exp(-self.epsilon_decay * self.episode_index))
    if len(reward_history) >= 5:
        rolling_reward = np.mean(reward_history[-5:])
        if rolling_reward > self.best_rolling_reward + adaptive_min_improvement:
            self.best_rolling_reward = rolling_reward
            self.episodes_since_improvement = 0
        else:
            self.episodes_since_improvement += 1
        if self.episodes_since_improvement >= adaptive_patience:
            self.epsilon = min(0.35, self.epsilon + adaptive_epsilon_boost)
            self.episodes_since_improvement = 0
```

#### Exponenciálny decay epsilon

$$\epsilon(t) = \max\left(\epsilon_{\min},\ \epsilon_0 \cdot e^{-\lambda t}\right)$$

kde $t$ = číslo epizódy, $\lambda$ = `epsilon_decay` = 0.03.

Epsilon klesá exponenciálne — spočiatku rýchlo, neskôr pomaly. Tým sa zabezpečí, že agent na začiatku veľa skúma a neskôr využíva naučené.

#### Adaptívny boost epsilon

**Novinka oproti klasickému Q-learningu**: ak rolling reward posledných 5 epizód nerastie `adaptive_patience` (= 8) epizód po sebe, epsilon sa zvýši o `adaptive_epsilon_boost` (= 0.08), ale maximálne na 0.35.

**Prečo?** Agent mohol "uviaznutí v lokálnom minime" — naučiť sa suboptimálnu stratégiu a prestať objavovať lepšie alternatívy. Dočasné zvýšenie náhodnosti mu pomôže "vykopnúť sa" z tejto situácie.

---

## 10. Tréningová slučka — `train_q_learning()`

```python
for episode in range(episodes):
    observations, _ = env.reset()
    agent.reset()
    terminated = False

    while not terminated:
        actions = agent.predict(observations, deterministic=False)
        next_observations, rewards, terminated, _, _ = env.step(actions)
        agent.update(float(rewards[0]), next_observations, terminated)
        observations = next_observations

    agent.finish_episode(episode_rewards)
```

Jedna epizóda = jeden celý rok (8760 krokov). Každý krok:
1. Agent dostane pozorovania
2. Vyberie joint akciu (ε-greedy)
3. Prostredie vykoná akciu a vráti nové pozorovania + reward
4. Agent aktualizuje Q-tabuľku (TD update)
5. Opakuje sa až kým `terminated = True`

`env.reset()` vráti dvojicu `(observations, info)` — preto `observations, _ = env.reset()`. Podčiarknutie `_` znamená "tuto hodnotu nepotrebujem".

`env.step(actions)` vráti 5 hodnôt:
```python
next_observations, rewards, terminated, truncated, info = env.step(actions)
```
V kóde je použitý `_` na ignorovanie `truncated` a `info`.

---

## 11. Evaluácia politiky — `run_policy()`

```python
def run_policy(agent, env, deterministic=True):
    ...
    kpis = base_env.evaluate()
    discomfort_proportion = float(
        kpis[(kpis['name'] == building.name) & (kpis['cost_function'] == 'discomfort_proportion')]['value'].iloc[0]
    )
```

Táto funkcia spustí agenta v deterministic móde (bez náhody, `epsilon = 0`) na celý rok a zaznamenáva:
- `reward` každú hodinu
- `grid_import_kwh` každú hodinu
- kumulatívne hodnoty oboch

Navyše zavolá `env.evaluate()` — CityLearn vráti DataFrame s KPI (Key Performance Indicators) pre každú budovu. Jeden z nich je `discomfort_proportion` — podiel hodín, kedy bola tepelná pohoda mimo povolených hraníc.

**Prečo `base_env = env.unwrapped`?** CityLearn môže byť zabalený do wrapperov (napr. pre normalizáciu). `.unwrapped` vráti pôvodné prostredie, kde sú dostupné atribúty ako `buildings`.

---

## 12. Detekcia stability — `estimate_stability_episode()`

```python
def estimate_stability_episode(rewards, window=10, tolerance=0.03):
    for index in range(window * 2 - 1, len(reward_array)):
        prev = reward_array[index - 2*window + 1 : index - window + 1]
        curr = reward_array[index - window + 1 : index + 1]
        if abs(curr_mean - prev_mean) / scale <= tolerance and
           std(curr) / abs(curr_mean) <= tolerance * 1.5:
            return index + 1
    return None
```

Táto funkcia hľadá, od ktorej epizódy sa výkon agenta **stabilizoval** — teda kedy prestal výrazne rásť.

**Logika:**
- Vezme dve po sebe idúce okná posledných výsledkov (každé `window=10` epizód)
- Ak sa ich priemery líšia o menej ako 3% (`tolerance=0.03`) **a zároveň** rozptyl v aktuálnom okne je malý → toto je "bod stability"

Výsledok sa uloží ako `stability_episode` v `ExperimentResult` a zobrazí v grafe aj tabuľke.

---

## 13. Výstupy skriptu

Po behu skript uloží do priečinka `outputs_q_learning1604_03/`:

| Súbor | Obsah |
|---|---|
| `summary_results.csv` | Hlavná tabuľka: grid import, úspory %, stability episode, training time |
| `learning_trace.csv` | Reward a epsilon pre každú epizódu tréningu |
| `q_table.npy` | Uložená Q-tabuľka (numpy formát) |
| `trajectory_fixed_strategy.csv` | Hodinový priebeh fixnej politiky |
| `trajectory_q_learning.csv` | Hodinový priebeh Q-learning agenta |
| `reward_vs_fixed_comparison.png` | Bar chart: grid import, úspory, stability |
| `reward_time_and_learning_comparison.png` | 4 grafy: reward v čase, kumulatívny reward, krivka učenia, kumulatívny grid import |
| `monthly_summary.csv` | Mesačné porovnanie úspor |
| `monthly_comparisons/` | 12 grafov — jeden za každý mesiac |

---

## 14. Hlavná experimentálna slučka — `run_experiment()`

```python
def run_experiment(schema_path, building_name, episodes, baseline_cooling, random_seed, output_dir, comparison_horizon):
    # 1. Spusti fixnú politiku a zaznamenaj výsledky
    # 2. Natrénuj Q-learning agenta
    # 3. Spusti natrénovaného agenta (deterministic)
    # 4. Vypočítaj úspory vs fixná politika
    # 5. Ulož všetky výstupy
```

Toto je "dirigent" celého experimentu. Spustí oba pokusy na **rovnakom prostredí s rovnakým random seedom** — dôležité pre férovosť porovnania.

---

## 15. Schematický prehľad celého toku

```
schema.json
    │
    ▼
CityLearnEnv (1 budova, 3 akcie, 6 observácií, WeatherOccupancyReward)
    │
    ├──► FixedPolicy → run_policy() → fixed_run (baseline)
    │
    └──► OwnAdaptiveTabularQLearning
              │
              ├── ObservationDiscretizer (6D obs → 1 int, 2880 stavov)
              ├── MultiActionDiscretizer (3D akcie → 45 joint akcií)
              ├── Q-tabuľka [2880 × 45]
              │
              ├── train_q_learning() ─── 700 epizód ─── TD update + ε-decay + adaptive boost
              │
              └── run_policy() (deterministic) → learned_run
                       │
                       ▼
              Porovnanie: úspory kWh, discomfort, stability episode
                       │
                       ▼
              Grafy + CSV výstupy
```

---

## 16. Zhrnutie kľúčových dizajnových rozhodnutí

| Rozhodnutie | Dôvod |
|---|---|
| Iba 1 budova | Tabulkový Q-learning neškáluje na viac budov |
| 6 observácií vrátane SOC | SOC je nevyhnutný — bez neho agent nevie, koľko energie má k dispozícii |
| Teplotné predpovede (nie aktuálna teplota) | Umožňuje preemptívne správanie (pred-chladzovanie) |
| Joint action space (45 kombinácií) | Agent optimalizuje všetky 3 zariadenia súčasne, nie nezávisle |
| WeatherOccupancyReward | Penalizácia škálovaná obsadenosťou a teplotou — realitnejšia ako čistý -grid_import |
| Adaptívny epsilon boost | Vyhnutie sa uviaznúť v lokálnom minime |
| Trieda RewardFunction (dedenie) | Jednoduché vymenenie reward funkcie pre porovnanie |
| `estimate_stability_episode()` | Meria tretie kritérium zadania: čas do stabilného riešenia |
| 700 epizód | Stavový priestor 2880 × 45 = 129 600 buniek potrebuje dostatočné preskúmanie |
