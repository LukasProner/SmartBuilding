# test2.py notes (jednoducha RL ukazka)

Tento dokument vysvetluje, co sa deje v `test2.py`, aby si tomu vedel lepsie rozumiet krok po kroku.

## 1) Co je ciel skriptu

Ciel je naucit jednoducheho RL agenta, aby riadil bateriu pocas 24 hodin tak, aby sa znizili naklady na elektrinu.

- Agent vidi hodinu a stav nabitia baterie (SOC).
- Agent vybera 1 z 3 akcii: vybijat, nic, nabijat.
- Odmena je zaporna cena elektriny (teda cim nizsia cena, tym lepsia odmena).

## 2) Kniznice a preco su tam

- `torch`: na tensor pre Q-tabulku a moznost bezat na CUDA, ak je dostupna.
- `numpy`: profily spotreby/cien, diskretizacia SOC, pomocne vypocty.
- `matplotlib`: graf priebehu ucenia.
- `tqdm`: progress bar pocas trenovania.
- `time`, `timedelta`: meranie casu treningu.

Poznamka: je to stale tabulkovy Q-learning, nie neuronova siet. `torch` je pouzity hlavne ako pohodlny backend (CPU/CUDA).

## 3) SimpleEnergyEnv (vlastne mini prostredie)

Trieda `SimpleEnergyEnv` simuluje 1 den (24 krokov):

- `demand_profile`: kolko energie budova potrebuje kazdu hodinu.
- `price_profile`: cena elektriny pre kazdu hodinu.
- `capacity_kwh`: kapacita baterie.
- `max_power_kwh`: max nabijanie/vybijanie za hodinu.
- `charge_eff`, `discharge_eff`: ucinnost baterie.

### reset()

- Nastavi `hour = 0`.
- Nastavi SOC baterie na 50 %.
- Vrati pociatocny stav `(hour, soc)`.

### step(action_idx)

Pre 1 krok prostredia:

1. Nacita aktualny dopyt a cenu podla hodiny.
2. Podla akcie vyrata nabijanie/vybijanie.
3. Aktualizuje SOC s ucinnostou a orezanim na rozsah `<0, capacity>`.
4. Spocita odber zo siete:
   - `grid_import = max(0, demand + charge - discharge)`
5. Spocita cenu:
   - `cost = grid_import * price`
6. Odmena:
   - `reward = -cost`
7. Posunie cas o 1 hodinu a vrati:
   - `next_state`, `reward`, `done`, `info`.

`done` je `True` po 24. kroku.

## 4) Stav, akcie a diskretizacia

### Stav

Pouzivame:

- `hour` (0..23)
- `soc_bin` (diskretizovany SOC)

### Akcie

- `0` = vybije bateriu (ak je co vybit)
- `1` = nerobi nic
- `2` = nabije bateriu (ak je miesto)

### soc_to_bin(...)

Funkcia `soc_to_bin` premapuje realny SOC do indexu `0..NUM_SOC_BINS-1`.

## 5) Q-learning cast

Q-tabulka ma tvar:

- `[hour, soc_bin, action]`

Inicializacia:

- `q_table = torch.zeros((24, NUM_SOC_BINS, NUM_ACTIONS), device=device)`

### choose_action(...)

- S pravdepodobnostou `epsilon` vyberie nahodnu akciu (exploration).
- Inak vyberie `argmax(Q)` (exploitation).

### Update pravidlo

Po kroku sa robi TD update:

- Ak je `done`:
  - `td_target = reward`
- Inak:
  - `td_target = reward + GAMMA * max_a' Q(next_state, a')`

Potom:

- `Q(s,a) = Q(s,a) + ALPHA * (td_target - Q(s,a))`

## 6) Epsilon plan

- Zacina na `EPSILON_START`.
- Kazdu epizodu sa nasobi `EPSILON_DECAY`.
- Nikdy nejde pod `EPSILON_END`.

To znamena:

- Na zaciatku viac skusa nahodne akcie.
- Neskor viac pouziva naucene spravanie.

## 7) Baseline vs learned policy

Skript porovnava:

- `run_baseline_episode`: bateria je stale idle (akcia 1).
- `run_greedy_episode`: agent berie najlepsiu akciu z Q-tab.

Vysledok:

- `Baseline cost`
- `Learned policy cost`
- `Savings %`

Ak je `Savings %` kladne, agent je lepsi ako baseline.

## 8) CUDA/CPU

`resolve_device()` vypise:

- `Device: CUDA | GPU: ...` ak je dostupna CUDA,
- inak `Device: CPU | CUDA not available`.

Q-tabulka bezi na vybranom zariadeni.

Dolezite:

- Toto mini prostredie je velmi male, takze CUDA vacsinou neprinesie velke zrychlenie.

## 9) Grafy

Po treningu sa kresli:

- episode return (sivy priebeh)
- moving average (hladsi trend)

Interpretacia:

- Return je zaporna cena, takze "vyssie" (menej zaporne) je lepsie.

## 10) Co menit pri experimentoch

Bezpecne miesta na experimentovanie:

1. `NUM_EPISODES`
- viac epizod = viac sanca naucit sa lepsiu politiku

2. `NUM_SOC_BINS`
- viac binov = jemnejsi stav, ale vacsia tabulka

3. `ALPHA`
- prilis vysoke: nestabilne
- prilis nizke: pomale ucenie

4. `GAMMA`
- vyssie = viac mysli na buducnost

5. `EPSILON_DECAY`
- pomalsi decay = dlhsie skumanie

6. profily v prostredi
- `demand_profile` a `price_profile` silno ovplyvnia, co sa agent nauci

## 11) Typicky workflow na cviceni

1. Spusti skript a pozri baseline vs learned.
2. Zvys/potom zniz `NUM_EPISODES`.
3. Zmen `NUM_SOC_BINS` a porovnaj stabilitu.
4. Pozri grafy, ci trend ide spravnym smerom.
5. Az potom prechadzaj na zlozitejsie RL (napr. DQN/DDPG).

## 12) Co je dalsi logicky krok

Ak chces plynuly prechod od tabulkoveho Q-learningu:

- najprv pridat viac stavov (napr. price bin, demand bin),
- potom prejst na DQN (neuralna siet namiesto tabulky),
- az potom riesit DDPG (spojite akcie).
