# DDPGeasy notes (velmi jednoduche vysvetlenie)

Tento dokument vysvetluje subor [DDPGeasy.py](DDPGeasy.py) tak, aby to bolo zrozumitelne aj bez hlbsej informatiky.

## 1) Co je ciel skriptu

Ciel je jednoduchy:
- nacitat prostredie CityLearn pre scenar climate zone 1,
- spustit velmi kratky trening DDPG agenta,
- overit, ze vsetko bezi bez chyby.

DDPG je RL metoda pre spojite akcie (nie len akcie typu chod-vlavo/chod-vpravo).
V CityLearn to dava zmysel, lebo riadenie energie je spojite (male/zvacsenie/nizsie vykony).

## 2) Co robi kazda cast kodu

### Importy
Na zaciatku sa nacitaju kniznice:
- numpy: praca s cislami a poliami,
- torch: neuronove siete,
- CityLearnEnv: simulacne prostredie budovy.

### SCENARIO
Premenna SCENARIO hovori, ake CityLearn data sa maju nacitat:
- citylearn_challenge_2020_climate_zone_1

### Funkcie unwrap_obs, parse_reset, parse_step
Tieto 3 male funkcie su "adapter":
- unifikuju format dat z CityLearn,
- aby zvysok kodu bol jednoduchy a nemusel riesit rozdielne API verzie.

Priklad:
- niekde reset vrati len observation,
- inde vrati observation + info,
- tieto funkcie to zjednotia.

### Trieda Actor
Actor je siet, ktora povie:
- "aky krok (akciu) mam urobit v tomto stave?"

Vstup:
- stav prostredia (observations)

Vystup:
- navrh akcie v rozsahu <-1, 1> (cez Tanh)

### Trieda Critic
Critic je siet, ktora hodnoti:
- "aka dobra je tato akcia v tomto stave?"

Vstup:
- stav + akcia

Vystup:
- jedno cislo (Q hodnota), teda odhad kvality.

### main() - inicializacia
V main sa spravi:
- seed pre opakovatelnost,
- vytvorenie CityLearn prostredia,
- zistenie rozmerov stavu a akcie,
- vytvorenie actor/critic sieti,
- vytvorenie target kopii sieti,
- optimizerov (Adam).

Parametre su nastavene velmi jednoducho:
- episodes = 1
- max_steps = 50

To je naschval, aby to bolo rychle overenie, nie dlhy trening.

### Replay buffer
replay = deque(maxlen=5000)

Sem sa ukladaju prechody:
- (stav, akcia, reward, dalsi_stav, done)

Neskor sa z toho nahodne vzorkuje mini-batch.
To pomaha stabilite ucenia.

### Treningova slucka
Pre kazdu epizodu a krok:
1. Actor navrhne akciu.
2. Prida sa maly sum (noise), aby agent skusal aj ine akcie.
3. Akcia sa prevedie z <-1,1> na realne limity prostredia.
4. Zavola sa env.step(...).
5. Ulozi sa prechod do replay bufferu.
6. Ked je dost dat (batch), trenuje sa:
   - Critic: MSE chyba medzi odhadom a cielom y
   - Actor: snazi sa maximalizovat hodnotenie od Critica
7. Target siete sa jemne aktualizuju cez tau (soft update).

### Koniec
Na konci sa vypise:
- reward za epizodu,
- a text, ze zaklad bezi.

## 3) Ako citat vysledok

Vypis typu:
- Episode 1: total_reward=-8092.97

Negativne cislo je normalne.
Dolezite je hlavne to, ze:
- kod bezi,
- trening prebehne,
- mas funkcny DDPG zaklad.

## 4) Co je zjednodusene (naschval)

Tento skript je velmi jednoduchy, preto:
- iba 1 epizoda,
- iba 50 krokov,
- mala siet (64 neuronov),
- bez ukladania modelu,
- bez grafov.

Je to spravene tak, aby bol pochopitelny start.

## 5) Ako spustit

Vo WSL terminali:

1. prejdi do projektu
2. spusti python z venv

Priklad:
- cd /mnt/c/Users/HP/OneDrive/Desktop/ISI/cvicenie1/SmartBuilding/venv310
- ./Scripts/python.exe DDPGeasy.py

## 6) Co si zapamatat jednou vetou

Actor navrhuje akciu, Critic ju hodnoti, a cez opakovanie krokov sa obaja postupne zlepsuju.
