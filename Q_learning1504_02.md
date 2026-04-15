# Vysvetlenie suboru Q_learning1504_02.py

Tento dokument vysvetluje cely skript jednoduchsie a po castiach.

## 1. Co je ciel tohto skriptu

Cielom skriptu je naucit jednoducheho RL agenta riadit chladenie v jednej budove v prostredi CityLearn.

Agent sa snazi:

- znizit spotrebu elektriny zo siete,
- pripadne brat do uvahy aj komfort ludi v budove,
- a nasledne sa porovnat s jednoduchou fixnou strategiou.

V skripte sa pouziva vlastna implementacia tabulkoveho Q-learningu.

To znamena, ze:

- CityLearn poskytuje simulovane prostredie budovy,
- ale agent, logika ucenia a aktualizacia Q-hodnot su napisane rucne v tomto subore.

## 2. Zakladna myslienka Q-learningu

Q-learning je algoritmus posilnovaneho ucenia.

Zakladna myslienka je taka, ze agent sa postupne nauci:

- v akom stave sa nachadza,
- aku akciu moze vykonat,
- aku odmenu dostane po vykonani akcie,
- a ci sa oplati tuto akciu zopakovat aj nabuduce.

Q-learning si uklada znalost do tabulky, ktora sa vola Q-table.

Kazda hodnota v Q-table hovori priblizne toto:

"Ak som v stave S a spravim akciu A, aky dobry vysledok mozem ocakavat?"

Postupne sa teda agent uci, ktore akcie su v ktorych stavoch dobre a ktore zle.

## 3. Ako to funguje v nasom pripade

V tomto skripte:

- prostredie je budova v CityLearn,
- stav je zlozeny z pocasia, obsadenosti a tepelneho stavu,
- akcia je uroven chladenia,
- reward je zalozeny na spotrebe energie, pripadne aj na diskomforte.

Jedna iteracia vyzera takto:

1. Agent dostane aktualny stav budovy.
2. Vyberie akciu, teda aku uroven chladenia ma nastavit.
3. Prostredie vykona tuto akciu.
4. Prostredie vrati novy stav a reward.
5. Agent si upravi Q-table.

Takto sa to opakuje vela krat pocas celej epizody a potom pocas viacerych epizod.

## 4. Co znamena stav agenta

Stav je definovany cez premennu `ACTIVE_OBSERVATIONS`.

Pouzivaju sa tieto 4 vstupy:

### `hour`

Aktualna hodina dna.

Pomaha agentovi zistit, ci je rano, obed alebo vecer, a teda ci sa typicky oplati chladit viac alebo menej.

### `outdoor_dry_bulb_temperature_predicted_1`

Predpoved vonkajsej teploty o 1 krok dopredu.

To je dolezite, lebo agent sa neuci len z aktualneho pocasia, ale aj z blizkej predikcie. Ak vidi, ze pride teplejsie obdobie, moze zareagovat skor.

### `occupant_count`

Pocet ludi v budove.

To hovori agentovi, ci je budova obsadena. Ked v budove nikto nie je, moze si dovolit iny rezim chladenia ako ked je plna ludi.

### `indoor_dry_bulb_temperature_cooling_delta`

Rozdiel medzi vnutornou teplotou a chladiacim setpointom.

Jednoducho povedane hovori, ci je vnutri prilis teplo oproti cielovej hodnote. To je klucove, pretoze agent potrebuje vediet, ci treba chladenie zvysit alebo znizit.

## 5. Preco stav diskretizujeme

Q-learning v tejto verzii pracuje s tabulkou.

Problem je, ze realne hodnoty v prostredi su spojite:

- teplota moze byt 21.37,
- obsadenost 3 osoby,
- rozdiel teploty 1.18,
- akcia chladenia 0.624 a podobne.

Ak by sme chceli mat tabulku pre kazdu moznu realnu hodnotu, bola by nekonecne velka.

Preto ich rozdelime do intervalov, teda binov.

Priklad:

- hodina sa rozdeli na 24 binov,
- predikovana teplota na 6 binov,
- obsadenost na 4 biny,
- cooling delta na 8 binov.

Takto ziskame konecny pocet stavov a mozeme pouzit klasicku Q-table.

## 6. Funkcia `ObservationDiscretizer`

Tato trieda sluzi na prevod skutocnych hodnot stavu do diskretneho stavu.

Robi dve hlavne veci:

1. Pre kazdu premennu pripravi hranice intervalov.
2. Pri konkretnom pozorovani zisti, do ktoreho intervalu spada kazda hodnota.

Potom z tychto intervalov vytvori jeden jediny index stavu.

Priklad myslienky:

- hodina spada do binu 10,
- predikovana teplota do binu 4,
- obsadenost do binu 2,
- cooling delta do binu 5.

Z toho sa vytvori jeden index, napriklad stav cislo 1234.

Q-table potom nemusi pracovat s 4 hodnotami, ale len s jednym indexom stavu.

## 7. Funkcia `ActionDiscretizer`

Tato trieda robi to iste, ale pre akcie.

Akcia je v tomto skripte len jedna:

- `cooling_device`

CityLearn povodne povoluje spojite hodnoty akcie, ale my ich rozdelime na pevny pocet urovni.

Tu je to nastavene cez `ACTION_BIN_COUNT = 9`.

To znamena, ze agent si nemoze vybrat lubovolnu realnu hodnotu chladenia, ale len jednu z 9 preddefinovanych urovni medzi minimom a maximom povolenym prostredim.

To je prakticke, lebo tabulkovy Q-learning potrebuje konecny pocet akcii.

## 8. Trieda `OwnTabularQLearning`

Toto je hlavna cast celeho algoritmu. Tu je napisany vlastny agent.

### Co obsahuje

Agent si drzi:

- `q_table` - tabulku s odhadom kvality akcii v jednotlivych stavoch,
- `epsilon` - pravdepodobnost nahodneho skusania akcii,
- `learning_rate` - ako rychlo sa uci,
- `discount_factor` - ako velmi berie do uvahy buduci zisk.

### Ako vybera akciu

Toto robi metoda `predict()`.

Najprv diskretizuje aktualny stav.

Potom rozhoduje medzi dvoma moznostami:

1. **Exploracia**
Agent skusi nahodnu akciu.

2. **Exploatacia**
Agent vyberie akciu, ktora ma v Q-table najlepsiu hodnotu.

Na zaciatku je `epsilon` vysoke, teda agent viac skusa.
Neskor sa `epsilon` znizuje a agent viac vyuziva to, co sa naucil.

### Ako sa uci

Toto robi metoda `update()`.

Pouziva standardny Q-learning update:

Q(s, a) = Q(s, a) + alpha * (reward + gamma * max Q(s', a') - Q(s, a))

Vyklad:

- `Q(s, a)` je aktualny odhad hodnoty akcie,
- `reward` je okamzita odmena,
- `max Q(s', a')` je najlepsi odhad v dalsom stave,
- `alpha` je rychlost ucenia,
- `gamma` je vaha buducich odmien.

Ak vysledok akcie dopadol dobre, hodnota v Q-table sa zvysi.
Ak dopadol zle, hodnota sa znizi.

### Ako sa meni epsilon

Toto robi metoda `finish_episode()`.

Po kazdej epizode sa `epsilon` zmensi.

To znamena:

- na zaciatku agent vela experimentuje,
- neskor viac vyuziva naucenu strategiu.

## 9. Reward funkcie

V skripte su dve reward funkcie.

### `EnergyOnlyReward`

Tato reward funkcia tresta len spotrebu elektriny zo siete.

Cim vacsia spotreba zo siete, tym horsi reward.

Toto nuti agenta hladat strategiu, ktora znizi dovoz energie zo siete.

### `OccupancyAwareReward`

Tato reward funkcia berie do uvahy:

- spotrebu zo siete,
- a zaroven aj prehrievanie, ked je budova obsadena.

To je realistickejsie, pretoze nechceme len setrit elektrinu, ale aj udrzat prijemnu teplotu, ked su v miestnosti ludia.

## 10. Fixna strategia

Trieda `FixedCoolingPolicy` predstavuje velmi jednoduchu referencnu strategiu.

Neuci sa.
Nevyhodnocuje stav.
Nereaguje na pocasie ani na obsadenost.

Vzdy vrati rovnaku hodnotu akcie chladenia.

To je dolezite, lebo potrebujeme nejaky jednoduchy referencny bod, s ktorym porovname RL agenta.

Inak povedane:

- fixna strategia = stale rovnake chladenie,
- RL agent = meni chladenie podla stavu a ucenia.

## 11. Funkcia `make_env()`

Tato funkcia vytvori prostredie CityLearn.

Nastavi:

- aky dataset sa ma pouzit,
- ktoru budovu budeme riadit,
- ake pozorovania budu aktivne,
- aku akciu budeme riadit,
- aku reward funkciu pouzijeme.

To znamena, ze tato funkcia pripravi cely "simulator sveta", v ktorom sa agent pohybuje.

## 12. Funkcia `train_q_learning()`

Tato funkcia robi samotny trening agenta.

Pre kazdu epizodu:

1. resetne prostredie,
2. resetne agenta,
3. dokola vybera akcie, dostava reward a uci sa,
4. po skonceni epizody ulozi celkovy reward,
5. vypise priebezny progres.

Vysledkom je `TrainingTrace`, kde su ulozene:

- rewardy po epizodach,
- hodnoty epsilon,
- cas treningu,
- odhad, v ktorej epizode sa ucenie stabilizovalo.

## 13. Funkcia `run_policy()`

Tato funkcia uz nic neuci.

Len vezme politiku a necha ju prejst cele prostredie.

Pouziva sa na vyhodnotenie po treningu.

Okrem sumarnej statistiky uklada aj priebeh v case:

- aku akciu agent volil,
- aky bol reward,
- aka bola obsadenost,
- aka bola teplota,
- aka bola grid spotreba.

To sa potom pouziva na grafy.

## 14. Funkcia `estimate_stability_episode()`

Tato funkcia sa pokusa odhadnut, kedy sa krivka ucenia zacala stabilizovat.

Porovnava dve po sebe iduce okna rewardov.

Ak je rozdiel medzi nimi maly a zaroven sa reward velmi nerozkmita, funkcia povie, ze agent uz sa sprava stabilnejsie.

Nie je to matematicke optimum, ale je to prakticky odhad stabilizacie.

## 15. Funkcia `save_learning_curve_plot()`

Tato funkcia kresli graf ucenia.

Na grafe vidis:

- reward v kazdej epizode,
- klzavy priemer rewardu,
- epsilon,
- pripadny odhad epizody stabilizacie.

Tento graf je velmi uzitocny, lebo ukazuje, ci sa agent zlepsuje alebo nie.

## 16. Funkcia `save_strategy_comparison_plot()`

Tato funkcia kresli porovnanie medzi fixnou strategiou a Q-learning agentom.

Na grafoch vidis:

- ake akcie chladenia voli fixna a RL strategia,
- aka je spotreba zo siete,
- ake je prehrievanie,
- a nepriamo aj suvis s obsadenostou.

To pomaha vizualne ukazat, ze agent sa nesprava stale rovnako, ale prisposobuje sa situacii.

## 17. Funkcia `run_experiment()`

Toto je hlavna riadiaca funkcia experimentu.

Robi tieto kroky:

1. Spusti fixnu strategiu.
2. Ulozi jej vysledky.
3. Pre kazdu reward funkciu vytvori nove prostredie.
4. Natrenuje vlastneho Q-learning agenta.
5. Vyhodnoti jeho spravanie.
6. Ulozi CSV subory a grafy.
7. Vypocita percenta uspor oproti fixnej strategii.

Teda tato funkcia riadi cely experiment od zaciatku az po finalne subory.

## 18. Funkcia `parse_reward_names()`

Tato funkcia prevedie textove nazvy reward konfiguracii z argumentov na skutocne triedy.

Priklad:

- `energy_only` -> `EnergyOnlyReward`
- `occupancy_aware` -> `OccupancyAwareReward`

## 19. Funkcia `main()`

Toto je vstupny bod programu.

Spravi toto:

1. Nacita argumenty z prikazoveho riadku.
2. Vypise, aky dataset, budovu a nastavenia ides pouzit.
3. Zavola `run_experiment()`.
4. Vypise sumar vysledkov.
5. Vypise, kde sa ulozili subory.

## 20. Ako sa agent vlastne uci v tomto konkretnom skripte

V skratke:

1. Dostane stav budovy.
2. Stav sa diskretizuje.
3. Agent si zvoli jednu z 9 moznych urovni chladenia.
4. Prostredie vykona krok simulacie.
5. Dostane reward.
6. Podla rewardu upravi Q-table.
7. Po mnohych krokoch a epizodach si vybuduje tabulku, ktora hovori, ake akcie sa oplatia v danych stavoch.

Takto sa teda uci na zaklade:

- aktualnej hodiny,
- predikcie vonkajsej teploty,
- poctu ludi v budove,
- aktualneho teplotneho rozdielu vnutri,
- a odmeny za svoje rozhodnutia.

## 21. Na zaklade coho sa rozhoduje, ci bola akcia dobra alebo zla

To urcuje reward.

Ak po akcii klesne spotreba zo siete, reward je lepsi.

Ak je zapnuta `OccupancyAwareReward`, tak navyse:

- ked je budova obsadena,
- a zaroven sa prehrieva,
- agent dostane vacsi trest.

Takze dobra akcia je taka, ktora:

- znizi grid spotrebu,
- a idealne nezhorsi komfort ludi.

## 22. Co je dolezite pochopit k vysledkom

Ak RL agent vyjde lepsie ako fixna strategia, znamena to, ze sa naucil pouzivat rozne urovne chladenia rozumnejsie nez konstanta hodnota.

To vsak neznamena, ze nasiel dokonale optimum.

Znamena to skor:

- v ramci zvoleneho stavoveho popisu,
- v ramci zvolenej diskretizacie,
- v ramci zvoleneho rewardu,
- a v ramci poctu epizod,

nasiel lepsie pravidla rozhodovania nez jednoducha fixna politika.

## 23. Jednoducha intuitivna predstava

Predstav si, ze agent je clovek, ktory sa kazdy den uci ovladat klimatizaciu.

Pozera sa na:

- kolko je hodin,
- ake bude pocasie,
- kolko ludi je vnutri,
- ci je v miestnosti prilis teplo.

Najprv skusa rozne moznosti naslepo.
Potom si zapamata, co fungovalo dobre.
Po case uz vie, ze v niektorych situaciach sa oplati chladit viac a v inych menej.

To je presne to, co robi tvoj Q-learning agent.

## 24. Co si z toho mas zapamatat najviac

Najdolezitejsie su tieto body:

1. Agent je vlastny, nie prevzaty z kniznice.
2. Uci sa pomocou Q-table.
3. Stav je tvorený casom, predikciou pocasia, obsadenostou a teplotnym rozdielom.
4. Akcia je uroven chladenia.
5. Reward hovori, ci bolo rozhodnutie dobre alebo zle.
6. Fixna strategia je len referencny bod, ktory sa neuci.
7. Grafy ukazuju, ci sa agent zlepsuje a ako sa sprava oproti fixnej politike.

## 25. Co by si mohol povedat na obhajobe

Ak sa ta budu pytat, mozes povedat nieco v tomto style:

"V praci som pouzil CityLearn ako simulacne prostredie budovy, ale samotny tabulkovy Q-learning agent som implementoval vlastnorucne. Agent sa uci na zaklade diskretizovaneho stavu, ktory obsahuje hodinu, predikovanu vonkajsiu teplotu, obsadenost budovy a rozdiel medzi vnutornou teplotou a chladiacim setpointom. Akciou je uroven chladenia. Ucenie prebieha pomocou aktualizacie Q-table na zaklade reward funkcie, ktora zohladnuje spotrebu zo siete a volitelne aj komfort pocas obsadenosti. Vysledky porovnavam s fixnou strategiou a zobrazuje ich aj krivka ucenia a casove grafy spravania oboch politik." 
