# 04 – Szybkie 600: zasady i sprawdzenie na prawdziwych losowaniach

## Zasady w skrócie

- Losuje się **6 liczb z 32**, **co 4 minuty** (od ok. 6:36 do 23:52).
- Skreślasz 6 liczb. Stawka 1,60 zł + dopłata 25% = **2,00 zł za zakład**.
- Wygrane są stałe:

| Trafienia | Wygrana | Szansa |
|---|---|---|
| 6 z 6 | 31 200 zł (52 raty po 600 zł) | 1 na 906 192 |
| 5 z 6 | 250 zł | 1 na 5 809 |
| 4 z 6 | 50 zł | 1 na 186 |
| 3 z 6 | 4 zł | 1 na 17,4 |
| 2 z 6 | 2 zł (zwrot ceny kuponu) | 1 na 4,0 |

- Szansa na jakąkolwiek wygraną: **31,04%** (1 na 3,2). Szansa na wygraną większą niż 2 zł: 6,29%.
- Zwrot: **53,55%** z każdej złotówki, plus ok. 1,6 pkt z dodatkowej puli (2% stawek, doliczanej losowo z szansą 1 na 10).

Źródło: regulamin gry i „Informacja uzupełniająca" od 23.06.2025 (patrz [Źródła](07-zrodla.md)).

## Sprawdzenie na 10 000 prawdziwych losowaniach

Dane: 10 000 kolejnych losowań z okresu **19.08–26.09.2026** (przy losowaniu co 4 minuty to ok. 38 dni gry).

### 1. Czy losowania są losowe? Tak.

| Test | Wynik | Ocena |
|---|---|---|
| Czy któreś liczby padają częściej | każda liczba 1 798–1 993 razy (oczekiwane 1 875); p = 0,85 | równomiernie |
| Czy liczby z poprzedniego losowania powtarzają się częściej | średnio 1,119 wspólnej liczby (oczekiwane 1,125); p = 0,48 | brak zależności |
| Czy częstość z pierwszej połowy przewiduje drugą | korelacja −0,16 (brak związku) | nie przewiduje |
| „Gorące" liczby (6 najczęstszych z ostatnich 10 / 50 / 200 / 1000 losowań) | 1,11–1,13 trafienia (oczekiwane 1,125) | szum |

### 2. Czy teoria zgadza się z rzeczywistością? Tak.

Jeden stały kupon w każdym z 10 000 losowań (koszt 20 000 zł):

| Kupon | Szansa na wygraną | Zwrot | Strata |
|---|---|---|---|
| **Teoria** | **31,04%** | **53,55%** | −9 290 zł |
| 1, 2, 3, 4, 5, 6 | 31,07% | 55,00% | −9 000 zł |
| 27, 28, 29, 30, 31, 32 | 30,46% | 50,94% | −9 812 zł |
| 7, 13, 19, 21, 26, 31 | 31,01% | 49,27% | −10 146 zł |
| **20 000 losowych stałych kuponów — średnio** | **31,04%** | **53,64%** | – |

- Średnia z 20 000 kuponów zgadza się z teorią niemal co do setnej części procentu.
- 1,15% kuponów wyszło na plus — **każdy wyłącznie dzięki jednemu trafieniu 6 z 6**. Teoria przewiduje, że
  w 10 000 losowań trafia to ok. 1,1% kuponów — też się zgadza.
- „Ciąg" 1–6 czy liczby z końca zakresu nie są ani lepsze, ani gorsze od innych.

### 3. Pokrycie wszystkich 32 liczb

6 kuponów (5 rozłącznych + 1 z liczbami 31, 32 i czterema powtórzonymi), koszt 12 zł na losowanie:

| | Wynik |
|---|---|
| Jakakolwiek wygrana | **99,51%** losowań |
| Zwrot ≥ wydane 12 zł | tylko **3,06%** losowań |
| Zwrot | 51,09% |
| Strata za 10 000 losowań | −58 688 zł |

Tak jak w Multi Multi: pokrycie daje prawie zawsze „jakąś" wygraną, ale w sumie traci się ok. połowę.

### 4. Czy model AI typuje lepiej? Nie.

Uczciwy test: model trenowany od nowa co 500 losowań **tylko na losowaniach wcześniejszych**, co losowanie
typuje 6 najbardziej prawdopodobnych liczb (7 990 losowań):

| | Zwrot | Szansa na wygraną |
|---|---|---|
| Model AI | 57,08% | 31,10% |
| Stały kupon 1–6 | 54,04% | 31,04% |
| Teoria | 53,55% | 31,04% |

Różnica model − stały kupon: +0,06 zł na losowanie, **z = +0,71 — to szum**. Szansa na wygraną jest praktycznie
identyczna — model nie trafia częściej.

## Wniosek

Szybkie 600 zwraca średnio ok. 53,5% — wyraźnie więcej niż Multi Multi poza promocją (41%) — ale **nadal jest
to strata ok. 0,93 zł na każdym kuponie**. Losowania co 4 minuty sprawiają, że łatwo wydać dużo w krótkim czasie.
