# 05 – Testy losowości i model AI: co sprawdziliśmy i jak

## Dane

| Gra | Losowań | Okres |
|---|---|---|
| Multi Multi | 10 000 (bez luk) | 17.01.2013 – 26.09.2026 |
| Szybkie 600 | 10 000 (bez luk) | 19.08.2026 – 26.09.2026 |

Wyniki pobrane z oficjalnego API lotto.pl. W Multi Multi pole `resultsJson` zawiera 19 liczb,
a `specialResults` — 20. liczbę (Plus); razem 20 wylosowanych liczb.

## Testy losowości Multi Multi (53 testy)

Pytanie: czy w historii losowań jest jakikolwiek wzorzec, który pozwoliłby typować lepiej niż przypadek?

| Co sprawdzano | Wynik |
|---|---|
| Czy któreś liczby padają częściej | lekkie odchylenie (p = 0,019), które **znika po uwzględnieniu liczby wykonanych testów** i **niczego nie przewiduje**: częstości z pierwszej połowy danych nie mają związku z drugą (korelacja 0,005) |
| Zmiany w czasie (10 bloków po 1000 losowań) | brak (p = 0,88) |
| Liczba Plus | równomierna na 1–80 (p = 0,96) |
| Zależność od poprzednich losowań (1–20 losowań wstecz) | brak |
| „Gorące" i „zimne" liczby — 24 warianty strategii | trafialność w granicach ±0,002 od losowej 0,25 |
| „20 najczęstszych liczb" | 5,0055 trafienia wobec 5,00 losowo (p = 0,75) |
| Pary liczb padające razem | zgodne z przypadkiem |
| Suma, parzyste/nieparzyste, niskie/wysokie | zgodne z teorią |
| Pora losowania | bez znaczenia |

**Wynik: żaden z 53 testów nie wykazał wzorca.** Test był na tyle czuły, że wykryłby przewagę rzędu 1–1,5%.
Nawet gdyby drobne odchylenie w częstości liczb było prawdziwe i dałoby się wskazać „najlepsze" liczby
(a nie da się), zysk wyniósłby ok. 0,04 trafienia na kupon 10-liczbowy — nic wobec 59% przewagi organizatora.

Organizator ma certyfikat bezpieczeństwa **WLA-SCS:2024** (poziom 2, ważny do 2029), a każde losowanie odbywa
się zarejestrowanym urządzeniem pod nadzorem komisji (regulamin §12–13).

Testy losowości Szybkie 600 opisano w [04 – Szybkie 600](04-szybkie-600.md) — wynik taki sam.

## Model AI w tym projekcie — co robi

Model (sieć neuronowa) dostaje dla każdej liczby historię: jak często padała, jak dawno, czy pada seriami,
jak często w ostatnich 50 losowaniach. Na tej podstawie ocenia, jak prawdopodobne jest, że liczba padnie
w następnym losowaniu. Następnie liczby są sortowane od „najbardziej" do „najmniej" prawdopodobnej
i dzielone na kupony.

Model jest zbudowany uczciwie: do oceny losowania używa **wyłącznie wcześniejszych losowań** (nie „podgląda przyszłości").

## Jak uczciwie sprawdzić model

Najczęstszy błąd: sprawdzanie modelu na tych samych losowaniach, na których się uczył. Model „pamięta"
te dane, więc wynik wygląda lepiej niż w rzeczywistości.

Uczciwy test (tzw. *walk-forward*): model uczy się tylko na losowaniach **wcześniejszych** i typuje losowania,
**których nigdy nie widział**. Tak jak prawdziwy gracz.

## Wyniki modelu

### Multi Multi — średnia liczba trafień (20 typowanych liczb, 50 ostatnich losowań)

| | Średnio trafień |
|---|---|
| Model | 5,14 |
| Losowe typowanie | 5,00 |
| Najczęstsze liczby z historii | 4,64 |

p = 0,61 → **brak sygnału**. Model w ogóle „przyznaje", że liczby są jednakowo prawdopodobne:
przewiduje dla każdej ok. 25,2%, a faktycznie padały w 25,0% przypadków.

### Multi Multi — 8 kuponów po 10 liczb, pieniądze (7 990 losowań, uczciwy test)

| Strategia | Wygrane | Strata | Zwrot |
|---|---|---|---|
| Model (co losowanie nowe kupony z rankingu) | 69 724 zł | −90 076 zł | 43,6% |
| Stałe kupony 1–10, …, 71–80 | 74 668 zł | −85 132 zł | 46,7% |
| Teoria | – | −94 386 zł | 40,9% |

Różnica model − stałe kupony: **z = −0,28 → szum**. O tym, kto wypada „lepiej", decydują pojedyncze rzadkie
trafienia 9 z 10 (10 000 zł), a nie model.

Nawet sprawdzany na losowaniach, na których się uczył (wynik zawyżony), model dał zwrot 38,0% wobec 45,0% stałych kuponów.

### Szybkie 600 (7 990 losowań, uczciwy test)

Model: zwrot 57,1%, szansa na wygraną 31,10%; stały kupon 1–6: 54,0% i 31,04%. **z = +0,71 → szum.**

## Wniosek

Historia losowań nie zawiera informacji o przyszłych wynikach. Żadna analiza — ani prosta („gorące liczby"),
ani zaawansowana (sieć neuronowa) — nie daje przewagi. Wyniki modelu w tym projekcie traktuj jako ciekawostkę,
nie jako sposób na wygraną.

## Jak powtórzyć test modelu samodzielnie

```bash
uv run src/main.py update --game MultiMulti            # wczytanie danych i trening
uv run src/main.py evaluate --game MultiMulti --retrain --last-n 50   # uczciwy test (kilkanaście minut)
```

Wynik pokaże średnią liczbę trafień modelu, wynik losowy, wynik „najczęstszych liczb" i p-value.
