# 01 – Podstawy: pojęcia wyjaśnione po ludzku

## Szansa „1 na N"

„Szansa 1 na 4" oznacza, że **średnio** co czwarty kupon coś wygra. Nie znaczy to, że wygrasz dokładnie
co czwarty raz — możesz przegrać 10 razy z rzędu albo wygrać 3 razy z rzędu. Dopiero przy bardzo wielu
kuponach wynik zbliża się do 1 na 4.

Przykład: w Multi Multi, skreślając 4 liczby, szansa na jakąkolwiek wygraną to 25,89%, czyli ok. **1 na 3,9**.

## Zwrot (RTP) — najważniejsza liczba w każdej grze

**Zwrot** (ang. *Return To Player*, RTP) mówi, ile **średnio** wraca do gracza z każdej wydanej złotówki,
gdy zagra się bardzo wiele razy.

- Zwrot 41% = z każdych 100 zł wydanych na kupony wraca średnio 41 zł, a **59 zł to strata**.
- Zwrot 57% = z każdych 100 zł wraca średnio 57 zł, strata 43 zł.
- Zwrot powyżej 100% oznaczałby zysk. **Żadna gra liczbowa w Polsce go nie ma** (sprawdziliśmy wszystkie, patrz [Porównanie gier](02-porownanie-gier.md)).

Zwrot liczy się tak: dla każdej możliwej wygranej mnoży się jej kwotę przez szansę jej trafienia,
sumuje i dzieli przez cenę kuponu. To czysta arytmetyka na oficjalnej tabeli wygranych — nic tu nie jest szacowane.

## Wartość oczekiwana

To samo co zwrot, ale w złotówkach: ile średnio wraca z jednego kuponu.
Kupon Multi Multi z 5 liczbami kosztuje 2,50 zł, a jego wartość oczekiwana to **1,03 zł**.
Każdy taki kupon to więc średnio **1,47 zł straty**.

Ważna zasada: **wartość oczekiwana kilku kuponów to po prostu suma wartości oczekiwanych każdego z nich** —
niezależnie od tego, czy kupony mają wspólne liczby, czy nie. Dlatego żaden układ kuponów
(„system", „pokrycie", „koło") nie zmienia średniego wyniku. Szczegóły: [Multi Multi](03-multi-multi.md).

## Obowiązkowa dopłata 25% — kupon kosztuje więcej, niż mówi „stawka"

Regulaminy wszystkich gier liczbowych Totalizatora nakładają **obowiązkową dopłatę 25% do stawki**.
Wygrane liczone są od stawki, ale płacisz stawkę + dopłatę:

| Gra | Stawka | Dopłata 25% | **Płacisz** |
|---|---|---|---|
| Multi Multi | 2,00 zł | 0,50 zł | **2,50 zł** |
| Keno | 1,60 zł | 0,40 zł | **2,00 zł** |
| Szybkie 600 | 1,60 zł | 0,40 zł | **2,00 zł** |
| Mini Lotto | 1,60 zł | 0,40 zł | **2,00 zł** |
| Lotto (od 18.08.2026) | 4,00 zł | 1,00 zł | **5,00 zł** |
| Ekstra Pensja | 4,00 zł | 1,00 zł | **5,00 zł** |
| Eurojackpot | 10,00 zł | 2,50 zł | **12,50 zł** |

Wszystkie zwroty w tej dokumentacji liczone są od **całej kwoty, którą płacisz** (stawka + dopłata).

## Gry ze stałymi wygranymi i gry z pulą

- **Stałe wygrane** (Multi Multi, Keno, Szybkie 600, Ekstra Pensja, Lotto Plus): za dane trafienie zawsze
  dostajesz tę samą kwotę z tabeli. Nieważne, ile osób wygrało. Tu **wybór liczb nie ma żadnego znaczenia**.
- **Pula dzielona** (wyższe wygrane w Lotto, Mini Lotto, Eurojackpot): pula pieniędzy dzielona jest między
  wszystkich zwycięzców. Szansa trafienia jest dla każdej kombinacji taka sama, ale jeśli wiele osób
  skreśliło to samo co ty, **dostaniesz mniej**. Tu jedyną udokumentowaną naukowo „przewagą" jest wybieranie
  kombinacji, których inni rzadko wybierają (patrz [Mity i fakty](06-mity-i-fakty.md)).

## Losowanie nie ma pamięci

Maszyna losująca (lub certyfikowany generator) nie „wie", jakie liczby padły wcześniej.
Każde losowanie zaczyna się od zera. Dlatego:

- liczba, która nie padła od 50 losowań, **nie jest „bardziej należna"**,
- liczba, która padała ostatnio często, **nie jest „gorąca"**,
- granie ciągle tymi samymi liczbami **nie zwiększa ani nie zmniejsza szans**.

Sprawdziliśmy to na prawdziwych wynikach — szczegóły w [Testach](05-testy-i-model.md).

## Wariancja — dlaczego ktoś wygrywa, choć średnio się traci

Średni wynik to jedno, a pojedyncze wyniki mocno się wahają. W 10 000 losowań Multi Multi
1000 różnych stałych zestawów kuponów dało zwrot od 34% do 166% — ale **tylko 7 na 1000 wyszło na plus
i każdy wyłącznie dzięki jednej, rzadkiej, bardzo wysokiej wygranej**. Pojedyncze szczęśliwe wyniki
nie oznaczają, że metoda działa — dlatego zawsze patrzymy na średnią z bardzo wielu prób.

## Test statystyczny i „p-value" w jednym akapicie

Gdy porównujemy np. model AI z losowymi kuponami, pytamy: „czy ta różnica mogła wyjść przypadkiem?".
**p-value** to szansa, że czysty przypadek dałby taki lub większy wynik. Umownie dopiero **p poniżej 0,05**
(5%) traktuje się jako sygnał. Podobnie **z** (tzw. wynik standaryzowany): różnica ma znaczenie dopiero,
gdy z jest większe niż ok. 2 lub mniejsze niż ok. −2. Wszystkie nasze testy modelu dały wartości daleko
od tych progów — różnice to szum.
