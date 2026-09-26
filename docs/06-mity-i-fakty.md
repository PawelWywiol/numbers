# 06 – Mity i fakty

| Mit | Fakt | Na podstawie |
|---|---|---|
| „Ta liczba dawno nie padła, więc zaraz musi" | Losowanie nie ma pamięci. Liczby „zimne" padają tak samo często jak inne. | 24 warianty strategii na 10 000 losowań Multi Multi: trafialność ±0,002 od przypadku |
| „Gorące liczby padają częściej" | Nie. Częstości z przeszłości nie przewidują przyszłości. | Korelacja pierwszej i drugiej połowy danych: 0,005 (MM), −0,16 (Szybkie 600) |
| „Trzeba grać ciągle tymi samymi liczbami, bo w końcu padną" | Stałe liczby mają dokładnie te same szanse co zmieniane. Nic się nie „kumuluje". | Stałe kupony vs kupony zmieniane co losowanie: ta sama strata |
| „Pokrycie wszystkich liczb gwarantuje wygraną" | Gwarantuje co najwyżej drobne trafienia, często bez wypłaty. 16×5 w Multi Multi: gwarantowane 0 zł, strata 23,54 zł na losowanie. | Kombinatoryka + 10 000 losowań |
| „Systemy / koła zwiększają szanse" | Zmieniają tylko rozkład wyników (częstsze małe wygrane), nie średnią. Każdy kupon ma swój stały zwrot. | Dokładny rachunek na wielu układach kuponów: identyczna średnia |
| „Sztuczna inteligencja przewidzi liczby" | Model z tego projektu w uczciwym teście wypadł jak przypadek (p = 0,61; z = −0,28; z = +0,71). | [Testy i model AI](05-testy-i-model.md) |
| „Ciąg 1, 2, 3, 4, 5, 6 nie może paść" | Ma dokładnie tę samą szansę co każda inna szóstka. | W Szybkie 600 kupon 1–6 wypadł jak średnia |
| „Większy kupon = większa szansa na zysk" | W Multi Multi zwrot jest prawie identyczny dla 3–10 liczb (ok. 41%). | Oficjalna tabela wygranych |
| „Opcja Plus się opłaca" | Podnosi zwrot o ok. 1 pkt, ale podwaja koszt — strata na kuponie rośnie dwukrotnie. | Oficjalna tabela wygranych |
| „Wystarczy grać dłużej, żeby się odegrać" | Im dłużej grasz, tym pewniej wynik zbliża się do średniej, czyli do straty. | Prawo wielkich liczb; 10 000 losowań |
| „Ktoś wygrał tym systemem, więc działa" | Pojedyncze wygrane to wariancja. 7 na 1000 stałych zestawów wyszło na plus — każdy dzięki jednemu rzadkiemu trafieniu. | Symulacja 1000 układów na prawdziwych danych |

## Jedyna udokumentowana naukowo „przewaga" — i gdzie działa

**Wybieranie niepopularnych kombinacji w grach z pulą dzieloną** (Lotto: wygrane za 4, 5 i 6 trafień;
Mini Lotto; Eurojackpot).

- Nie zwiększa szansy trafienia — ta jest dla każdej kombinacji taka sama.
- Zmniejsza liczbę osób, z którymi dzielisz pulę, jeśli trafisz.
- Gracze masowo wybierają daty urodzin (1–31), „szczęśliwe" liczby (7, 11) i wzory na kuponie (przekątne, linie).
  W Niemczech w 1999 r. padły liczby 2, 3, 4, 5, 6, 26 — nagrodę za 5 trafień wygrało 38 008 osób i każda
  dostała **20–40 razy mniej niż zwykle**.
- **Nie działa** w grach ze stałymi wygranymi (Multi Multi, Keno, Szybkie 600, Ekstra Pensja) ani dla stałej
  wygranej za 3 trafienia w Lotto (35 zł).
- Nawet z tą przewagą gra nadal jest na minusie. Brak badań o tym, jakie liczby wybierają polscy gracze.

Źródła: Wang i in. 2016, Hauser-Rethaller i König 2002, Baker i McHale 2009 i inne (patrz [Źródła](07-zrodla.md)).

## Ile postawić? Matematyka odpowiada: nic

Kryterium Kelly'ego — naukowa metoda wyznaczania optymalnej stawki — mówi, że przy grze ze stratą
średnią (a taką jest każda gra liczbowa) **optymalna stawka to 0 zł**. Jeśli grasz dla rozrywki,
ustal z góry budżet, którego stratę akceptujesz.
