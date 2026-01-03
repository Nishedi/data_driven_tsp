# Ekstrakcja Cech z Instancji TSP

## Przegląd

Model sieci neuronowej nie analizuje bezpośrednio plików TSP. Zamiast tego, ekstraktuje **7 cech numerycznych** z każdej instancji problemu TSP, które opisują charakterystykę problemu. Te cechy są następnie używane jako dane wejściowe do sieci neuronowej, która przewiduje optymalne parametry algorytmu Simulated Annealing.

## Proces Ekstrakcji Cech

### 1. Wczytanie Danych

Gdy plik TSP jest wczytywany (np. z formatu TSPLIB), współrzędne miast są przekształcane w obiekt `TSPInstance`:
- Współrzędne miast: tablica NumPy o wymiarach `(n_cities, 2)` 
- Macierz odległości: obliczana automatycznie dla wszystkich par miast

### 2. Funkcja `get_features()` (plik: `tsp_solver.py`, linie 37-64)

Model wywołuje metodę `get_features()`, która oblicza **7 cech** opisujących instancję TSP:

```python
def get_features(self) -> np.ndarray:
    distances = self.distance_matrix[np.triu_indices(self.n_cities, k=1)]
    
    features = [
        self.n_cities,                                      # Cecha 1
        np.mean(distances),                                 # Cecha 2
        np.std(distances),                                  # Cecha 3
        np.max(distances) - np.min(distances),             # Cecha 4
        np.max(distances) / (np.min(distances) + 1e-8),    # Cecha 5
        np.std(self.cities[:, 0]),                         # Cecha 6
        np.std(self.cities[:, 1]),                         # Cecha 7
    ]
    
    return np.array(features, dtype=np.float32)
```

## Szczegółowy Opis Cech

### **Cecha 1: Liczba Miast** (`self.n_cities`)
- **Co to jest**: Całkowita liczba miast w instancji TSP
- **Dlaczego ważna**: Większe problemy zazwyczaj wymagają:
  - Wyższej temperatury początkowej
  - Więcej iteracji na poziom temperatury
  - Wolniejszego chłodzenia (wyższy współczynnik chłodzenia)
- **Przykład**: Dla problemu z 50 miastami: `cecha_1 = 50`

### **Cecha 2: Średnia Odległość** (`np.mean(distances)`)
- **Co to jest**: Średnia wszystkich odległości między parami miast
- **Jak obliczana**: Suma wszystkich odległości / liczba par miast
- **Dlaczego ważna**: Wskazuje na skalę problemu:
  - Większa średnia odległość → większa przestrzeń poszukiwań
  - Wymaga wyższej temperatury początkowej dla właściwej eksploracji
- **Przykład**: Jeśli miasta są rozmieszczone w przestrzeni 100×100, średnia odległość może wynosić ~50

### **Cecha 3: Odchylenie Standardowe Odległości** (`np.std(distances)`)
- **Co to jest**: Miara rozproszenia odległości między miastami
- **Dlaczego ważna**: Wskazuje na zróżnicowanie odległości:
  - Wysokie odchylenie standardowe → duża różnorodność odległości
  - Niskie odchylenie → odległości są podobne
  - Wpływa na strategię eksploracji w algorytmie SA
- **Przykład**: Odchylenie = 20 oznacza, że większość odległości jest w zakresie średnia ± 20

### **Cecha 4: Zakres Odległości** (`max - min`)
- **Co to jest**: Różnica między najdłuższą a najkrótszą odległością
- **Dlaczego ważna**: Pokazuje ekstremalne wartości w problemie:
  - Duży zakres → niektóre miasta są bardzo blisko, inne bardzo daleko
  - Wpływa na akceptację gorszych rozwiązań w SA
- **Przykład**: Jeśli min = 5, max = 120, to zakres = 115

### **Cecha 5: Stosunek Odległości** (`max / min`)
- **Co to jest**: Iloraz najdłuższej i najkrótszej odległości
- **Dlaczego ważna**: Względna miara skali problemu:
  - Wysoki stosunek (np. 50) → bardzo duże zróżnicowanie
  - Niski stosunek (np. 2-3) → odległości są proporcjonalne
  - Pomaga modelowi zrozumieć strukturę geometryczną
- **Przykład**: max=100, min=5 → stosunek = 20

### **Cecha 6: Odchylenie Standardowe Współrzędnych X** (`np.std(cities[:, 0])`)
- **Co to jest**: Rozproszenie miast wzdłuż osi X
- **Dlaczego ważna**: Wskazuje na rozkład przestrzenny:
  - Wysokie odchylenie → miasta rozłożone szeroko w poziomie
  - Niskie odchylenie → miasta skupione w wąskim pasie pionowym
  - Pomaga zrozumieć kształt problemu
- **Przykład**: Miasta w kwadracie 100×100 będą miały odchylenie ~29

### **Cecha 7: Odchylenie Standardowe Współrzędnych Y** (`np.std(cities[:, 1])`)
- **Co to jest**: Rozproszenie miast wzdłuż osi Y
- **Dlaczego ważna**: Podobnie jak cecha 6, ale dla wymiaru pionowego:
  - Razem z cechą 6 określa "kształt" problemu
  - Porównanie cech 6 i 7 pokazuje, czy problem jest bardziej "wydłużony" czy "okrągły"
- **Przykład**: Jeśli cecha_6 = 40, cecha_7 = 10 → problem jest wydłużony poziomo

## Architektura Sieci Neuronowej

### Dane Wejściowe
```
Wektor cech: [7 wartości zmiennoprzecinkowych]
```

### Struktura Sieci (plik: `neural_network.py`)
```
Warstwa 1:  Linear(7 → 64)   + ReLU + LayerNorm + Dropout(0.2)
Warstwa 2:  Linear(64 → 64)  + ReLU + LayerNorm + Dropout(0.2)
Warstwa 3:  Linear(64 → 32)  + ReLU + LayerNorm
Warstwa 4:  Linear(32 → 4)   [parametry wyjściowe]
```

### Parametry Wyjściowe

Sieć przewiduje **4 parametry** dla algorytmu Simulated Annealing:

1. **Temperatura Początkowa** (Initial Temperature)
   - Zakres: `[10, ~110]`
   - Aktywacja: `Softplus(x) * 100 + 10`
   - Wyższa dla większych/bardziej złożonych problemów

2. **Współczynnik Chłodzenia** (Cooling Rate) ⚠️ **POPRAWIONO**
   - Zakres: `[0.95, 0.999]` (po poprawce)
   - Aktywacja: `Sigmoid(x) * 0.049 + 0.95`
   - Wyższe wartości (bliżej 0.999) = wolniejsze chłodzenie = więcej iteracji

3. **Temperatura Minimalna** (Minimum Temperature)
   - Zakres: `[0, ~0.1]`
   - Aktywacja: `Softplus(x) * 0.1`
   - Określa moment zakończenia algorytmu

4. **Iteracje na Temperaturę** (Iterations per Temperature)
   - Zakres: `[50, ~250]`
   - Aktywacja: `Softplus(x) * 200 + 50`
   - Więcej iteracji dla większych problemów

## Normalizacja Cech

Przed przekazaniem do sieci neuronowej, cechy są **normalizowane**:

```python
features_normalized = (features - mean) / (std + 1e-8)
```

Gdzie `mean` i `std` są obliczane z danych treningowych. Ta normalizacja:
- Zapewnia, że wszystkie cechy mają podobną skalę
- Przyspiesza uczenie sieci neuronowej
- Poprawia stabilność numeryczną

## Przykład Przepływu Danych

```
Plik TSP (np. berlin20.tsp)
         ↓
   Parser TSPLIB
         ↓
Współrzędne miast: [(x1,y1), (x2,y2), ..., (x20,y20)]
         ↓
   TSPInstance
         ↓
Macierz odległości: [20×20 matrix]
         ↓
   get_features()
         ↓
Wektor cech: [20, 45.2, 18.3, 67.5, 8.9, 28.4, 29.1]
         ↓
   Normalizacja
         ↓
Znormalizowany wektor: [-0.5, 0.2, -0.1, 0.8, 1.2, -0.3, -0.2]
         ↓
   Sieć Neuronowa
         ↓
Parametry SA: [temp=95.2, cooling=0.973, min_temp=0.02, iters=180]
         ↓
Solver Simulated Annealing
         ↓
   Rozwiązanie TSP
```

## Dlaczego Te Cechy?

Te 7 cech zostało wybranych, ponieważ:

1. **Są niezależne od układu współrzędnych** - nie zmieniają się przy przesunięciu czy obrocie
2. **Opisują skalę problemu** - cechy 1, 2 wskazują wielkość
3. **Opisują złożoność** - cechy 3, 4, 5 wskazują trudność
4. **Opisują geometrię** - cechy 6, 7 wskazują kształt rozkładu
5. **Są łatwe do obliczenia** - nie wymagają złożonych algorytmów
6. **Są informatywne** - korelują z optymalnymi parametrami SA

## Wpływ Poprawki na Model

Po wprowadzeniu poprawki współczynnika chłodzenia:
- Model **nadal ekstraktuje te same 7 cech**
- **Zmienił się tylko zakres wyjściowy** współczynnika chłodzenia: `[0.9, 1.0]` → `[0.95, 0.999]`
- Po **ponownym wytrenowaniu**, model nauczy się przewidywać wyższe wartości współczynnika chłodzenia
- To pozwoli na **wolniejsze chłodzenie** i **lepszą eksplorację** przestrzeni rozwiązań
