import pandas as pd
from pathlib import Path
import numpy as np
from numpy.random import Philox, Generator
from datetime import datetime
from numpy.typing import NDArray
import math
import random
import sys
"""
u = Anlegezeitpunkt                                       (nicht gegeben)
v = Anlegeort                                             (nicht gegeben)
a = Ankunftszeit am Hafen                                 (gegeben)
s = Länge des Schiffes                                    (gegeben)
p = Operationszeit                                        (gegeben)
w = Priorität                                             (gegeben)
c = Fertigstellung des Schiffs zur Stunde c -> c = u + p  (nicht gegeben)
"""

basis_ordner = Path(__file__).parent # Ermittelt Ordner, in dem die Python-Datei liegt.
csv_pfad = basis_ordner / "Vorlage.xlsx" # Pfad zur CSV-Datei relativ zum Skript-Ordner
df = pd.read_excel(csv_pfad, index_col=0, parse_dates=['u', 'a', 'c']) # Erste Spalte als Index behandeln und 'u', 'a', 'c' in Timestamp umwandeln)
df['a'] = df['a'].astype('datetime64[s]').astype('int64') # Wandelt z.B. (2023-01-02 22:42:20 -> <class 'pandas._libs.tslibs.timestamps.Timestamp'>) in (1672617774 -> <class 'numpy.int64'>)
df['u'] = df['u'].astype('datetime64[s]').astype('int64')
df['c'] = df['c'].astype('datetime64[s]').astype('int64')
#df['p'] = pd.to_timedelta(df['p']).dt.total_seconds().astype('int64')
df['p'] = pd.to_timedelta(df['p'].astype(str)).dt.total_seconds().astype('int64') # Wandelt die time-Objekte vektorisiert in Strings um und parest sie als Timedelta --- (00:23:27 -> <class 'datetime.time'>) zu ("00:23:27" -> String) zu (1407 -> <class 'numpy.int64'>)

row_count: int = len(df) # 25
print("row_count: ", row_count)
column_count: int = len(df.columns) # 11
print("column_count: ", column_count)
print(f"\nSpalten:\n{', '.join(df.columns)}") # name, typ, teu, u, v, a, s, p, w, yard_pos, c


best_fitness_history: list[float] = []
worst_fitness_history: list[float] = []
#genes_of_best_individuals: list[tuple[float, float, float]] = []
best_individuals: list[NDArray[np.int64]] = []


# Constraints
max_birthing_spots: int = 20
max_yard_pos = df["yard_pos"].max() # 16
a_min_limit: datetime = df['a'].min() # 1672617774
#a_min_limit: np.int64 = np.datetime64(a_min).astype("datetime64[s]").astype(np.int64)

a_max_int64: np.int64 = df['a'].max()
a_max_2_timestamp: pd.Timestamp = pd.to_datetime(a_max_int64, unit='s') # np.int64 in pd.Timestamp konvertieren
next_day:pd.Timestamp = a_max_2_timestamp.normalize() + pd.Timedelta(days=1) # Uhrzeit auf 00:00:00 setzen und 1 Tag addieren
a_max_limit: np.int64 = np.int64(next_day.timestamp()) # pd.Timestamp in np.int64 konvertieren

number_vessels: int = len(df) # 25
number_birthing_spots: int = 20

v_min_limit: int = 1
v_max_limit: int = 20
u_yyMMdd: np.int64 = np.int64(pd.to_datetime(df['a'].iloc[0], unit='s').normalize().timestamp())
# Erstes Element von Spalte a holen (es ist in int64),
# zu datetime casten, um normalize ausfühen zu können,
# h, min, s abschneiden mit normalize,
# als int64 speichern.



# Erzeugt Start-Population
# Population repräsentiert als (Populationsgröße, Anzahl_Entscheidungsvariablen, Anzahl_Schiffe), wobei die Entscheidungsvariablen u und v umfassen.
# Beispiel (4, 2, 25) -> Also Population bestehend aus vier Individuen, 2 Entscheidungsvariablen und 25 Schiffe.
# u_v1 -> Gen u von Vessel 1
# [ [[u_v1 u_v2 ... u_v25] [v_v1 v_v2 ... v_25]] [[u_v1 u_v2 ... u_v25] [v_v1 v_v2 ... v_25]] [[u_v1 u_v2 ... u_v25] [v_v1 v_v2 ... v_25]] [[u_v1 u_v2 ... u_v25] [v_v1 v_v2 ... v_25]]]]
def create_initial_population(population_size: int) -> NDArray[np.int64]:
    global a_min_limit
    global a_max_limit
    global number_vessels
    global number_birthing_spots

    rng = np.random.default_rng(seed=42)
    
    # Alle u-Werte für die gesamte Population aufeinmal erzeugen
    u_pop: NDArray[np.int64] = rng.integers(a_min_limit, a_max_limit + 1, size=(population_size, number_vessels))
    
    # Alle v-Werte für die gesamte Population aufeinmal erzeugen
    v_pop: NDArray[np.int64] = rng.integers(1, number_birthing_spots + 1, size=(population_size, number_vessels))
    
    # Beide 2D-Arrays zu einem 3D-Array (population_size, number_vessels) stapeln
    # axis=1 sorgt dafür, dass u (Spalte 0) und v (Spalte 1) pro Schiff nebeneinander liegen
    population: NDArray[np.int64] = np.stack([u_pop,v_pop], axis=1)

    return population



# Berechnet den Fitnesswert eines Individuums
# Individuum repräsentiert als (Anzahl_Entscheidungsvariablen, Anzahl_Schiffe)
# Beispiel (2, 25), Also Individuum bestehend aus 2 Entscheidungsvariablen und 25 Schiffe.
# \[ \sum_{i=1}^{v} \left( w_i * (c_i - a_i) \right) + \sum_{i=1}^{v} \left( yardpos_i - v_i \right) \]
def fitness_function(individual: NDArray[np.int64]) -> float:
    global df

    u: NDArray[np.int64] = individual[0]
    v: NDArray[np.int64] = individual[1]

    a: NDArray[np.int64] = df['a'].to_numpy()
    p: NDArray[np.int64] = df['p'].to_numpy()
    w: NDArray[np.int64] = df['w'].to_numpy()
    yard_pos: NDArray[np.int64] = df['yard_pos'].to_numpy()
    c: NDArray[np.int64] = u + p

    sum_1: np.int64 = np.sum(w * (c - a))
    sum_2: np.int64 = np.sum(yard_pos - v)
    fitness: float = float(sum_1 + sum_2)
    return fitness


# Ermittlung der Elternpopulation
# Es wird eine Stichprobe von drei Fitnesswerten aus fittnesses entnommen.
# Davon wird der Index des besten Fitnesswertes gespeichert.
# Das Individuum der drei mit dem besten Fitnesswert wird in selected hinzugefügt.
# Der stärkste kann sich mehrmals durchsetzen.
# Die Populationsgröße von "selected" reduziert sich also nicht von y auf x, sondern ist gleich groß,
# bestehend aus den besten Individuen der Kandidatenpopulation "population".
def selection(population: NDArray[np.int64], fitnesses: list[float]) -> NDArray[np.int64]:
    rng3 = np.random.default_rng(56)
    selected: NDArray[np.int64] = np.array([], dtype=np.int64)
    for _ in range(len(population)):
        sample_indicies: NDArray[np.int64] = rng3.choice(len(fitnesses), size=3, replace=False)
        best_sample_fitness_idx: int = sample_indicies.min()
        selected = np.append(selected, population[best_sample_fitness_idx])
    return population



# Parent repräsentiert als (Anzahl_Entscheidungsvariablen, Anzahl_Schiffe)
# Beispiel (2, 25), Also Parent bestehend aus 2 Entscheidungsvariablen und 25 Schiffe.
def crossover(parent1: NDArray[np.int64], parent2: NDArray[np.int64]) ->tuple[NDArray[np.int64], NDArray[np.int64]]:
    # Zufallsmaske: True -> nimm von parent1;   False -> nimm von parent2
    # Eine Liste mit 25 zufälligen Wahrheitswerten, die wie eine Schablone auf die Spalten gelegt werden.
    # Es werden "number_vessels" zufällige Zahlen zwischen 0 und 1 erzeugt.
    # Zahlen größer als 0.5 werden als True gewertet.
    #mask = np.random.rand(number_vessels) > 0.5
    rng2 = np.random.default_rng(25)
    mask = rng2.random(number_vessels) > 0.5

    child1 = np.empty_like(parent1)
    child2 = np.empty_like(parent1)

    child1[:, mask] = parent1[:, mask]
    child1[:, ~mask] = parent2[:, ~mask]

    child2[:, mask] = parent2[:, mask]
    child2[:, ~mask] = parent1[:, ~mask]

    return child1, child2


# Mutieren von Kind
# Child repräsentiert als (Anzahl_Entscheidungsvariablen, Anzahl_Schiffe)
def mutation(child: NDArray[np.int64], mutation_rate: float):
    global v_min_limit
    global v_max_limit
    global u_yyMMdd

    rng4 = np.random.default_rng(47)
    mutated: NDArray[np.int64] = child.copy()

    # u mutieren - Erlaubter bereich von 00:00:01 bis 23:56:56 - Das Datum (2.1.2023) bleibt immer gleich
    u_mask: NDArray[np.int64] = rng4.random(mutated[0].shape) < mutation_rate
    if np.any(u_mask):
        random_seconds = rng4.integers(1, 86400, size=np.sum(u_mask), dtype='int64')
        mutated[0, u_mask] = u_yyMMdd + random_seconds

    # v mutieren - Erlaubter bereich von "v_min_limit" bis "v_max_limit+1"
    v_mask: NDArray[np.int64] = rng4.random(mutated[1].shape) < mutation_rate
    if np.any(v_mask):
        new_values = rng4.integers(low=v_min_limit, high=v_max_limit+1, size=np.sum(v_mask), dtype='int64')
        mutated[1, v_mask] = new_values

    return mutated


# Bestimmt das beste Individuum
# Individuum als Form von (Anzahl_Entscheidungsvariablen, Anzahl_Schiffe)
def genetic_algorithm(population_size: int, mutation_rate: float, target_fitness: float = 0.0, max_generation: int = 20000, stagnation_limit: int = 15) -> NDArray[np.int64]:
    current_fitness = float('inf')
    current_geneation: int = 0
    global best_fitness_history
    global worst_fitness_history
    global best_individuals
    
    population: NDArray[np.int64] = create_initial_population(population_size)

    while current_fitness != target_fitness:
        if current_geneation == max_generation:
            print("ABBRUCH - Maximale Anzahl an Generationen erreicht!\n")
            break
        # Prüft, ob die letzten `stagnation_limit` Werte alle nahe an `current_fitness` liegen
        elif len(best_fitness_history) >= stagnation_limit:
            recent_history = best_fitness_history[-stagnation_limit:]
            if all(math.isclose(f, current_fitness, abs_tol=1e-15) for f in recent_history):
                print("ABBRUCH - Fitnesswert hat Stagnationslimit erreicht!\n")
                break

        fitnesses: list[float] = [fitness_function(individuum) for individuum in population] # Liste von Fitnesswerten jedes Individuums.

        # Fürs plotten ----------------------<
        best_fit_idx: int = int(np.argmin(fitnesses))
        best_fitness: float = fitnesses[best_fit_idx]
        best_fitness_history.append(best_fitness)

        worst_fit_idx: int = int(np.argmax(fitnesses))
        worst_fitness: float = fitnesses[worst_fit_idx]
        worst_fitness_history.append(worst_fitness)

        best_individuum: NDArray[np.int64] = population[best_fit_idx]
        best_individuals.append(best_individuum.copy())
        # ----------------------------------->

        population = selection(population, fitnesses)

        next_population: NDArray[np.int64] = np.array([], dtype=np.int64)
        #next_population: list[NDArray[np.int64]] = []
        for i in range(0, population_size, 2): # Paarweises durchlaufen der Elternpopulation
            parent1: NDArray[np.int64] = population[i] # Dimension (Individuen, Anzahl_Schiffe, Anzahl_Gene) (2, 25, 2)
            parent2: NDArray[np.int64] = population[i+1]
            
            child1, child2 = crossover(parent1, parent2)

            # Die nächste Generation sollte eine Liste sein. Denn das mehrfach-Befüllen von NPArrays ist langsamer als bei Listen, weil die Inhalte immer kopiert werden müssen.
            next_population = np.append(next_population, mutation(child1.copy(), mutation_rate))
            next_population = np.append(next_population, mutation(child2.copy(), mutation_rate))

            next_population = np.append(next_population, best_individuum) # Elitismus - Das fiteste Individuum wird unverändert in die nächste Generation übernommen.
            next_population = next_population # Generationswechsel - Die neu erzeugte Generation wird zur Arbeitsgrundlage für den nächsten Schleifendurchlauf.

        current_fitness = best_fitness
        current_geneation += 1
    print("Generation erreicht: Generation ", current_geneation)
    return best_individuum
    



genetic_algorithm(population_size=4, mutation_rate=0.2)
winner_fitness: float = best_fitness_history[len(best_fitness_history) - 1]
print("Fitnesswert von bestem Indiviuum: ", winner_fitness) # 620917.0