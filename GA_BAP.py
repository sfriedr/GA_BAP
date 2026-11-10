import pandas as pd
from pathlib import Path
import numpy as np
from numpy.random import Philox, Generator
from datetime import datetime
from numpy.typing import NDArray
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
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
pfad = basis_ordner / "data/Vorlage.csv" # Pfad zur CSV-Datei relativ zum Skript-Ordner
df = pd.read_csv(pfad, sep=";", parse_dates=['u', 'a', 'c']) # 'u', 'a', 'c' von String zu Pandas.Timestamp casten
df['a'] = df['a'].astype('datetime64[s]').astype('int64') # Wandelt Pandas.Timestamp (2023-01-02 22:42:20) in NumPy.int64 (1672617774)
df['u'] = df['u'].astype('datetime64[s]').astype('int64')
df['c'] = df['c'].astype('datetime64[s]').astype('int64')
#df['p'] = pd.to_timedelta(df['p']).dt.total_seconds().astype('int64')
df['p'] = pd.to_timedelta(df['p'].astype(str)).dt.total_seconds().astype('int64') # Wandelt die time-Objekte vektorisiert in Strings um und parest sie als Timedelta --- (00:23:27 -> <class 'datetime.time'>) zu ("00:23:27" -> String) zu (1407 -> <class 'numpy.int64'>)


row_count: int = len(df) # 25
print("row_count: ", row_count)
column_count: int = len(df.columns) # 11
print("column_count: ", column_count)
print(f"\nSpalten:\n{', '.join(df.columns)}") # name, typ, teu, u, v, a, s, p, w, yard_pos, c
print()

rng = np.random.default_rng(seed=42)
best_fitness_history: list[float] = []
worst_fitness_history: list[float] = []
#genes_of_best_individuals: list[tuple[float, float, float]] = []
best_individuals: list[NDArray[np.int64]] = []
reached_generation: int = 0


# Constraints
max_birthing_spots: int = 20
max_yard_pos = df["yard_pos"].max() # 16
a_min_limit: datetime = df['a'].min() # 1672617774 entspricht 023-02-01 00:02:54
#a_min_limit: np.int64 = np.datetime64(a_min).astype("datetime64[s]").astype(np.int64)

a_max_int64: np.int64 = df['a'].max()
a_max_2_timestamp: pd.Timestamp = pd.to_datetime(a_max_int64, unit='s') # np.int64 in pd.Timestamp konvertieren
next_day:pd.Timestamp = a_max_2_timestamp.normalize() + pd.Timedelta(days=1) # Uhrzeit auf 00:00:00 setzen und 1 Tag addieren
a_max_limit: np.int64 = np.int64(next_day.timestamp()) # pd.Timestamp in np.int64 konvertieren

number_vessels: int = len(df) # 25
number_birthing_spots: int = 20

v_min_limit: int = 1
v_max_limit: int = 20
a_min: np.int64 = np.int64(pd.to_datetime(df['a'].min(), unit='s').normalize().timestamp()) # Holt nur das reine Datum yyyyMMdd der frühesten Ankunft.
# Erstes Element von Spalte a holen (es ist in int64),
# zu datetime casten, um normalize ausfühen zu können,
# h, min, s abschneiden mit normalize,
# als int64 speichern.
# 1675209600 entspricht 2023-02-01 00:00:00
a_max: np.int64 = a_min + 86400 # 1675296000 entspricht 2023-02-02 00:00:00


# Strafen
PENALTY_CONFLICT: float = 10.0 # Strafe pro überlappenden Schiffspaar. Addiert 1.000.000.000 zur Fitness.
PENALTY_QUAY: float = 1e9 # Strafe pro Kaiplatz, der über den Kai hinausragt. Addiert 1.000.000.000 zur Fitness.


ALPHA: float = 0.5
BETA: float = 0.5
A: NDArray[np.int64] = df['a'].to_numpy(dtype=np.int64) # Ankunftszeiten aller Schiffe
P: NDArray[np.int64] = df['p'].to_numpy(dtype=np.int64)
W: NDArray[np.int64] = df['w'].to_numpy(dtype=np.int64)
S: NDArray[np.int64] = df['s'].to_numpy(dtype=np.int64)
YARD_POS: NDArray[np.int64] = df['yard_pos'].to_numpy(dtype=np.int64)
V_UPPER: NDArray[np.int64] = v_max_limit - S + 1
D_MAX: float = float(np.sum(np.maximum(YARD_POS - v_min_limit, V_UPPER - YARD_POS)))
T_MAX: float = float(np.sum(W * (a_max + P - A)))
OVERLAP_NORM: float = float((number_vessels - 1) * np.sum(P * S))


# Erzeugt Start-Population
# Population repräsentiert als (Populationsgröße, Anzahl_Entscheidungsvariablen, Anzahl_Schiffe), wobei die Entscheidungsvariablen u und v umfassen.
# Beispiel (4, 2, 25) -> Also Population bestehend aus vier Individuen, 2 Entscheidungsvariablen und 25 Schiffe.
# u_v1 -> Gen u von Vessel 1
# [ [[u_v1 u_v2 ... u_v25] [v_v1 v_v2 ... v_25]] [[u_v1 u_v2 ... u_v25] [v_v1 v_v2 ... v_25]] [[u_v1 u_v2 ... u_v25] [v_v1 v_v2 ... v_25]] [[u_v1 u_v2 ... u_v25] [v_v1 v_v2 ... v_25]]]]
def create_initial_population(population_size: int) -> NDArray[np.int64]:
    global rng
    global v_min_limit
    global a_min_limit
    global a_max_limit
    global a_max
    global number_vessels
    global number_birthing_spots
    global V_UPPER
    
    # Alle u-Werte für die gesamte Population aufeinmal erzeugen
    u_pop: NDArray[np.int64] = rng.integers(a_min_limit, a_max_limit + 1, size=(population_size, number_vessels))
    
    # Alle v-Werte für die gesamte Population aufeinmal erzeugen
    v_pop: NDArray[np.int64] = rng.integers(1, number_birthing_spots + 1, size=(population_size, number_vessels))
    
    # Beide 2D-Arrays zu einem 3D-Array (population_size, number_vessels) stapeln
    # axis=1 sorgt dafür, dass u (Spalte 0) und v (Spalte 1) pro Schiff nebeneinander liegen
    population: NDArray[np.int64] = np.stack([u_pop,v_pop], axis=1)

    # Schiffe dürfen nicht vor ihrer Ankunftszeit anlegen (u >= a) und sie dürfen nicht nach dem erlaubten Zeitraum (also 2.1.2023 00:00:00 - 3.1.2023 00:00:00) anlegen (u <= a_max)
    population[:, 0, :] = np.clip(population[:, 0, :], A, a_max)

    # schiffe müssen an die Kaimauer größentechnisch passen (v <= V_UPPER)
    population[:, 1, :] = np.clip(population[:, 1, :], v_min_limit, V_UPPER)

    return population



# Berechnet den Fitnesswert eines Individuums
# Individuum repräsentiert als (Anzahl_Entscheidungsvariablen, Anzahl_Schiffe)
# Beispiel (2, 25), Also Individuum bestehend aus 2 Entscheidungsvariablen und 25 Schiffe.
# Zielfunktion: \[ \sum_{i=1}^{n} w_i * (c_i - a_i) + \sum_{i=1}^{n} \lvert yardpos_i - v_i \rvert \]
def fitness_function(individual: NDArray[np.int64]) -> float:
    global ALPHA
    global BETA
    global A
    global P
    global W
    global S
    global YARD_POS
    global D_MAX
    global T_MAX

    U: NDArray[np.int64] = individual[0]
    V: NDArray[np.int64] = individual[1]
    C: NDArray[np.int64] = U + P
    v_end: NDArray[np.int64] = V + S

    time_term: float = float(np.sum(W * (C - A))) / T_MAX
    dist_term: float = float(np.sum(np.abs(YARD_POS - V))) / D_MAX
    objective: float = ALPHA * time_term + BETA * dist_term

    time_ov = np.minimum(C[:, None], C[None, :]) - np.maximum(U[:, None], U[None, :])
    space_ov = np.minimum(v_end[:, None], v_end[None, :]) - np.maximum(V[:, None], V[None, :])

    conflict: NDArray[np.bool_] = (time_ov > 0) & (space_ov > 0)
    conflict = np.triu(conflict, k=1)
    n_conflicts: int = int(conflict.sum())

    overlap_area: float = float(np.sum(np.where(conflict, time_ov * space_ov, 0))) / OVERLAP_NORM

    return objective + PENALTY_CONFLICT * n_conflicts + overlap_area




# Ermittlung der Elternpopulation
# Es wird eine Stichprobe von drei Fitnesswerten aus fittnesses entnommen.
# Davon wird der Index des besten Fitnesswertes gespeichert.
# Das Individuum der drei mit dem besten Fitnesswert wird in selected hinzugefügt.
# Der stärkste kann sich mehrmals durchsetzen.
# Die Populationsgröße von "selected" reduziert sich also nicht von y auf x, sondern ist gleich groß,
# bestehend aus den besten Individuen der Kandidatenpopulation "population".
def selection(population: NDArray[np.int64], fitnesses: list[float]) -> NDArray[np.int64]:
    global rng
    fit: NDArray[np.float64] = np.asarray(fitnesses) # shape (Populationsgröße,)
    selected: list[NDArray[np.int64]] = []
    for _ in range(len(population)):
        idx: NDArray[np.int64] = rng.choice(len(fit), size=3, replace=False) # shape (3,)
        winner: np.int64 = idx[np.argmin(fit[idx])]
        selected.append(population[winner].copy())
    return np.stack(selected)



# Parent repräsentiert als (Anzahl_Entscheidungsvariablen, Anzahl_Schiffe)
# Beispiel (2, 25), Also Parent bestehend aus 2 Entscheidungsvariablen und 25 Schiffe.
def crossover(parent1: NDArray[np.int64], parent2: NDArray[np.int64]) ->tuple[NDArray[np.int64], NDArray[np.int64]]:
    # Zufallsmaske: True -> nimm von parent1;   False -> nimm von parent2
    # Eine Liste mit 25 zufälligen Wahrheitswerten, die wie eine Schablone auf die Spalten gelegt werden.
    # Es werden "number_vessels" zufällige Zahlen zwischen 0 und 1 erzeugt.
    # Zahlen größer als 0.5 werden als True gewertet.
    #mask = np.random.rand(number_vessels) > 0.5
    global rng
    mask = rng.random(number_vessels) > 0.5

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
    global A
    global rng
    global v_min_limit
    global v_max_limit
    global a_min # 1675209600 entspricht 2023-02-01 00:00:00
    global a_max # 1675296000 entspricht 2023-02-02 00:00:00

    mutated: NDArray[np.int64] = child.copy()

    # u mutieren - Erlaubter bereich von 00:00:00 bis 00:00:00 des Folgetages - Das Datum (1.2.2023) bleibt immer gleich
    u_mask: NDArray[np.int64] = rng.random(mutated[0].shape) < mutation_rate
    if np.any(u_mask):
        mutated[0, u_mask] = rng.integers(a_min, a_max+1, size=int(np.sum(u_mask)), dtype='int64')

    # v mutieren - Erlaubter bereich von "v_min_limit" bis "v_max_limit+1"
    v_mask: NDArray[np.int64] = rng.random(mutated[1].shape) < mutation_rate
    if np.any(v_mask):
        mutated[1, v_mask] = rng.integers(low=v_min_limit, high=v_max_limit+1, size=int(np.sum(v_mask)), dtype='int64')

    mutated[0] = np.maximum(mutated[0], A) # u >= a -> zu kleine u-Werte auf Ankunftszeit anheben. Denn Schiffe können nicht vor ihrer Ankunftszeit am Hafen anlegen.

    return mutated


# Bestimmt das beste Individuum
# Individuum als Form von (Anzahl_Entscheidungsvariablen, Anzahl_Schiffe)
def genetic_algorithm(population_size: int, mutation_rate: float, target_fitness: float = 0.0, max_generation: int = 20000, stagnation_limit: int = 300) -> NDArray[np.int64]:
    current_fitness = float('inf')
    current_geneation: int = 0
    global reached_generation
    global best_fitness_history
    global worst_fitness_history
    global best_individuals

    if population_size % 2 != 0:
        raise ValueError("Die Populationgröße muss gerade sein, denn Eltern werden paarweise gekreuzt.")
    
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
        #print("fitnesses: ", fitnesses,"\n\n")

        # Fürs plotten ----------------------<
        best_fit_idx: int = int(np.argmin(fitnesses))
        best_fitness: float = fitnesses[best_fit_idx]
        best_fitness_history.append(best_fitness)

        worst_fit_idx: int = int(np.argmax(fitnesses))
        worst_fitness: float = fitnesses[worst_fit_idx]
        worst_fitness_history.append(worst_fitness)

        best_individuum: NDArray[np.int64] = population[best_fit_idx].copy()
        best_individuals.append(best_individuum.copy())
        # ----------------------------------->

        parents: NDArray[np.int64] = selection(population, fitnesses)

        next_population: list[NDArray[np.int64]] = []
        for i in range(0, population_size, 2): # Paarweises durchlaufen der Elternpopulation
            parent1: NDArray[np.int64] = parents[i] # Dimension (Individuen, Anzahl_Schiffe, Anzahl_Gene) (2, 25, 2)
            parent2: NDArray[np.int64] = parents[i+1]
            
            child1, child2 = crossover(parent1, parent2)

            # Die nächste Generation sollte eine Liste sein. Denn das mehrfach-Befüllen von NPArrays ist langsamer als bei Listen, weil die Inhalte immer kopiert werden müssen.
            next_population.append(mutation(child1, mutation_rate))
            next_population.append(mutation(child2, mutation_rate))

        next_population[0] = best_individuum.copy() # Elitismus - Das fiteste Individuum wird unverändert in die nächste Generation übernommen.
        population = np.stack(next_population) # Generationswechsel - Die neu erzeugte Generation wird zur Arbeitsgrundlage für den nächsten Schleifendurchlauf.

        current_fitness = best_fitness
        current_geneation += 1
    reached_generation = current_geneation
    print("Generation erreicht: Generation ", reached_generation)
    return best_individuum
    



def draw_individual(df_winner_indiv: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 8)) # Diagramm erstellen
    cmap = plt.colormaps['tab10'] # Farbpalette aus 10 Farben

    # idx = 0
    # row = id: 17, name: CMA CGM Marco Polo, typ: Panamax, teu: 4939, u: 2023-02-01 00:25:00, v: 10, a: 2023-02-01 00:02:00, s: 4, p: 0 days 00:23:27, w: 2, yard_pos: 16, c: 2023-02-01 00:48:00
    for idx, row in df_winner_indiv.iterrows():
        anlegezeitpunkt = mdates.date2num(row['u']) # x_1
        anlege_pos = row['v'] # y_1
        operationszeit = row['p'].total_seconds() / 86400.0 # Breite entlang x-Achse in Sekunden
        schiff_laenge = row['s'] # Höhe entlang y-Achse

        # Schiff als Rechteck zeichnen
        # patches.Rectangle((x, y), breite, hoehe)
        vessel = patches.Rectangle(
            (anlegezeitpunkt, anlege_pos),
            operationszeit,
            schiff_laenge,
            linewidth=1.5,
            edgecolor='black',
            facecolor='gray', #cmap(idx % 10), # Farben wiederholen sich nach jedem 10. Schiff.
            alpha=0.75,
            label=row['name']
        )
        ax.add_patch(vessel)

        # Beschriftung in das Rechteck setzen
        ax.text(
            anlegezeitpunkt,
            anlege_pos,
            f"{row['name']}\n(ID: {row['id']})",
            #ha='center',
            #va='center',
            fontsize=9,
            weight='bold',
            color='black'
        )


    zeit_nullpunkt = df_winner_indiv['u'].min()
    for i in range(len(df_winner_indiv)):
        for j in range(i + 1, len(df_winner_indiv)):  # j startet bei i+1, um Doppelnennungen zu vermeiden
            vessel_a = df_winner_indiv.iloc[i] # Enthält eine ganze Zeile.
            vessel_b = df_winner_indiv.iloc[j]
            
            vessel_a_x1 = mdates.date2num(vessel_a['u'])
            vessel_a_xLength = vessel_a['p'].total_seconds() / 86400.0
            vessel_a_x2 = vessel_a_x1 + vessel_a_xLength

            vessel_a_y1 = vessel_a['v']
            vessel_a_yLength = vessel_a['s']
            vessel_a_y2 = vessel_a_y1 + vessel_a_yLength

            vessel_b_x1 = mdates.date2num(vessel_b['u'])
            vessel_b_timeLength = vessel_b['p'].total_seconds() / 86400.0
            vessel_b_x2 = vessel_b_x1 + vessel_b_timeLength

            vessel_b_y1 = vessel_b['v']
            vessel_b_yLength = vessel_b['s']
            vessel_b_y2 = vessel_b_y1 + vessel_b_yLength


            overlap_x1 = max(vessel_a_x1, vessel_b_x1)
            overlap_x2 = min(vessel_a_x2, vessel_b_x2)
            overlap_y1 = max(vessel_a_y1, vessel_b_y1)
            overlap_y2 = min(vessel_a_y2, vessel_b_y2)


            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                conflict_rec_x1 = overlap_x1
                conflict_rec_y1 = overlap_y1
                conflict_rec_width = overlap_x2 - overlap_x1
                conflict_rec_height = overlap_y2 - overlap_y1

                # Zeichne das rote Konflikt-Rechteck
                conflict_rect = patches.Rectangle(
                    (conflict_rec_x1, conflict_rec_y1),
                    width=(conflict_rec_width), 
                    height=(conflict_rec_height), 
                    linewidth=0,            # Kein Rahmen
                    facecolor='red', 
                    alpha=0.5,              # Teiltransparent, damit man die Schiffe darunter noch sieht
                    hatch='//'              # Schraffierung für bessere Sichtbarkeit
                )
                ax.add_patch(conflict_rect)
                
                print(f"KONFLIKT GEFUNDEN: {vessel_a['name']} überschneidet sich mit {vessel_b['name']}!")


    # Diagramm-Begrenzungen:
    #max_zeit = (df['u'] + df['p']).max()
    #max_raum = (df['v'] + df['s']).max()

    max_zeit = (df_winner_indiv['c']).max()
    min_zeit = df_winner_indiv['a'].min()
    max_raum = (df_winner_indiv['v'] + df_winner_indiv['s']).max()
    
    ax.xaxis_date() # X-Achse Datums- und Uhrzeitwerte
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m. %H:%M')) # Anzeigeformat von Datum definieren
    #ax.xaxis.set_major_locator(mdates.HourLocator(interval=2)) # Intervall (alle 2h ein Tick)
    fig.autofmt_xdate() # Datum leicht schräg stellen.

    ax.set_xlim(min_zeit, max_zeit)
    #ax.set_ylim(0, max_raum)
    ax.set_yticks(np.arange(0, max_raum+1, 2))

    # Beschriftungen und Styling
    ax.set_title('Berth Allocation Problem (BAP) - Time-Space-Diagramm', fontsize=14, weight='bold', pad=15)
    ax.set_xlabel('Anlegezeitpunkt', fontsize=12)
    ax.set_ylabel('Kai-Position', fontsize=12)

    ax.grid(True, linestyle='--', alpha=0.5) # Raster hinzufügen

    # Layout optimieren und anzeigen
    plt.tight_layout()
    plt.show()



# Dataframe erstellen aus Daten vom Vorlage-Dataframe "df" und den Chromosomen des besten Indivuduums
def convert_best_indiv_to_df(individual: NDArray[np.int64]) -> pd.DataFrame:
    global df
    df_winner_indiv = pd.DataFrame()
    columns = ['id', 'name', 'typ', 'teu', 'a', 's', 'p', 'w', 'yard_pos']
    df_winner_indiv = df[columns].copy()
    df_winner_indiv['a'] = pd.to_datetime(df_winner_indiv['a'], unit='s')
    df_winner_indiv['p'] = pd.to_timedelta(df_winner_indiv['p'], unit='s')
    df_winner_indiv['u'] = pd.to_datetime(individual[0], unit='s')
    df_winner_indiv['v'] = individual[1]
    df_winner_indiv['c'] = df_winner_indiv['u'] + df_winner_indiv['p']

    #df_winner_indiv['a'] = df_winner_indiv['a'].astype('datetime64[s]').astype('int64')
    #df_winner_indiv['c'] = df_winner_indiv['c'].astype('datetime64[s]').astype('int64')
    return df_winner_indiv


# Zeichnet die Entwicklung der besten Fitnesswerte über die Generationen
def draw_best_fitness_evolvement():
    global reached_generation

    fig, ax = plt.subplots()
    max_fitness_values: list[float] = best_fitness_history
    min_fitness_values: list[float] = worst_fitness_history
    x_values = range(reached_generation)

    ax.plot(x_values, max_fitness_values, label='Best Fitness', color='black')
    ax.fill_between(x_values, min_fitness_values, max_fitness_values, color='gray', alpha=0.5, label='Fitness Range')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness Over Generations')
    ax.legend()
    plt.show()



def get_total_stay_time(df_winner_indiv: pd.DataFrame) -> pd.Timedelta:
    parameter_arr: NDArray[np.int64] = df_winner_indiv[['c', 'a']].values.astype('datetime64[s]').T.astype(np.int64) # Form (3, 25)
    c: NDArray[np.int64] = parameter_arr[0]
    a: NDArray[np.int64] = parameter_arr[1]
    total_stay_time: np.int64 = np.sum(c - a)
    total_stay_time: pd.Timedelta = pd.to_timedelta(total_stay_time, unit='s')
    return total_stay_time

def get_total_distance(df_winner_indiv: pd.DataFrame) -> np.int64:
    parameter_arr: np.ndarray = df_winner_indiv[['yard_pos', 'v']].values.T.astype(np.int64) # Form (2, 25)
    yard_pos: NDArray[np.int64] = parameter_arr[0]
    v: NDArray[np.int64] = parameter_arr[1]
    total_distance: np.int64 = np.sum(abs(yard_pos - v))
    return total_distance




best_individual: NDArray[np.int64] = genetic_algorithm(population_size=100, mutation_rate=0.2, max_generation=2000)
winner_fitness: float = best_fitness_history[len(best_fitness_history) - 1]
print("Fitnesswert von bestem Indiviuum: ", winner_fitness) # 620917.0
df_winner_indiv: pd.DataFrame = convert_best_indiv_to_df(best_individual)
total_stay_time: pd.Timedelta = get_total_stay_time(df_winner_indiv)
print("Gesamte Aufenthaltszeit aller Schiffe: ", total_stay_time)
total_distance: np.int64 = get_total_distance(df_winner_indiv)
print("Gesamte Distanz aller Schiffe zu den Yard-Positionen: ", total_distance, "m")
print()
# print("\nbest_fitness_history: ", best_fitness_history)
draw_individual(df_winner_indiv)
draw_best_fitness_evolvement()