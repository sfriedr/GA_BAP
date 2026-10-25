# Evolutionärer Algorithmus (Noch nicht fertig - wird noch entwickelt!)
Python-Anwendung, welche die Meta-Heuristik eines evolutionären Algorithmus implementiert zur Lösung des Berth Allocation Problems (BAP). Es wird eine XLSX-Datei impotiert, welche als Vorlage für den Liegeplatzplan dient. Der Algo bestimmt die optimalen Anlegezeiten und Anlegepositionen aller Schiffe, die an einem Tag in den Hafen einlaufen. Dabei wird der finale Liegeplatzplan als XLSX-Datei erstellt, sowie diverse Diagramme zur Visualisierung der Entwicklung des Plans.

## Terminologie der Meta-Heuristik nach diesem Anwendungsfall
* Gen: Z.B. die Anliegezeit von Schiff_1 oder der Anlegeort von Schiff_23 als numpy.int64
* Chromosom: Gesamtheit aller Gene eines Schiffes
* Individuum: Der vollst. Liegeplatzplan als numpy.typing.NDArray[numpy.int64] der Form (Anzahl_Entscheidungsvariablen, Anzahl_Schiffe)
* Population: Ist die Gesamtheit aller Individuen als numpy.typing.NDArray[numpy.int64] der Form (Populationsgröße, Anzahl_Entscheidungsvariablen, Anzahl_Schiffe)
* Phänotyp: Der geplottete Liegeplatzplan als Weg-Zeit-Diagramm.

## Abküurzungsverzeichnis
* u = Anlegezeitpunkt
* v = Anlegeort
* a = Ankunftszeit am Hafen
* s = Länge des Schiffes
* p = Operationszeit
* w = Priorität
* c = Fertigstellung des Schiffs zur Stunde c -> c = u + p  (nicht gegeben)

## Abbruchbedingungen
* Erreichen eines Zielfitnesswertes
* Erreichen einer fest definierten Anzahl an Generationen.
* Stagnation -> Fitnesswert verbessert sich über bestimmte Anzahl an Generationen nicht.

## Pseudocode
* Aktuelle Population = Ausgangspopulation
* Bewerte Indivividuen in Ausgangspopulation
* Wiederhole (Erzeuge neue Generation)
    * Selektion zur Reproduktion aus aktueller Population
    * Replikation
    * Variation
    * Bewerte Individuen der neuen Population
    * Selektion für Überleben und Bilden neuer aktueller Population
Bis Abbruchkriterium erreicht

## Fitnessfunktion
Der Fitnesswert ist die Wartezeit bzw. Aufenthaltszeit unter Berücksichtigung der Gewichtung einzelner Schiffe und minimaler Distanz der Container zum Yard.
\[ \sum_{i=1}^{v} \left( w_i * (c_i - a_i) \right) + \sum_{i=1}^{v} \left( yardpos_i - v_i \right) \]


## Ausgaben
* Liegeplatzplan als Time-Space-Diagramm
* Entwicklung der Gene des besten Individuums über Generationen
* Entwicklung des jeweils besten Fitnesswertes pro Generation