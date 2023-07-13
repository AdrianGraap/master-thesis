# my_relation_detection

In diesem Ordner befinden sich die Dateien für die durchgeführten Experimente.

Die Experimente wurden mit zwei verschiedenen Ansätzen durchgeführt.

Diese jeweiligen Experimente der beiden Ansätze findet man in den Ordnern `DBpedia/transformers_approach` und `DBpedia/tensorflow_approach`.

Für die Experimente aus `DBpedia/transformers_approach` wird die Umgebung `master-thesis` benötigt.

Für die Experimente aus `DBpedia/tensorflow_approach` wird die Umgebung `tf` benötigt.

Die Umgebungen werden benötigt, wenn man die Skripte direkt ausführen möchte.

In den jeweiligen Ordnern sind zusätzlich noch eine `Dockerfile` und eine `requirements.txt` enthalten, wenn man die
Experimente in einem Docker-Container starten möchte. 

Für die Analyse der Experimente befindet sich in jedem Ordner ein weiterer Ordner `analysis`, welcher dafür sorgt, dass 
die darin enthaltene Analyse auf jedes Experiment durchgeführt wird.

Für die Fehleranalyse stehen die Skripte in `DBpedia/error_analysis` bereit.