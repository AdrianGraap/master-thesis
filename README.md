# Masterarbeit von Adrian Graap

Dieses Repository enthält die Dateien, die zur Durchführung der Experimente in der Masterarbeit verwendet wurden.

Im Ordner `testing` findet man Dateien, die für die Reproduktionsstudie verwendet wurde.

Im Ordner `my_relation_detection` befinden sich die Dateien für die eigene Implementierung einer Relation Prediction

Um die Dateien und Skripte selbst auszuführen, werden zwei vorgefertigte Conda-Umgebungen bereitgestellt. Diese können 
mit `conda env create -f master-thesis.yml` oder `conda env create -f tf.yml` erzeugt werden. 

Für die Skripte in `testing` wird die Umgebung `master-thesis` benötigt.

Die Skripte in `my_relation_detection` benötigen entweder die Umgebung `master-thesis` oder `tf`, weitere Informationen
findet man in dem Ordner selbst.
