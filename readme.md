# Vergleich von Audio Klassifikations-Verfahren
## Vorhaben
Im Laufe der Ausarbeitung werden verschiedene Methoden der Audioklassifikation angewandt und die jeweiligen Ergebnisse
miteinander verglichen. Als Klassifikationsproblem wird dabei Verkehrszählung genutzt, also das Erkennen von Fahrzeugen
innerhalb eines Audiosignals. Die erforderlichen Daten werden im Rahmen der Arbeit erhoben und automatisch gelabelt
(mehr dazu im Abschnitt *Umsetzung*). Die Bachelorarbeit stellt dabei den gesamten Prozess des Vorhabens, ausgehend von 
der Datenerhebung bis hin zum Training des Modells, vor und diskutiert die Ergebnisse. Falls möglich, werden die 
Ergebnisse ebenfalls in verkürzter Form als Paper veröffentlicht.

## Umsetzung
In der Datenerhebung werden Videos von Straßen mit vorbeifahrenden Fahrzeugen aufgenommen und weiter verarbeitet. Über 
das Video wird mithilfe von Computer Vision bestimmt, wie viele Autos tatsächlich vorbeigefahren sind. Aus der Audiodatei 
des Videos werden Features generiert, mit denen dann die verschiedenen Ansätzen der Klassifikation weiterverarbeitet. 
Zum maschinellen Lernen wird Tensorflow2 in Kombination mit Keras verwendet, im Bereich Computer Vision kommt Open-CV 
zum Einsatz. Das genaue Labeling der Daten sowie die weiteren Schritte lassen sich dem Code entnehmen.

### YOLO-Objekterkennung
Im Rahmen des Labelings, also der Verarbeitung der Videos, wird zur Erkennung der Fahrzeuge die Darknet Implementierung
des YOLO-Frameworks (YOLOv4) mit OpenCV benutzt. Diese wurde bereits auf dem MS COCO-Datensatz trainiert und kann über 
80 verschiedene Kategorien bzw. Gegenstände erkennen. Dabei kann der Nutzer entscheiden, on das standardmäßige Netz 
(genauere Erkennung, aber langsamer) oder das kleinere Netz (schneller, aber etwas ungenauer) verwendet werden soll. 
Die Gewichte sowie Konfigurationsdateien können über die untenstehenden Links heruntergeladen werden. Außerdem werden
die Klassenbezeichnungen des COCO-Datensatzes benötigt, um die erkannten Objekte filtern und beschriften zu können. Alle
heruntergeladenen Dateien sollten innerhalb des _ComputerVision_ Pakets im Ordner _YOLO_ gespeichert werden. Alternativ 
können in der Klasse _ObjectDetection.py_ die Pfade manuell angepasst werden.

**YOLOv4**

Weights-File: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

Config-File: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg

**YOLOv4-Tiny**

Weights-File: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

Config-File: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg

## Sonstiges
Die extrahierten Audiodateien sowie die zugehörigen Label werden nach Abschluss der Arbeit auf Kaggle zur
Verfügung gestellt. Die Videodaten können aus datenschutzrechtlichen Gründen nicht verfügbar gemacht werden. 

**Kaggle-Datensatz:** Link folgt... 

# Comparing audio classification approaches
## Project
In the course of the research different methods of audio classification are applied and the respective results
compared to each other. Traffic counting, i.e. the recognition of vehicles within an audio signal, is used as a 
classification problem. The required data is collected as part of the work and automatically labeled (more on this in 
the *Implementation* section). The bachelor thesis represents the entire process of the project, starting from data 
collection to model training, and discusses the results. If possible, results also will be published in an 
abbreviated form as a paper.

## Implementation
In the data collection, videos of roads with passing vehicles are recorded and further processed. Through the video, 
computer vision is used to determine how many cars actually passed by. Features are generated from the audio file of 
the video, which is then used to train the neural network. For machine learning, Tensorflow2 is used in combination 
with Keras, and Open-CV is used for computer vision. The exact labeling of the data as well as the further steps can 
be seen in the code.

### YOLO-Object Detection
In the context of labeling, i.e. the processing of the videos, the darknet implementation of the YOLO framework (YOLOv4) 
with OpenCV is used to recognize the vehicles. This has already been trained on the MS COCO dataset and can recognize 
over 80 different categories or objects. The user can decide whether to use the default model (more accurate recognition, 
but slower) or the smaller model (faster, but slightly less accurate). The weights as well as configuration files can be 
downloaded from the links below. In addition, the class labels of the COCO dataset are needed to filter and label the 
recognized objects. All downloaded files should be stored inside the _ComputerVision_ package in the _YOLO_ folder. 
Alternatively, the paths can be adjusted manually in the _ObjectDetection.py_ class.

**YOLOv4**

Weights-File: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

Config-File: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg

**YOLOv4-Tiny**

Weights-File: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

Config-File: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg

## Other
The extracted audio files as well as the corresponding label files will be made available on Kaggle after completion of 
the project. The video data cannot be made available for privacy reasons. 

**Kaggle dataset:** Link follows...