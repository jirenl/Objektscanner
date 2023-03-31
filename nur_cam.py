#! /usr/bin/env python
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#Main-Skript für die Bild-Klassifikation

import drivers
import argparse         #System-relevante Imports
import sys

from time import sleep
import time


import cv2
from tflite_support.task import core
from tflite_support.task import processor       #Imports für die Objekterkennung
from tflite_support.task import vision

# Fenster für die Ergebnisvisualisierung
_ROW_SIZE = 20  # pixels
_LEFT_MARGIN = 24  # pixels
_PADDING = 10
_TEXT_COLOR = (172, 246, 200)  # mint
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_FPS_AVERAGE_FRAME_COUNT = 10

def run(model: str, max_results: int, score_threshold: float, num_threads: int,
        enable_edgetpu: bool, camera_id: int, width: int, height: int) -> None:         #deklariert argumente ohne inhalt (wird im unteren teil gefüllt)

 #   Argumentenbeschreibeung: ###########
 #     model: Name des TFLite-Models.
 #     max_results: Maximale Anzahl von Bilderkennungsergebnisse.
 #     score_threshold: The score threshold of classification results.
 #     num_threads: Anzahl der CPU Threads für das betreiben der Modelle.
 #     enable_edgetpu: Whether to run the model on EdgeTPU.
 #     camera_id: Die Kamera-ID wird an OpenCV gegeben.
 #     width: Die Breite für das Fenster der Kamera.
 #     height: Die Höhe für das Fenster der Kamera.
 ##############################

  # Initialisiert die Bild-Klassifikation
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)

  #"Aktiviert" Die Objekt-Erkennung
  classification_options = processor.ClassificationOptions(
      max_results=max_results, score_threshold=score_threshold)
  options = vision.ImageClassifierOptions(
      base_options=base_options, classification_options=classification_options)

  classifier = vision.ImageClassifier.create_from_options(options)

  # Benötigte Variablen für die FPS-Anzeige (Bilder pro Sekunde)
  counter, fps = 0, 0
  start_time = time.time()

  # Startet die Videoaufnahmeingabe von der Kamera 
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
  print("Es geht los mit dem Scan! ")
  # Eine While-Schleife soll solange neue Bilder machen bis das Programm mit "ESC" geschlossen wird
  while cap.isOpened():
    #sleep(1)
    success, image = cap.read()
    if not success:
      sys.exit(
          'FEHLER: Das Lesen der Kamera ist nicht moeglich. Bitte verifizieren Sie die Kamera-Einstellungen.'
      )

    counter += 1        #zählt die Programmabläufe mit
    image = cv2.flip(image, 1)  #spiegelt die Kamera, damit die Ausgabe wie eine "Spiegel" scheint

    # Konvertiert das Bild von BRG zu RGB, weil TFLite dies für die Verarbeitung benötigt.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Erstellt ein TensorImage aus dem RGB BIld
    tensor_image = vision.TensorImage.create_from_array(rgb_image)
    # Listet alle möglichen Klassifikationsergebnisse ein
    categories = classifier.classify(tensor_image)
    
    # Zeigt alle möglichen Klassifikationsergebnisse auf der Anzeige im Fenster
    for idx, category in enumerate(categories.classifications[0].categories):
      category_name = category.category_name
      score = round(category.score, 2)
      result_text = category_name + ' (' + str(round(score*100,2)) + '%)'  #gibt den Objektnamen mit Prozentwahrscheinlichkeit aus
      text_location = (_LEFT_MARGIN, (idx + 2) * _ROW_SIZE)
      cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                  _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)


    # Errechnet die wahrscheinliche FPS
    if counter % _FPS_AVERAGE_FRAME_COUNT == 0:
      end_time = time.time()
      fps = _FPS_AVERAGE_FRAME_COUNT / (end_time - start_time)
      start_time = time.time()

    # zeigt die FPS an
    fps_text = 'FPS = ' + str(int(fps))
    text_location = (_LEFT_MARGIN, _ROW_SIZE)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

    # Stoppt das Programm wenn der Benutzer "ESC" gedrückt hat
    if cv2.waitKey(1) == 27:  #27 ist für den ACII-Wert ESCAPE
      break
    cv2.imshow('image_classification', image)

  cap.release()
  cv2.destroyAllWindows()

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Name of image classification model.',
      required=False,
      default='efficientnet_lite0.tflite')
  parser.add_argument(
      '--maxResults',
      help='Max of classification results.',
      required=False,
      default=3)
  parser.add_argument(
      '--scoreThreshold',
      help='The score threshold of classification results.',
      required=False,
      type=float,
      default=0.0)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      default=480)
  args = parser.parse_args()

  run(args.model, int(args.maxResults),
      args.scoreThreshold, int(args.numThreads), bool(args.enableEdgeTPU),              #siehe oben: setzt die Argumenten mit den initialisierten Argumenten ein 
      int(args.cameraId), args.frameWidth, args.frameHeight)
    
    
if __name__ == '__main__':      #die Hauptfunktion wird erst ausgeführt wenn die main nicht verändert wurde von Imports.
  main()

