# Tarea-1 Deteccion y Emparejamiento de Rasgos

Programa en Python utilizando opencv para detección y emparejamiento de rasgos en imágenes.

El programa pregunta al usuario si desea realizar la operación de detección o emparejamiento de rasgos.

Para la detección de rasgos, el usuario podrá escoger entre 10 algoritmos de detección: Good Features to Track, FAST, BRIEF, ORB, AGAST, AKAZE, BRISK, KAZE, SIFT y SURF.

Para el emparejamiento de rasgos tenemos 2 métodos: Brute Force (Fuerza Bruta) o FLANN. Para cada uno deestos, podremos escoger entre 6 algoritmos de detección: 
SIFT, SURF, KAZE, BRIEF, BRISK, ORB y AKAZE.

Aunque el codigo incluye SURF para detección, este algoritmo está patentado y excluido de la configuración de opencv al momento de escribir el programa por lo que si se utiliza, 
el programa simplemente nos despliega un mensaje de advertencia y nos regresa al menú.

El programa puede sintetizarse ya que varias lineas de codigo se repiten.

Otra posible mejora sería dar la posibilidad al usuario de ajustar ciertos parámetros en los procesos para obtener mejores resultados.

Bibliografía:

Good Features to Track:
https://docs.opencv.org/master/d4/d8c/tutorial_py_shi_tomasi.html

FAST:
https://docs.opencv.org/master/df/d0c/tutorial_py_fast.html

BRIEF:
https://docs.opencv.org/master/dc/d7d/tutorial_py_brief.html

ORB:
https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html

AGAST:
https://docs.opencv.org/3.4/d7/d19/classcv_1_1AgastFeatureDetector.html

AKAZE:
https://docs.opencv.org/3.4/d8/d30/classcv_1_1AKAZE.html

BRISK:
https://docs.opencv.org/3.4/de/dbf/classcv_1_1BRISK.html

KAZE:
https://docs.opencv.org/master/d3/d61/classcv_1_1KAZE.html

SIFT:
https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html

SURF:
https://docs.opencv.org/master/df/dd2/tutorial_py_surf_intro.html

Emparejamiento (Brute Force y FLANN):
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

Videos:

Learn OPENCV in 3 Hours with Python:
https://www.youtube.com/watch?v=WQeoO7MI0Bs&t=476s

Feature Detection and Matching + Image Classifier Project:
https://www.youtube.com/watch?v=nnH55-zD38I&t=1502s
