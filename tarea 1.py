import cv2
import numpy as np
from matplotlib import pyplot as plt

def mainMenu():
    print("1.Deteccion de rasgos")
    print("2.Emparejamiento de rasgos")
    print("3.Salir")
    seleccion=int(input("Seleccionar una de las opciones:"))
    if seleccion==1:
        Deteccion()
    elif seleccion==2:
        Emparejamiento()
    elif seleccion==3:
        exit()
    else:
        print("Error, seleccionar una opcion de 1-3")
        mainMenu()

# 1.Deteccion de rasgos
def Deteccion():
    print("1.Good Features to Track")
    print("2.FAST")
    print("3.BRIEF")
    print("4.ORB")
    print("5.AGAST")
    print("6.AKAZE")
    print("7.BRISK")
    print("8.KAZE")
    print("9.SIFT")
    print("10.SURF")
    print("11.Regresar a pantalla de inicio")
    metodoDet=int(input("Seleccionar un metodo de deteccion:"))

    #Metodo Good Features to track
    if metodoDet==1:
        img = cv2.imread(input("Nombre de la imagen de entrada:"))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        corners = np.int0(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img, (x, y), 3, 255, -1)
        plt.imshow(img), plt.show()

    #Metodo FAST
    elif metodoDet==2:
        img = cv2.imread(input("Nombre de la imagen de entrada:"))
        # Initiate FAST object with default values
        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(img, None)
        img2 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
        cv2.imshow('Con supresion de no maximos', img2)
        cv2.waitKey(0)
        # Disable nonmaxSuppression
        fast.setNonmaxSuppression(0)
        kp = fast.detect(img, None)
        img3 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
        cv2.imshow('Sin supresion de no maximos', img3)
        cv2.waitKey(0)

    #Metodo BRIEF
    elif metodoDet==3:
        img = cv2.imread(input("Nombre de la imagen de entrada:"))
        star = cv2.xfeatures2d.StarDetector_create()
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        kp = star.detect(img, None)
        kp, des = brief.compute(img, kp)
        img2=cv2.drawKeypoints(img,kp,None,color=(255,0,0))
        cv2.imshow("BRIEF",img2)
        cv2.waitKey(0)
        print(brief.descriptorSize())
        print(des.shape)

    #Metodo ORB
    elif metodoDet==4:
        img = cv2.imread(input("Nombre de la imagen de entrada:"))
        orb = cv2.ORB_create()
        kp = orb.detect(img, None)
        kp, des = orb.compute(img, kp)
        img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        plt.imshow(img2), plt.show()

    #Metodo AGAST
    elif metodoDet==5:
        img = cv2.imread(input("Nombre de la imagen de entrada:"))
        agast = cv2.AgastFeatureDetector_create()
        kp = agast.detect(img, None)
        img2 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
        cv2.imshow('Con supresion de no maximos', img2)
        cv2.waitKey(0)
        agast.setNonmaxSuppression(0)
        kp = agast.detect(img, None)
        img3 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
        cv2.imshow('Sin supresion de no maximos', img3)
        cv2.waitKey(0)

    #Metodo AKAZE
    elif metodoDet==6:
        img = cv2.imread(input("Nombre de la imagen de entrada:"))
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = cv2.AKAZE_create()
        kp, des = detector.detectAndCompute(imgGray, None)
        img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        cv2.imshow("AKAZE", img2)
        cv2.waitKey(0)

    #Metodo BRISK
    elif metodoDet==7:
        img = cv2.imread(input("Nombre de la imagen de entrada:"))
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = cv2.BRISK_create()
        kp, des = detector.detectAndCompute(imgGray, None)
        img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        cv2.imshow("AKAZE", img2)
        cv2.waitKey(0)

    # Metodo KAZE
    elif metodoDet==8:
        img = cv2.imread(input("Nombre de la imagen de entrada:"))
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = cv2.KAZE_create()
        kp, des = detector.detectAndCompute(imgGray, None)
        img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        cv2.imshow("KAZE", img2)
        cv2.waitKey(0)

    # Metodo SIFT
    elif metodoDet==9:
        img = cv2.imread(input("Nombre de la imagen de entrada:"))
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = cv2.SIFT_create()
        kp = detector.detect(imgGray, None)
        img2 = cv2.drawKeypoints(imgGray, kp, img)
        cv2.imshow("SIFT",img2)
        cv2.waitKey(0)

    # Metodo SURF
    elif metodoDet==10:
        print("Este algoritmo esta patentado y esta excluido de esta configuracion")
        #img = cv2.imread(input("Nombre de la imagen de entrada:"))
        #imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #detector = cv2.xfeatures2d.SURF_create(400)
        #kp, des = detector.detectAndCompute(img, None)
        #img2 = cv2.drawKeypoints(imgGray, kp, img)
        #cv2.imshow("SURF",img2)
        #cv2.waitKey(0)
        Deteccion()

    # Regresar a pantalla de inicio
    elif metodoDet==11:
        mainMenu()

    #Error
    else:
        print("Error, seleccionar una opcion de 1-10")
        Deteccion()

# 2.Emparejamiento de rasgos
def Emparejamiento():
    print("1.Fuerza Bruta")
    print("2.FLANN")
    print("3.Regresar a panatalla de inicio")
    metodoEmp=int(input("Seleccionar un metodo de emparejamiento:"))

    #Metodo Fuerza Bruta
    if metodoEmp==1:
        print("1.SIFT")
        print("2.SURF")
        print("3.KAZE")
        print("4.BRIEF")
        print("5.BRISK")
        print("6.ORB")
        print("7.AKAZE")
        metodoDetEmp = int(input("Seleccionar el metodo de deteccion:"))

        # SIFT
        if metodoDetEmp == 1:
            img1 = cv2.imread(input("Nombre de la imagen 1:"),0)
            img2 = cv2.imread(input("Nombre de la imagen 2:"),0)
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
            cv2.imshow("matches", img3)
            cv2.waitKey(0)

        # SURF
        elif metodoDetEmp == 2:
            print("Este algoritmo esta patentado y esta excluido de esta configuracion")
            #img1 = cv2.imread(input("Nombre de la imagen 1:"), 0)
            #img2 = cv2.imread(input("Nombre de la imagen 2:"), 0)
            #detector = cv2.xfeatures2d.SURF_create(400)
            #kp1, des1 = detector.detectAndCompute(img1, None)
            #kp2, des2 = detector.detectAndCompute(img2, None)
            #bf = cv2.BFMatcher()
            #matches = bf.knnMatch(des1, des2, k=2)
            #good = []
            #for m, n in matches:
                #if m.distance < 0.75 * n.distance:
                    #good.append([m])
            #img3=cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
            #cv2.imshow("matches",img3)
            #cv2.waitKey(0)
            Emparejamiento()

        # Kaze
        elif metodoDetEmp == 3:
            img1 = cv2.imread(input("Nombre de la imagen 1:"), 0)
            img2 = cv2.imread(input("Nombre de la imagen 2:"), 0)
            kaze = cv2.KAZE_create()
            kp1, des1 = kaze.detectAndCompute(img1, None)
            kp2, des2 = kaze.detectAndCompute(img2, None)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
            cv2.imshow("matches", img3)
            cv2.waitKey(0)

        # BRIEF
        elif metodoDetEmp == 4:
            img1 = cv2.imread(input("Nombre de la imagen 1:"), 0)
            img2 = cv2.imread(input("Nombre de la imagen 2:"), 0)
            star = cv2.xfeatures2d.StarDetector_create()
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            kp1 = star.detect(img1, None)
            kp1, des1 = brief.compute(img1, kp1)
            kp2 = star.detect(img2, None)
            kp2, des2 = brief.compute(img2, kp2)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
            cv2.imshow("matches", img3)
            cv2.waitKey(0)

        # BRISK
        elif metodoDetEmp == 5:
            img1 = cv2.imread(input("Nombre de la imagen 1:"), 0)
            img2 = cv2.imread(input("Nombre de la imagen 2:"), 0)
            detector = cv2.BRISK_create()
            kp1, des1 = detector.detectAndCompute(img1, None)
            kp2, des2 = detector.detectAndCompute(img2, None)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
            cv2.imshow("matches", img3)
            cv2.waitKey(0)

        # ORB
        elif metodoDetEmp==6:
            img1 = cv2.imread(input("Nombre de la imagen 1:"),0)
            img2 = cv2.imread(input("Nombre de la imagen 2:"),0)
            orb=cv2.ORB_create()
            kp1, des1 =orb.detectAndCompute(img1,None)
            kp2, des2 = orb.detectAndCompute(img2,None)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
            cv2.imshow("matches",img3)
            cv2.waitKey(0)

        #AKAZE
        elif metodoDetEmp==7:
            img1 = cv2.imread(input("Nombre de la imagen 1:"),0)
            img2 = cv2.imread(input("Nombre de la imagen 2:"),0)
            akaze=cv2.AKAZE_create()
            kp1, des1 =akaze.detectAndCompute(img1,None)
            kp2, des2 = akaze.detectAndCompute(img2,None)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
            cv2.imshow("matches",img3)
            cv2.waitKey(0)

        # Error
        else:
            print("Error")
            Emparejamiento()

    #Metodo FLANN
    elif metodoEmp==2:
        print("1.SIFT")
        print("2.SURF")
        print("3.KAZE")
        print("4.BRIEF")
        print("5.BRISK")
        print("6.ORB")
        print("7.AKAZE")
        metodoDetEmp = int(input("Seleccionar el metodo de deteccion:"))

        # SIFT
        if metodoDetEmp == 1:
            img1 = cv2.imread(input("Nombre de la imagen 1:"), 0)
            img2 = cv2.imread(input("Nombre de la imagen 2:"), 0)
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            des1=np.float32(des1)
            des2=np.float32(des2)
            matches = flann.knnMatch(des1, des2, k=2)
            matchesMask = [[0,0] for i in range(len(matches))]
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    matchesMask[i] = [1, 0]
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask,
                               flags=0)
            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
            cv2.imshow("matches", img3)
            cv2.waitKey(0)

        # SURF
        elif metodoDetEmp == 2:
            print("Este algoritmo esta patentado y esta excluido de esta configuracion")
            #img1 = cv2.imread(input("Nombre de la imagen 1:"), 0)
            #img2 = cv2.imread(input("Nombre de la imagen 2:"), 0)
            #surf = cv2.SURF_create()
            #kp1, des1 = surf.detectAndCompute(img1, None)
            #kp2, des2 = surf.detectAndCompute(img2, None)
            #FLANN_INDEX_KDTREE = 1
            #index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            #search_params = dict(checks=50)  # or pass empty dictionary
            #flann = cv2.FlannBasedMatcher(index_params, search_params)
            #des1 = np.float32(des1)
            #des2 = np.float32(des2)
            #matches = flann.knnMatch(des1, des2, k=2)
            #matchesMask = [[0, 0] for i in range(len(matches))]
            #for i, (m, n) in enumerate(matches):
                #if m.distance < 0.7 * n.distance:
                    #matchesMask[i] = [1, 0]
            #draw_params = dict(matchColor=(0, 255, 0),
                               #singlePointColor=(255, 0, 0),
                               #matchesMask=matchesMask,
                               #flags=0)
            #img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
            #cv2.imshow("matches", img3)
            #cv2.waitKey(0)
            Emparejamiento()

        # Kaze
        elif metodoDetEmp == 3:
            img1 = cv2.imread(input("Nombre de la imagen 1:"), 0)
            img2 = cv2.imread(input("Nombre de la imagen 2:"), 0)
            kaze = cv2.KAZE_create()
            kp1, des1 = kaze.detectAndCompute(img1, None)
            kp2, des2 = kaze.detectAndCompute(img2, None)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            des1 = np.float32(des1)
            des2 = np.float32(des2)
            matches = flann.knnMatch(des1, des2, k=2)
            matchesMask = [[0, 0] for i in range(len(matches))]
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    matchesMask[i] = [1, 0]
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask,
                               flags=0)
            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
            cv2.imshow("matches", img3)
            cv2.waitKey(0)

        # BRIEF
        elif metodoDetEmp == 4:
            img1 = cv2.imread(input("Nombre de la imagen 1:"), 0)
            img2 = cv2.imread(input("Nombre de la imagen 2:"), 0)
            star = cv2.xfeatures2d.StarDetector_create()
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            kp1 = star.detect(img1, None)
            kp1, des1 = brief.compute(img1, kp1)
            kp2 = star.detect(img2, None)
            kp2, des2 = brief.compute(img2, kp2)
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2
            search_params = dict(checks=300)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            matchesMask = [[0, 0] for i in range(len(matches))]
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    matchesMask[i] = [1, 0]
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask,
                               flags=0)
            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
            cv2.imshow("matches", img3)
            cv2.waitKey(0)

        # BRISK
        elif metodoDetEmp == 5:
            img1 = cv2.imread(input("Nombre de la imagen 1:"), 0)
            img2 = cv2.imread(input("Nombre de la imagen 2:"), 0)
            detector = cv2.BRISK_create()
            kp1, des1 = detector.detectAndCompute(img1, None)
            kp2, des2 = detector.detectAndCompute(img2, None)
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            matchesMask = [[0, 0] for i in range(len(matches))]
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    matchesMask[i] = [1, 0]
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask,
                               flags=0)
            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
            cv2.imshow("matches", img3)
            cv2.waitKey(0)

        # ORB
        elif metodoDetEmp == 6:
            img1 = cv2.imread(input("Nombre de la imagen 1:"), 0)
            img2 = cv2.imread(input("Nombre de la imagen 2:"), 0)
            detector = cv2.ORB_create()
            kp1, des1 = detector.detectAndCompute(img1, None)
            kp2, des2 = detector.detectAndCompute(img2, None)
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            matchesMask = [[0, 0] for i in range(len(matches))]
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    matchesMask[i] = [1, 0]
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask,
                               flags=0)
            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
            cv2.imshow("matches", img3)
            cv2.waitKey(0)

        # AKAZE
        elif metodoDetEmp == 7:
            img1 = cv2.imread(input("Nombre de la imagen 1:"), 0)
            img2 = cv2.imread(input("Nombre de la imagen 2:"), 0)
            detector = cv2.AKAZE_create()
            kp1, des1 = detector.detectAndCompute(img1, None)
            kp2, des2 = detector.detectAndCompute(img2, None)
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            matchesMask = [[0, 0] for i in range(len(matches))]
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    matchesMask[i] = [1, 0]
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask,
                               flags=0)
            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
            cv2.imshow("matches", img3)
            cv2.waitKey(0)

        # Error
        else:
            print("Error")
            Emparejamiento()


    elif metodoDetEmp == 3:
        mainMenu()

    else:
        print("Error, seleccionar una opcion de 1-3")
        Emparejamiento()


#main routine
mainMenu()

