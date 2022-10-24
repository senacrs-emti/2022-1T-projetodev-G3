#!/usr/bin/python3
import cv2 #esse é pra funcionamento de computer vision
import numpy as np
from matplotlib import pyplot as plt #aqui é pra poder visualizar imagens

#def Pega_Frame():
    #cap = cv2.VideoCapture(0) #o parametro da função é pra ser o 'numero' do dispositivo de captura
    #ret, frame = cap.read() #pega os frames do dispositivo de captura

   # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #aqui ele usa a biblioteca do matplot pra visualizar o que está sendo armazenado na variavel frame

    #plt.show() #mostra a imagem pra gente

cap = cv2.VideoCapture(0)

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=5)

while cap.isOpened(): #aqui ele faz um loop pra tirar vários frames enquanto o dispositivo de captura está ativo
    _, frame = cap.read()
    height, width, _ = frame.shape

    roi = frame[150:850, 0:350] #extrai a região de interesse, ou seja, define um frame pra aplicar o contorno

    mask = object_detector.apply(frame) #aqui aplicamos uma mascara para tornar o tracking dos objetos mais fácil

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    for cnt in contours:   
        
        area = cv2.contourArea(cnt)
        if area > 100:                                              
            cv2.drawContours(frame, [cnt], -1,(0, 255, 0 ), 2)      
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)


    cv2.imshow('Frame', roi)                                                                             
    cv2.imshow('Mask', mask)
    cv2.imshow('Webcam', frame)
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cv2.imshow('GrayWebcam', gray)
    #cv2.imshow('HSV', hsv)

    #lower_red = np.array([0, 20, 0])
    #upper_red = np.array([180, 255, 255])

    #mascara = cv2.inRange(hsv, lower_red, upper_red)
    #cv2.imshow = ('mascara', mascara)

    if cv2.waitKey(1) & 0xFF == ord('q'): #aqui ele espera a tecla ser apertada pra quebrar o loop e consequententemente fechar o programa
        break
cap.release()
cv2.destroyAllWindows 