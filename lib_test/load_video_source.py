#!/usr/bin/python3
import cv2 #esse é pra funcionamento de computer vision
from matplotlib import pyplot as plt #aqui é pra poder visualizar imagens

#def Pega_Frame():
    #cap = cv2.VideoCapture(0) #o parametro da função é pra ser o 'numero' do dispositivo de captura
    #ret, frame = cap.read() #pega os frames do dispositivo de captura

   # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #aqui ele usa a biblioteca do matplot pra visualizar o que está sendo armazenado na variavel frame

    #plt.show() #mostra a imagem pra gente

cap = cv2.VideoCapture(0)

while cap.isOpened(): #aqui ele faz um loop pra tirar vários frames enquanto o dispositivo de captura está ativo
    ret, frame = cap.read()

    cv2.imshow('Webcam', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cv2.imshow('GrayWebcam', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'): #aqui ele espera a tecla ser apertada pra quebrar o loop e consequententemente fechar o programa
        break
cap.release()
cv2.destroyAllWindows