#!/usr/bin/python3
import cv2 #esse é pra funcionamento de computer vision
from matplotlib import pyplot as plt #aqui é pra poder visualizar imagens

#conectando à webcam
cap = cv2.VideoCapture(0) #o parametro da função é pra ser o 'numero' do dispositivo de captura
ret, frame = cap.read() #pega os frames do dispositivo de captura

plt.imshow(frame) #aqui ele usa a biblioteca do matplot pra visualizar o que está sendo armazenado na variavel frame

plt.show() #mostra a imagem pra gente


