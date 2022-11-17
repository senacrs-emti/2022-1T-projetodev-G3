import cv2
import numpy as np
import tkinter
import time
from keras.models import load_model


#Aqui ele importa o modelo, que eu exportei do Teachable Machine
#A gente utiliza o Keras pra carregar esse modelo
model = load_model('./arduino_leonardo_pro_micro/keras_model.h5')

#Aqui a gente armazena em uma variável o dispositivo de captura, ou seja, qual webcam você vai usar. Utiliza o opencv
camera = cv2.VideoCapture(1)

#Aqui ele lê o arquivo de classes de imagens, para a futura identificação, separando por linhas.
labels = open('./arduino_leonardo_pro_micro/labels.txt', 'r').readlines()

#Aqui criamos um loop para que o código ocorra enquanto a câmera está ativa
while camera.isOpened():
    #Delay para não ficar floodando o terminal
    time.sleep(2)
    #Aqui armazenamos se a câmera está ativa em 'ret', e armazenamos a leitura da câmera em 'image'
    ret, image = camera.read()
    #Aqui definimos um tamanho para a nossa janela, isso é importante para a visão de máquina
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    #Aqui utilizamos o opencv para abrir uma janela de visualização, chamada 'Webcam', utilizando os dados armazenados em 'image'
    cv2.imshow('Webcam', image)
    #Aqui transformamos os dados para um array com numpy e definimos o tamanho da tela como visto antes para a visão de máquina poder analisar corretamente de acordo com o modelo gerado
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    #Aqui armazenamos em 'probabilities' o atual item mostrado na câmera, utilizando a função de predição com o modelo, aplicando este "predict" nos dados da variável image
    probabilities = model.predict(image)
    #Aqui definimos qual a probabilidade de ser uma determinada label, e armazenamos isto como string em 'label_probability'
    label_probability = str(labels[np.argmax(probabilities)])
        
    window = Tk()
    window.title("Identificação de recipiente")
    if label_probability[0:1] == "0":
        print("Puxando recipiente de plástico, aguarde...")
    elif label_probability[0:1] == "1":
        print("Puxando recipiente de alumínio, aguarde...")
    elif label_probability[0:1] == "2":
        print("Puxando recipiente de alumínio, aguarde...")
    else:
        print("Este resíduo será encaminhado para o recipiente misto para melhor averiguação.")
    janela = mainloop()
    print(label_probability[0:1])
    
    keyboard_input = cv2.waitKey(1)
    # 27 ASCII = esc 
    if keyboard_input == 27:
        break


camera.release()
cv2.destroyAllWindows()

