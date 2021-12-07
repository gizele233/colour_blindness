import streamlit as st
import cv2
import imutils
# from daltonize import daltonize
import numpy
from PIL import Image


# def redim(img, largura): #função para redimensionar uma imagem
#     alt = int(img.shape[0]/img.shape[1]*largura)
#     img = cv2.resize(img, (largura, alt), interpolation = cv2.INTER_AREA)

#     return img

def correcao(RGB, filtro):     
    # if frame.mode in ['1', 'L']: # Don't process black/white or grayscale images
    #     return 
    # frame = frame.copy()
    # im = frame.convert('RGB') 
    # RGB = numpy.asarray(im, dtype=float) 
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) *
    # RGB = numpy.asarray(frame, dtype=float) *

    # Transformation matrix for Deuteranope (a form of red/green color deficit)
    lms2lmsd = numpy.array([[1,0,0],[0.494207,0,1.24827],[0,0,1]])
    # Transformation matrix for Protanope (another form of red/green color deficit)
    lms2lmsp = numpy.array([[0,2.02344,-2.52581],[0,1,0],[0,0,1]])
    # Transformation matrix for Tritanope (a blue/yellow deficit - very rare)
    lms2lmst = numpy.array([[1,0,0],[0,1,0],[-0.395913,0.801109,0]])

    # Colorspace transformation matrices
    rgb2lms = numpy.array([[17.8824,43.5161,4.11935],[3.45565,27.1554,3.86714],[0.0299566,0.184309,1.46709]])
    lms2rgb = numpy.linalg.inv(rgb2lms)
    # Daltonize image correction matrix
    err2mod = numpy.array([[0,0,0],[0.7,1,0],[0.7,0,1]])

     # Get the requested image correction
    if filtro == 2:
        lms2lms_deficit = lms2lmsd
    elif filtro == 3:
        lms2lms_deficit = lms2lmsp
    elif filtro == 4:
        lms2lms_deficit = lms2lmst

    # Transform to LMS space
    LMS = numpy.zeros_like(RGB)               
    for i in range(RGB.shape[0]):
        for j in range(RGB.shape[1]):
            rgb = RGB[i,j,:3]
            LMS[i,j,:3] = numpy.dot(rgb2lms, rgb) #dot faz produto escalar entre duas matrizes

    # Calculate image as seen by the color blind
    _LMS = numpy.zeros_like(RGB)  
    for i in range(RGB.shape[0]):
        for j in range(RGB.shape[1]):
            lms = LMS[i,j,:3]
            _LMS[i,j,:3] = numpy.dot(lms2lms_deficit, lms) #numpy.dot faz o produto escalar entre duas matrizes

    _RGB = numpy.zeros_like(RGB) 
    for i in range(RGB.shape[0]):
        for j in range(RGB.shape[1]):
            _lms = _LMS[i,j,:3]
            _RGB[i,j,:3] = numpy.dot(lms2rgb, _lms)

    # Calculate error between images
    error = (RGB-_RGB)

    # Daltonize
    #Calcular valores de compensação
    ERR = numpy.zeros_like(RGB) 
    for i in range(RGB.shape[0]):
        for j in range(RGB.shape[1]):
            err = error[i,j,:3]
            ERR[i,j,:3] = numpy.dot(err2mod, err)

    #Adicionar valores de compensação a imagem original
    dtpn = ERR + RGB
  
    for i in range(RGB.shape[0]):
        for j in range(RGB.shape[1]):
            dtpn[i,j,0] = max(0, dtpn[i,j,0])
            dtpn[i,j,0] = min(255, dtpn[i,j,0])
            dtpn[i,j,1] = max(0, dtpn[i,j,1])
            dtpn[i,j,1] = min(255, dtpn[i,j,1])
            dtpn[i,j,2] = max(0, dtpn[i,j,2])
            dtpn[i,j,2] = min(255, dtpn[i,j,2])

    result = dtpn.astype('uint8')
    # converted = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) *
    converted = Image.fromarray(result, mode='RGB')
    # im_converted = Image.fromarray(result, mode='RGB')

    return converted

def show_image():
    img_file_buffer = st.sidebar.file_uploader('Carregue uma imagem', type=['jpg','jpeg','png'])
    
    im_original = Image.open(img_file_buffer)
    im = im_original.copy()
    im = im.convert('RGB') 
    frame = numpy.asarray(im, dtype=float)

    # captura = cv2.VideoCapture(0)
    # captura = cv2.VideoCapture('video2.mp4')

    # frame = cv2.imread('folhas.png') *

    # frame = Image.open("gizele.jpg")
    
    # You may need to convert the color.
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # captura = Image.fromarray(frame)

    filtro=1

    st.sidebar.markdown('## Filtro para a tabela teste')
    tipos = numpy.unique(['normal', 'deuteranopia', 'protanopia', 'tritanopia'])
    filtro = st.sidebar.selectbox('Selecione a categoria para apresentar na tabela', options = tipos)

    # while True:
        # ret, frame = captura.read()
    
    if img_file_buffer is not None:
        st.write(filtro)
        frame = imutils.resize(frame, width=700)
        # cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # frame = Image.fromarray(numpy.uint8(cv2_im))

#         frame = redim(frame, 320)
        
        # desenhando sobre a câmera
        # fonte = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(frame,'Menu',(13,65), fonte, 1,(210,105,30),2,cv2.LINE_AA)
        # cv2.putText(frame,'1-Img Original',(13,100), fonte, 0.5,(210,105,30),1,cv2.LINE_AA)
        # cv2.putText(frame,'2-Deuteranopia',(13,120), fonte, 0.5,(210,105,30),1,cv2.LINE_AA)
        # cv2.putText(frame,'3-Protanotopia',(13,140), fonte, 0.5,(210,105,30),1,cv2.LINE_AA)
        # cv2.putText(frame,'4-Tritanopia',(13,160), fonte, 0.5,(210,105,30),1,cv2.LINE_AA)
        # cv2.putText(frame,'esc-sair',(13,180), fonte, 0.5,(210,105,30),1,cv2.LINE_AA)
        
        # if filtro == 1:
        #     cv2.imshow('Estudo OpenCV- Filtro', frame)
            
        # if filtro == 2: 
        #     deut = correcao(frame, 2)
        #     cv2.imshow('Estudo OpenCV- Deuteranopia', deut)
        
        # if filtro == 3: 
        #     prot = correcao(frame, 3)
        #     cv2.imshow('Estudo OpenCV- Protanotopia', prot)
        
        # if filtro == 4: 
        #     trit = correcao(frame, 4)
        #     cv2.imshow('Estudo OpenCV- Tritanopia', trit)
        

        if filtro == 'normal':
            # cv2.imshow('Estudo OpenCV', frame)
            st.image(im_original, width=700)
            
        if filtro == 'deuteranopia': 
            deut = correcao(frame, 2)
            # cv2.imshow('Estudo OpenCV', deut)
            st.image(deut)
        
        if filtro == 'protanopia': 
            prot = correcao(frame, 3)
            # cv2.imshow('Estudo OpenCV', prot)
            st.image(prot)
        
        if filtro == 'tritanopia': 
            trit = correcao(frame, 4)
            # cv2.imshow('Estudo OpenCV', trit)
            st.image(trit)

        # key = cv2.waitKey(1) 
        
        # if the 'esc' key is pressed, stop the loop
        # if key == 27:   
        #     break
        
        # elif key == -1: 
        #     continue
        
        # elif key == 49:
        #     filtro= 1
        
        # elif key == 50:
        #     filtro= 2
            
        # elif key == 51:
        #     filtro= 3
        #     st.sidebar.markdown
        # elif key == 52:
        #     filtro= 4 
    else:
        st.title('Adicione uma imagem')
            

    # captura.release() #libera a captura
    # cv2.destroyAllWindows() #libera a memória
    
def main():
    st.title('Correção de imagens :sunglasses:')
    st.write('Uso de processamento de imagem para correção de imagens para daltônicos')
    show_image()
    return 0
 
if __name__ == '__main__':
    main()

