import streamlit as st
import imutils
import numpy
from PIL import Image


def correcao(RGB, filtro):     

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
    st.sidebar.markdown('## Carregue uma imagem')
    img_file_buffer = st.sidebar.file_uploader('Selecione uma imagem', type=['jpg','jpeg','png'])


    if img_file_buffer is not None:
        im_original = Image.open(img_file_buffer)
        im = im_original.copy()
        im = im.convert('RGB') 
        frame = numpy.asarray(im, dtype=float)

        st.sidebar.markdown('## Escolha o tipo de daltonismo')
        filtro = st.sidebar.selectbox('Selecione o tipo', ('normal', 'deuteranopia', 'protanopia', 'tritanopia'))

        st.write('➜Tipo escolhido:', filtro)
        frame = imutils.resize(frame, width=600)
        
        if filtro == 'normal':
            # cv2.imshow('Estudo OpenCV', frame)
            st.image(im_original, width=600)
            
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
    else:
        img = Image.open('photocapa.jpg')
        st.image(img)
        return 0
            

    # captura.release() #libera a captura
    # cv2.destroyAllWindows() #libera a memória
    
def main():
    st.title('Aplicação de processamento de imagens para daltônicos')
    show_image()
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    return 0
 
if __name__ == '__main__':
    main()


