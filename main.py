import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from numpy import asarray
import pickle
import numpy as np

st.title("Reconocimiento de Digitos Escritos a Mano")

st.subheader("Dibuja un digito")

# Specify canvas parameters in application
drawing_mode = "freedraw"
stroke_width = 3
stroke_color = "#000"
bg_color = "#fff"
realtime_update = True

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=realtime_update,
    height=140,
    width=140,
    drawing_mode=drawing_mode,
    key="canvas",
)

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    st.subheader("Imagen redimensionada")

    # Transformaciones de imagen
    # Crear imagen a partir de array
    im = Image.fromarray(canvas_result.image_data)
    # 1. escala de grises
    im = ImageOps.grayscale(im)
    im = ImageOps.invert(im)
    # 2. resize a 28x28
    im = im.resize((28, 28))

    st.image(im)

    result_array = asarray(im).flatten()
    # st.write(result_array.shape)
    # st.write(result_array)

    # Create column names from 'pixel0' to 'pixel783'
    column_names = [f'pixel{i}' for i in range(784)]
    # Reshape the array to (1, 784) and convert to DataFrame
    df = pd.DataFrame(result_array.reshape(1, -1), columns=column_names)

    df = df / 255.0
    df = df.values.reshape(-1, 28, 28, 1)

    # Carga de modelo con pickle

    # from pathlib import Path
    #
    # my_file = Path("model_pickle")
    # st.write(my_file.is_file())

    # open a file, where you stored the pickled data
    file = open('model_pickle', 'rb')

    # dump information to that file
    model = pickle.load(file)

    # close the file
    file.close()

    results = model.predict(df)
    st.subheader("Probabilidad asignada a cada número")
    st.write(results)
    results = np.argmax(results, axis=1)
    results = pd.Series(results, name="Label")

    st.header("El numero que dibujaste es:")
    st.header(f"{results[0]}")

if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")

    # st.subheader("Canvas objects")
    # st.dataframe(objects)
