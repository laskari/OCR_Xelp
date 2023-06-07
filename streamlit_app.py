import streamlit as st
from http import client
import os,json
import pandas as pd
import requests
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
st.header("Xelpmoc - Optical Character Recognition - Document AI")

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

def TrOCR_predict(pixel_values, processor, model):
    generated_ids = model.generate(pixel_values,output_scores=True,return_dict_in_generate=True, max_length = 64)
    predicted_text = processor.batch_decode(generated_ids[0], skip_special_tokens=True)
    return predicted_text


df = pd.DataFrame()
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    content = uploaded_file.read()
    st.image(uploaded_file)
    image = Image.open(uploaded_file)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    predicted_text = TrOCR_predict(pixel_values, processor, model)[0]
    texts = predicted_text

    st.write(texts)
