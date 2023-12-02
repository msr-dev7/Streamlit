import streamlit as st 
import pandas as pd
import numpy as np
from pycaret.regression import load_model, predict_model
from pycaret.datasets import get_data

dados = get_data('heart_disease')
#Algumas colunas estão com espaço em branco no final do nome
dados.rename(columns=lambda x: x.strip(), inplace=True)

modelo = load_model('recursos/modelo-previsao-heart-disease')

def trad(x):
	return 'Masculino' if x == 1 else 'Feminino'

def trad2(x):
	return 'Sim' if x == 1 else 'Não'



st.header('Previsão de Doença Cardíaca')
st.write('Entre com as caracteristicas do paciente para fazer uma previsão.')



#Widgets para fazer os inputs do modelo

col1, col2, col3 = st.columns([3,3,3])


with col1:
	age = st.slider(label = 'Idade', 
		min_value=18, 
		max_value=64, 
		value= 40, 
		step=1, 
		help='Entre com a idade do indivíduo')
	resting_blood_pressure = st.slider(label = 'Pressão arterial em repouso', 
		min_value=80, 
		max_value=220, 
		value= 120, 
		step=1, 
		help='Entre com o valor')
	serum_cholestoral = st.number_input(label = 'Colesterol sérico em mg/dL', 
		min_value = 100, 
		max_value = 600, 
		value = 200, 
		step = 1) 
	maximum_heart_rate = st.number_input(label = 'Frequência cardíaca máxima atingida', 
		min_value = 50, 
		max_value = 250, 
		value = 150, 
		step = 1) 
with col2: 
	oldpeak = st.slider(label = 'Depressão do segmento ST induzida pelo exercício', 
		min_value=0., 
		max_value=6.5, 
		value= 1.5, 
		step=0.1, 
		help='Entre com o valor')	
	resting_electrocardiographic = st.selectbox(label = 'Resultados eletrocardiográficos em repouso', 
		options = dados['resting electrocardiographic results'].unique())
	number_vessels = st.selectbox(label = 'Número de vasos principais coloridos por fluoroscopia', 
		options = dados['number of major vessels'].unique())
	thal = st.selectbox(label = 'Thalassemia', 
		options = dados['thal'].unique())

with col3:
	sex	= st.radio('Sexo', [1, 0], format_func = trad)
	chest_pain_type = st.selectbox(label = 'Tipo de dor no peito', 
		options = dados['chest pain type'].unique())
	slope_of_peak = st.selectbox(label = 'Inclinação do segmento ST durante o pico do exercício', 
		options = dados['slope of peak'].unique())
	fasting_blood_sugar = st.radio('Açúcar no sangue em jejum > 120 mg/dL', [1,0], format_func = trad2)


#Criar um DataFrame com os inputs exatamente igual ao dataframe em que foi treinado o modelo
aux = {'age': [age],
		'resting blood pressure': [resting_blood_pressure],
		'serum cholestoral in mg/dl': [serum_cholestoral],
		'maximum heart rate achieved': [maximum_heart_rate],
		'oldpeak': [oldpeak],
		'resting electrocardiographic results': [resting_electrocardiographic],
		'number of major vessels': [number_vessels],
		'thal': [thal],
		'sex' : [sex],
		'chest pain type': [chest_pain_type],
		'fasting blood sugar > 120 mg/dl': [fasting_blood_sugar],
		'slope of peak': [slope_of_peak]}

prever = pd.DataFrame(aux)

st.write(prever)

#Usar o modelo salvo para fazer previsao nesse Dataframe

_, c1, _ = st.columns([2,3,1])

with c1:
	botao = st.button('Verificar previsão',
		type = 'primary',
		use_container_width = True)

if botao:
	previsao = predict_model(modelo, data = prever)
	valor = previsao.loc[0,'prediction_label']
	resposta = "possui Doença Cardíaca" if valor == 1 else "não possui Doença Cardíaca"
	st.write(f'### O paciente {resposta}.')