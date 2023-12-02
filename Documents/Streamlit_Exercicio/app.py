import streamlit as st 
import pandas as pd
import numpy as np
from pycaret.regression import load_model, predict_model
from pycaret.datasets import get_data


st.set_page_config(
    page_title="Previsão de Doença Cardíaca",
    page_icon="\u2764\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded")


st.title('❤️ Previsão de Doença Cardíaca')
st.write('Entre com as caracteristicas do paciente para fazer uma previsão:')

dados = get_data('heart_disease')
#Algumas colunas estão com espaço em branco no final do nome
dados.rename(columns=lambda x: x.strip(), inplace=True)

modelo = load_model('recursos/modelo-previsao-heart-disease')

def trad(x):
	return 'Masculino' if x == 1 else 'Feminino'

def trad2(x):
	return 'Sim' if x == 1 else 'Não'

col1, col2, col3, col4 = st.columns([3,3,3,3])


with col1:
	age = st.slider(label = 'Idade', 
		min_value=18, 
		max_value=80, 
		value= 40, 
		step=1, 
		help='Entre com a idade do paciente')
	sex	= st.radio('Sexo', [1, 0], format_func = trad)
	serum_cholestoral = st.number_input(label = 'Colesterol sérico (mg/dL)', 
		min_value = 100, 
		max_value = 600, 
		value = 200, 
		step = 1,
		help='O nível de colesterol sérico no sangue do paciente, medido em miligramas por decilitro (mg/dL).') 

resting_electro_ordenadas = sorted(dados['resting electrocardiographic results'].unique())
with col2: 
	maximum_heart_rate = st.number_input(label = 'Frequência cardíaca', 
		min_value = 50, 
		max_value = 250, 
		value = 150, 
		step = 1,
		help='A frequência cardíaca máxima alcançada durante um teste de esforço.') 
	oldpeak = st.slider(label = 'Depressão do segmento ST', 
		min_value=0., 
		max_value=6.5, 
		value= 1.5, 
		step=0.1, 
		help='A depressão do segmento ST no eletrocardiograma induzida pelo exercício.')	
	resting_electrocardiographic = st.selectbox(label = 'Resultado eletrocardiograma', 
		options = resting_electro_ordenadas,
		help='O resultado do eletrocardiograma em repouso do paciente.',
		placeholder="Select contact method...")

number_vessels_ordenadas = sorted(dados['number of major vessels'].unique())
thal_ordenadas = sorted(dados['thal'].unique())

with col3:
	number_vessels = st.selectbox(label = 'Número de vasos', 
		options = number_vessels_ordenadas,
		help='O número de principais vasos sanguíneos do coração coloridos por fluoroscopia.')
	resting_blood_pressure = st.slider(label = 'Pressão arterial em repouso', 
		min_value=80, 
		max_value=220, 
		value= 120, 
		step=1, 
		help='A pressão arterial sistólica em repouso do paciente.')
	thal = st.selectbox(label = 'Thalassemia', 
		options = thal_ordenadas,
		help='Um tipo de anemia hereditária que afeta a produção de hemoglobina.')

chest_pain_type_ordenadas = sorted(dados['chest pain type'].unique())
slope_of_peak_ordenadas = sorted(dados['slope of peak'].unique())
with col4:
	chest_pain_type = st.selectbox(label = 'Tipo de dor no peito', 
		options = chest_pain_type_ordenadas,
		help='O tipo de dor no peito relatado pelo paciente.')
	fasting_blood_sugar = st.radio('Açúcar no sangue em jejum > 120 mg/dL', [1,0], 
		format_func = trad2,
		help='O paciente tem açúcar no sangue em jejum acima de 120 mg/dL?')
	slope_of_peak = st.selectbox(label = 'Inclinação do segmento ST', 
		options = slope_of_peak_ordenadas,
		help='Inclinação do segmento ST durante o pico do exercício.')
	


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
prever_exibicao = prever
prever_exibicao = prever_exibicao.rename(columns={
    'age': 'Idade',
    'resting blood pressure': 'Pressão arterial em repouso',
    'serum cholestoral in mg/dl': 'Colesterol sérico (mg/dL)',
    'maximum heart rate achieved': 'Frequência cardíaca',
    'oldpeak': 'Depressão do segmento ST',
    'resting electrocardiographic results': 'Resultado eletrocardiograma',
    'number of major vessels': 'Número de vasos',
    'thal': 'Thalassemia',
    'sex': 'Sexo',
    'chest pain type': 'Tipo de dor no peito',
    'fasting blood sugar > 120 mg/dl': 'Açúcar no sangue em jejum > 120 mg/dL',
    'slope of peak': 'Inclinação do segmento ST'
})
st.dataframe(prever_exibicao)

def previsao():
	previsao = predict_model(modelo, data = prever)
	valor = previsao.loc[0,'prediction_label']
	resposta = "possui Doença Cardíaca" if valor == 1 else "não possui Doença Cardíaca"
	c0.write(f'### O paciente {resposta}.')

c0,_,c1 = st.columns([3,1,2])

with c1:
	st.button('Verificar previsão',
		type = 'primary',
		use_container_width = True,
		on_click=previsao)