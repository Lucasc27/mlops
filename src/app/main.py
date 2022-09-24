from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import pickle
import json
import os

colunas = ['tamanho','ano','garagem']
modelo = pickle.load(open('../../models/modelo.sav','rb'))

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'lucas'#os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = '123'#os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)

@app.route('/')
def home():
    return "Minha primeira API."

@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt_br',to='en')
    polaridade = tb_en.sentiment.polarity
    return "polaridade: {}".format(polaridade)

@app.route('/cotacao_old/', methods=['POST'])
@basic_auth.required
def cotacao_old():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])

@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():
    dados = request.get_json()
    dados_input = [[lista[col] for col in colunas] for lista in [dados[i] for i in dados.keys()]]
    preco = modelo.predict(dados_input)
    return jsonify(preco=list(preco))

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)