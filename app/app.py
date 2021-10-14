from flask import Flask, request
from flask_restx import Resource, Api, Namespace, fields, reqparse

import torch
import pytorch_lightning as pl
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer
)

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from IntentTagger import IntentTagger

import tensorflow_text
import tensorflow_hub as hub
import numpy as np

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker, Query
from waitress import serve

import json
import logging


import py_eureka_client.eureka_client as eureka_client
eureka_client.init(eureka_server="http://ec2-44-199-108-119.compute-1.amazonaws.com:8761/eureka/",
                   eureka_protocol="http",
                   # eureka_context="/eureka/v2",
                   app_name="chatbot-service",
                   instance_ip="34.207.149.185",
                   instance_port=5000)

app = Flask(__name__)
api = Api(app)

engine = create_engine(
    'mysql://admin:HYhmiYxkwD5xMAs@db-maynardcode.cbjharuwcybj.us-east-1.rds.amazonaws.com:3306/maynardcode', echo=False)
Base = declarative_base()
Base.metadata.reflect(engine)

db_session = scoped_session(sessionmaker(bind=engine))

print('App initialized')

categorias = ['int_boleta',
              'int_craest',
              'int_economia',
              'int_eeggcc',
              'int_eventos',
              'int_matricula',
              'int_reclamo_notas',
              'int_retiro',
              'int_salud',
              'int_transferencia',
              'int_tutores']


class PythonPredictor:
    def __init__(self):
        self.device = "cpu"
        self.tokenizer = BertTokenizer.from_pretrained(
            'dccuchile/bert-base-spanish-wwm-cased')
        self.trained_model = IntentTagger.load_from_checkpoint(
            './best-checkpoint.ckpt',
            n_classes=len(categorias)
        )

        self.trained_model.eval()
        self.trained_model.freeze()

    def predict(self, payload):

        response = {'intencion': None, 'probabilidad_int': None,
                    'rpta': None, 'probabilidad_preg': None}

        test_comment = payload
        encoding = self.tokenizer.encode_plus(
            test_comment,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',
        )
        _, test_prediction = self.trained_model(
            encoding["input_ids"], encoding["attention_mask"])
        test_prediction = test_prediction.flatten().numpy()

        best_label = None
        best_prediction = 0.0
        for label, prediction in zip(categorias, test_prediction):
            # print(f"{label}: {prediction}")
            if prediction > best_prediction:
                best_label = label
                best_prediction = prediction
                response['intencion'] = best_label
                response['probabilidad_int'] = str(best_prediction)

        return response


predictor = PythonPredictor()
embed = hub.load(
    "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

print('Finished init')

logging.basicConfig(filename='record.log', level=logging.DEBUG,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')


resource_fields = api.model('Resource', {
    'data': fields.String,
})

parser = reqparse.RequestParser()
parser.add_argument('data', type=str, help='variable 1')


class Pregunta_Frecuente(Base):
    __table__ = Base.metadata.tables['pregunta_frecuente']


def similarity(intencion, text):
    faqs = []
    datas = []
    for item in db_session.query(Pregunta_Frecuente):
        faqs.append(item.pregunta)
        datas.append(item)

    # compute embeddings
    text_result = embed(text)
    faqs_results = embed(faqs)

    # compute similarity matrix
    similarity_matrix = np.inner(text_result, faqs_results)
    print(similarity_matrix)

    # find index of max prob
    max_index = None
    max_prob = None
    THRESHOLD = 0.7

    if similarity_matrix[0].size != 0:
        max_index = np.where(similarity_matrix[0] == np.amax(
            similarity_matrix[0]))[0][0]
        max_prob = similarity_matrix[0][max_index]

    if max_index == None or max_prob < THRESHOLD:
        return ('Disculpa por no poder responder tu pregunta he contactado a alguien para ayudarte', str(max_prob))

    return (datas[max_index].respuesta, str(max_prob))


@api.route('/chatbot')
class Chatbot(Resource):

    @api.doc(parser=parser)
    def post(self):
        text = parser.parse_args()['data']
        response = predictor.predict(text)

        # if no match return no intention
        # TODO move threshhold logic inside of function
        if float(response['probabilidad_int']) < 0.5:
            response['intencion'] = 'int_ninguna'
            response['probabilidad_preg'] = '0.0'
            response['rpta'] = 'Disculpa podrias refrasear la pregunta?'
            return response

        # search tensorflow similarity
        response['rpta'], response['probabilidad_preg'] = similarity(
            response['intencion'], text)

        return response

    def get(self):
        return {'categorias': categorias}


# @api.route('/chatbot')
# class Chatbot(Resource):
#     # @api.doc(parser=parser)
#     def post(self):
#         # text = parser.parse_args()['data']
#         # response = predictor.predict(text)
#         # return response
#         return None

#     def get(self):
#         # return {'categorias': categorias}
#         for item in db_session.query(Pregunta_Frecuente).filter(Pregunta_Frecuente.id_pregunta_frecuente == 1):
#             print(item.id_pregunta_frecuente)
#         #     print(item)
#         # for item in db_session.query(Pregunta_Frecuente):
#         #     print(item.id_pregunta_frecuente)
#         return None


if __name__ == '__main__':
    app.run(debug=True, port=5000)
    # serve(app, host='127.0.0.1', port=5000)
