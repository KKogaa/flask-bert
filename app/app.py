import torch
import pytorch_lightning as pl
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer
)
import json
import logging

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from IntentTagger import IntentTagger

from flask import Flask, request
from flask_restx import Resource, Api, Namespace, fields, reqparse

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker, Query

import json

import py_eureka_client.eureka_client as eureka_client
eureka_client.init(eureka_server="http://ec2-44-199-108-119.compute-1.amazonaws.com:8761/eureka/",
                   eureka_protocol="http",
                   # eureka_context="/eureka/v2",
                   app_name="chatbot-service",
                   instance_ip="3.82.208.24",
                   instance_port=5000)

app = Flask(__name__)
api = Api(app)

engine = create_engine(
    'mysql://admin:HYhmiYxkwD5xMAs@db-maynardcode.cbjharuwcybj.us-east-1.rds.amazonaws.com:3306/maynardcode', echo=False)
Base = declarative_base()
Base.metadata.reflect(engine)

db_session = scoped_session(sessionmaker(bind=engine))

print('App initialized')

categorias = ['cat_boleta',
              'cat_craest',
              'cat_economia',
              'cat_eeggcc',
              'cat_eventos',
              'cat_matricula',
              'cat_reclamo_notas',
              'cat_retiro',
              'cat_salud',
              'cat_transferencia',
              'cat_tutores']


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
        THRESHOLD = 0.5
        for label, prediction in zip(categorias, test_prediction):
            # print(f"{label}: {prediction}")
            if prediction > best_prediction and prediction > THRESHOLD:
                best_label = label
                best_prediction = prediction

        if best_label is None:
            return {'cat_ninguno': '0.0'}

        return {best_label: str(best_prediction)}


predictor = PythonPredictor()

print('Finished init')

logging.basicConfig(filename='record.log', level=logging.DEBUG,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')


@api.route('/categorizar')
class Categorizar(Resource):

    def post(self):
        text = parser.parse_args()['data']
        response = predictor.predict(text)
        return response

    def get(self):
        return {'categorias': categorias}


class Pregunta_Frecuente(Base):
    __table__ = Base.metadata.tables['pregunta_frecuente']


@api.route('/chatbot')
class Chatbot(Resource):
    # @api.doc(parser=parser)
    def post(self):
        # text = parser.parse_args()['data']
        # response = predictor.predict(text)
        # return response
        return None

    def get(self):
        # return {'categorias': categorias}
        for item in db_session.query(Pregunta_Frecuente).filter(Pregunta_Frecuente.id_pregunta_frecuente == 1):
            print(item.id_pregunta_frecuente)
        #     print(item)
        # for item in db_session.query(Pregunta_Frecuente):
        #     print(item.id_pregunta_frecuente)
        return None


if __name__ == '__main__':
    app.run(debug=True, port=5000)
