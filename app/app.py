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
from ToxicCommentTagger import ToxicCommentTagger

from flask import Flask, request
from flask_restx import Resource, Api, Namespace, fields, reqparse

# https://abhtri.medium.com/flask-api-documentation-using-flask-restx-swagger-for-flask-84be13d70e0
# https://flask-restx.readthedocs.io/en/latest/scaling.html

app = Flask(__name__)
api = Api(app)

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
        self.trained_model = ToxicCommentTagger.load_from_checkpoint(
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


resource_fields = api.model('Resource', {
    'data': fields.String,
})

parser = reqparse.RequestParser()
parser.add_argument('data', type=str, help='variable 1')

predictor = PythonPredictor()

print('Finished init')

logging.basicConfig(filename='record.log', level=logging.DEBUG,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')


@api.route('/categorizar')
class Categorizar(Resource):
    @api.doc(parser=parser)
    def post(self):
        text = parser.parse_args()['data']
        response = predictor.predict(text)
        return response

    def get(self):
        return {'categorias': categorias}


if __name__ == '__main__':
    app.run(debug=True, port=5000)
