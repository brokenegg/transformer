from flask import Flask
from flask import request

from brokenegg_transformer import model_params
from brokenegg_transformer import transformer
from brokenegg_transformer.utils import tokenizer
import tensorflow.compat.v1 as tf
import numpy as np
import os

MODEL_DIR = os.getenv("MODEL_DIR", "/tmp")
VOCAB_FILE = os.path.join(MODEL_DIR, 'wikimatrix_lang10.spm64k.model')

langs = sorted({
  'en', 'es', 'fr', 'ru', 'de', 'ja', 'ar', 'zh', 'el', 'ko'
})
lang_map = {k: v + 64000 for v, k in enumerate(langs)}

subtokenizer = tokenizer.Subtokenizer(VOCAB_FILE)

params = model_params.BASE_PARAMS.copy()
params["dtype"] = tf.float32
with tf.name_scope("model"):
    model = transformer.create_model(params, is_train=False)
init_weight_path = tf.train.latest_checkpoint(MODEL_DIR)
print('Restoring from %s' % init_weight_path)
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(init_weight_path)

def translate(text, lang_id):
    encoded = subtokenizer.encode(text, add_eos=True)
    output, score = model.predict([np.array([encoded], dtype=np.int64), np.array([lang_id], dtype=np.int32)])
    encoded_output = []
    for _id in output[0]:
        _id = int(_id)
        if _id == tokenizer.EOS_ID:
            break
        encoded_output.append(_id)
    decoded_output = subtokenizer.decode(encoded_output)
    return decoded_output

app = Flask(__name__)

@app.route('/')
def hello_world():
    input_text = request.args.get('input', '')
    lang = request.args.get('lang', 'en')
    lang_id = lang_map.get(lang, 0)
    print(input_text)
    print(lang_id)
    return translate(input_text, lang_id)