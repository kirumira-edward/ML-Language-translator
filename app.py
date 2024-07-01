import time
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask_caching import Cache

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
@cache.cached(timeout=300, key_prefix='translation')
def translate():
    src_lang = request.form['src_lang']
    tgt_lang = request.form['tgt_lang']
    text = request.form['text']

    start_time = time.time()

    # Translate text
    translated_text = translate_text(text, src_lang, tgt_lang)

    end_time = time.time()
    translation_time = end_time - start_time

    return jsonify({
        'translated_text': translated_text,
        'translation_time': translation_time
    })

def translate_text(text, src_lang="eng_Latn", tgt_lang="swh_Latn"):
    """
    Translates text from source language to target language using the NLLB model.
    """
    tokenizer.src_lang = src_lang
    encoded_input = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_input,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text

if __name__ == '__main__':
    app.run(debug=True)
