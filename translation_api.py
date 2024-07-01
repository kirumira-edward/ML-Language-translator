import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B")

app = FastAPI()

class TranslationRequest(BaseModel):
    text: str
    src_lang: str = "eng_Latn"
    tgt_lang: str = "swh_Latn"

class TranslationResponse(BaseModel):
    translated_text: str
    translation_time: float

def translate_text(text, src_lang="eng_Latn", tgt_lang="swh_Latn"):
    """
    Translates text from source language to target language using the NLLB model.
    """
    # Set the source language
    tokenizer.src_lang = src_lang

    # Tokenize the input text
    encoded_input = tokenizer(text, return_tensors="pt")

    # Generate translation
    generated_tokens = model.generate(
        **encoded_input,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )

    # Decode the translated tokens
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    return translated_text

def measure_translation_time(text, src_lang="eng_Latn", tgt_lang="swh_Latn"):
    """
    Measures the time taken to translate text from source language to target language.
    """
    start_time = time.time()
    translated_text = translate_text(text, src_lang, tgt_lang)
    end_time = time.time()
    
    translation_time = end_time - start_time
    return translated_text, translation_time

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    try:
        translated_text, translation_time = measure_translation_time(request.text, request.src_lang, request.tgt_lang)
        return TranslationResponse(translated_text=translated_text, translation_time=translation_time)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("translation_api:app", host="127.0.0.1", port=8000, reload=True)
