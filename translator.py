import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B")

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

# Example usage for 100 words

text ="In the heart of the bustling city, there is a serene park that offers a peaceful escape from the chaos. The park is adorned with vibrant flowers and towering trees, providing shade and beauty. A clear, sparkling pond sits at the center, where ducks and swans gracefully glide. Children laugh and play on the playground, while adults stroll along the winding paths or sit on benches reading books. The sound of birds chirping and the rustle of leaves in the breeze create a soothing atmosphere. It is a place where people come to relax, recharge, and enjoy nature's tranquility"
translated_text, translation_time = measure_translation_time(text, src_lang="eng_Latn", tgt_lang="swh_Latn")
print(f"Translated text: {translated_text}")
print(f"Time taken for translation: {translation_time:.2f} seconds")
