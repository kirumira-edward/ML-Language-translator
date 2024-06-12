import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B")

def translate_text(text, src_lang="swh_Latn", tgt_lang="eng_Latn"):
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

# Example usage
text = "Jina lako ni nani"
translated_text = translate_text(text, src_lang="swh_Latn", tgt_lang="eng_Latn")
print(f"Translated text: {translated_text}")
