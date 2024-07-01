import time
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B")

@st.cache_data
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

def measure_translation_time(text, src_lang="eng_Latn", tgt_lang="swh_Latn"):
    """
    Measures the time taken to translate text from source language to target language.
    """
    start_time = time.time()
    translated_text = translate_text(text, src_lang, tgt_lang)
    end_time = time.time()
    translation_time = end_time - start_time
    return translated_text, translation_time

# Streamlit GUI
def main():
    st.title("Language Translation Time Analysis")
    st.write("Translate text from English to Swahili and analyze translation time.")

    text = st.text_area("Enter the text to be translated:", value="In the heart of the city, there is a small park...")
    num_words = len(text.split())

    if st.button("Translate"):
        translated_text, translation_time = measure_translation_time(text)
        st.write(f"**Translated text:** {translated_text}")
        st.write(f"**Time taken for translation:** {translation_time:.2f} seconds")

    # Plot graph
    st.subheader("Translation Time vs Number of Words")
    st.write("Analyzing the relationship between the number of words and translation time.")

    num_words_list = list(range(1, num_words + 1))
    translation_times = []

    progress_bar = st.progress(0)
    for i, num_words in enumerate(num_words_list):
        dummy_text = " ".join(["word"] * num_words)
        _, translation_time = measure_translation_time(dummy_text)
        translation_times.append(translation_time)
        progress_bar.progress((i + 1) / len(num_words_list))

    import pandas as pd

    data = pd.DataFrame({
        'Number of Words': num_words_list,
        'Translation Time (s)': translation_times
    })

    st.line_chart(data.set_index('Number of Words'))

if __name__ == "__main__":
    main()
