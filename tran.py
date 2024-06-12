import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Translator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B")

    def nllb_translate_text(self, text, src_lang, tgt_lang):
        """
        Translates the given text from source language to target language using NLLB.
        """
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Generate the translated text
        translated_tokens = self.model.generate(**inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang])
        
        # Decode the tokens to get the translated text
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        
        return translated_text

    def translate_text_custom(self, text, src_lang, tgt_lang):
        """
        Translates the given text from source language to target language using NLLB.
        """
        return self.nllb_translate_text(text, src_lang, tgt_lang)

    def translate_local_to_english(self, message, language):
        if language == 'lug_Latn':
            return self.translate_text_custom(
                message,
                src_lang='lug_Latn',
                tgt_lang='eng_Latn'
            )
        elif language == 'swh_Latn':
            return self.translate_text_custom(
                message,
                src_lang='swh_Latn',
                tgt_lang='eng_Latn'
            )
        return message

    def translate_english_to_local(self, message, language):
        if language == 'lug_Latn':
            return self.translate_text_custom(
                message,
                src_lang='eng_Latn',
                tgt_lang='lug_Latn'
            )
        elif language == 'swh_Latn':
            return self.translate_text_custom(
                message,
                src_lang='eng_Latn',
                tgt_lang='swh_Latn'
            )
        return message

# Example usage:
if __name__ == "__main__":
    translator = Translator()
    
    # Get user input
    source_text = input("Enter the text to translate: ")
    target_language = input("Enter the target language (lug_Latn for Luganda, swa_Latn for Swahili, eng_Latn for English): ")
    
    # Determine source language
    source_language = input("Enter the source language (lug_Latn for Luganda, swa_Latn for Swahili, eng_Latn for English): ")
    
    # Translate the text
    translated_text = translator.translate_text_custom(source_text, source_language, target_language)
    print(f"Translated text: {translated_text}")