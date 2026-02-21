from deep_translator import GoogleTranslator

def translate(text,lang):
    try:
        return GoogleTranslator(source='auto', target=lang).translate(text)
    except:
        return text
