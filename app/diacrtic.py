from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.generation.utils import GenerationMixin

def diacritic_text(text: str):
    tokenizer = AutoTokenizer.from_pretrained("glonor/byt5-arabic-diacritization")
    model = AutoModelForSeq2SeqLM.from_pretrained("glonor/byt5-arabic-diacritization")

    inputs = tokenizer(text, return_tensors="pt")

    outputs = model.generate(**inputs, max_new_tokens=128)
    diacritized = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return diacritized
