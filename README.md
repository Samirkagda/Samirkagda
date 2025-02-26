
```python
from textblob import TextBlob
from transformers import pipeline

def correct_and_rephrase(text):
    # Grammar correction using TextBlob
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    
    # Rephrasing using Hugging Face transformers
    paraphraser = pipeline("text2text-generation", model="t5-base")
    rephrased_text = paraphraser(corrected_text, max_length=50, do_sample=False)
    
    return rephrased_text[0]['generated_text']

# Input text
input_text = "Paste your text here with grammar mistakes."

# Correct and rephrase
output_text = correct_and_rephrase(input_text)
print("Corrected and Rephrased Text: ", output_text)
```
