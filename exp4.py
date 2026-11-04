from transformers import MarianMTModel, MarianTokenizer

# ---------------------------------------
# 1. Select the translation model
# ---------------------------------------

model_name = "Helsinki-NLP/opus-mt-en-fr"


# ---------------------------------------
# 2. Load tokenizer and model
# ---------------------------------------

print("Loading translation model...")

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

print("Model loaded successfully!")


# ---------------------------------------
# 3. Get input from the user
# ---------------------------------------

text = input("\nEnter an English sentence: ")


# ---------------------------------------
# 4. Tokenize the input
# ---------------------------------------

encoded = tokenizer(
    text,
    return_tensors="pt"
)


# ---------------------------------------
# 5. Generate French translation
# ---------------------------------------

translated_tokens = model.generate(
    **encoded,
    max_length=100
)


# ---------------------------------------
# 6. Decode the generated tokens
# ---------------------------------------

translation = tokenizer.decode(
    translated_tokens[0],
    skip_special_tokens=True
)


# ---------------------------------------
# 7. Display the result
# ---------------------------------------

print("\n" + "=" * 40)
print("MACHINE TRANSLATION")
print("=" * 40)

print("English :", text)
print("French  :", translation)

print("=" * 40)
