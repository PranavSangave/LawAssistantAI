# Install the Transformers library if you haven't already
# pip install transformers

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can choose different variants like "gpt2-medium" for larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Input text for generating continuation
input_text = "Hey"

# Encode the input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate text continuation
output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
