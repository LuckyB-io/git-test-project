from transformers import pipeline

# Load a small text-generation pipeline (uses distilgpt2 by default if no model specified)
generator = pipeline("text-generation", model="distilgpt2")

# Generate text
prompt = "After surviving load shedding and two cups of rooibos, a South African dev finally got transformers running locally. He threw his hands in the air and yelled:"

result = generator(
    prompt,
    max_new_tokens=80,          # Cleaner: controls how many NEW tokens to generate
    num_return_sequences=1,
    do_sample=True,             # Random sampling for creativity
    temperature=0.9,            # Higher = more creative/wild (0.7–1.0 is fun)
    top_p=0.95,                 # Nucleus sampling — keeps it coherent
    truncation=True
)

print("\nGenerated text:")
print(result[0]['generated_text'])