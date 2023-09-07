from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import pdfplumber

# Replace with the path to your PDF file
pdf_path = r"C:\Users\Lenovo\PycharmProjects\LawModel\COI.pdf"

# Function to extract text from a PDF and save it as a text file
def extract_text_and_save_to_text_file(pdf_path, output_text_file):
    with pdfplumber.open(pdf_path) as pdf:
        extracted_text = ""
        for page in pdf.pages:
            extracted_text += page.extract_text()

    # Save the extracted text to a text file
    with open(output_text_file, "w", encoding="utf-8") as text_file:
        text_file.write(extracted_text)


# Replace with the desired output text file name
output_text_file = "output_text.txt"

# Call the function to extract text and save it to the text file
extract_text_and_save_to_text_file(pdf_path, output_text_file)

print(f"Text extracted from {pdf_path} and saved to {output_text_file}.")

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can choose different variants like "gpt2-medium" for larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load your custom dataset
dataset_path = "output_text.txt"
train_dataset = TextDataset(tokenizer=tokenizer, file_path=dataset_path, block_size=128)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./my_custom_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Start fine-tuning
trainer.train()

# Save the fine-tuned model
trainer.save_model()
