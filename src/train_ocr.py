import torch
from datasets import load_dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
import os

def main():
    print("Tuning OCR model")
    
    # paths
    DATA_DIR = os.path.join("data", "ocr_data")
    OUTPUT_DIR = os.path.join("outputs", "models", "trocr-tuned-ocr")
    
    # load dataset
    dataset = load_dataset('imagefolder', data_dir=DATA_DIR, split='train')
    print(f"Loaded dataset with {len(dataset)} samples.")

    # load processor and model
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # define transformations
    def transform(examples):
        pixel_values = processor(images=examples["image"], return_tensors="pt").pixel_values
        labels = processor.tokenizer(examples["text"], padding="max_length", max_length=64).input_ids
        examples["pixel_values"] = pixel_values
        examples["labels"] = [
            [(l if l != processor.tokenizer.pad_token_id else -100) for l in label] for label in labels
        ]
        return examples

    processed_dataset = dataset.map(
        function=transform, batched=True, remove_columns=['text', 'image']
    )
    train_dataset = processed_dataset
    eval_dataset = processed_dataset

    # training arguments
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy="steps",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        fp16=torch.cuda.is_available(),
        output_dir=OUTPUT_DIR,
        logging_steps=2,
        save_steps=10,
        eval_steps=5,
        num_train_epochs=40,
        save_total_limit=1,
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.image_processor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    
    print(f"Tuned model saved to '{training_args.output_dir}'")

if __name__ == "__main__":
    main()