from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
import torch

class MistralTrainer:
    def __init__(self, config):
        self.model_name = config['model']['name']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configuration 4-bit quantization
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        # Configuration LoRA depuis le fichier config
        self.lora_config = LoraConfig(
            r=config['lora']['rank'],
            lora_alpha=config['lora']['alpha'],
            lora_dropout=config['lora']['dropout'],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config['lora']['target_modules']
        )
        
        self.load_model()

    def load_model(self):
        # Chargement du modèle quantifié
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map="auto"
        )
        
        # Application de LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        
        # Chargement du tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def train(self, dataset, output_dir="./results"):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=100,
            save_strategy="epoch",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        self.save_model(output_dir)
        
    def save_model(self, output_dir):
        self.model.save_pretrained(output_dir)
