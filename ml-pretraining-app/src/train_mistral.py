import os
import yaml
from loader import DataLoader
from models.mistral_trainer import MistralTrainer
from datasets import Dataset

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    data_loader = DataLoader(config['model']['name'])
    
    # Charger et préparer les données
    training_examples = data_loader.load_data()
    dataset = Dataset.from_dict({
        'text': [
            f"[INST] {ex['instruction']}\n\n{ex['input']} [/INST] {ex['output']}"
            for ex in training_examples
        ]
    })
    
    # Initialiser et entraîner avec LoRA
    trainer = MistralTrainer(config)
    trainer.train(dataset, output_dir=config['training']['output_dir'])

if __name__ == "__main__":
    main()
