def log_message(message):
    """Logs a message to the console."""
    print(f"[LOG] {message}")

def calculate_accuracy(predictions, labels):
    """Calculates the accuracy of predictions."""
    correct = (predictions == labels).sum()
    total = labels.size
    return correct / total

def save_model(model, filepath):
    """Saves the trained model to the specified filepath."""
    model.save(filepath)

def load_model(filepath):
    """Loads a model from the specified filepath."""
    from tensorflow.keras.models import load_model
    return load_model(filepath)