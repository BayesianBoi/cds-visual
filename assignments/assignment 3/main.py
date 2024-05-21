import os
from src.data_loading import load_data, create_generators
from src.train_model import train_model
from src.evaluate_model import evaluate_model

# Define the data folder
data_folder = "./in/Tobacco3482"

# Ensure the /out folder exists
if not os.path.exists("out"):
    os.makedirs("out")

# Load data
train_files, test_files, train_labels, test_labels, categories = load_data(data_folder)

# Create data generators
train_generator, test_generator = create_generators(train_files, test_files, train_labels, test_labels)

# Training the model
model, history = train_model(train_generator, test_generator, categories)

# Evaluate the model
evaluate_model(model, test_generator, history)
