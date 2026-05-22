# Cats vs Dogs Image Classifier

This project builds, trains, and evaluates a deep learning image classifier that distinguishes cats from dogs.
It progresses from a custom convolutional neural network (CNN) built from scratch through hyperparameter tuning and transfer learning with VGG16, ResNet50, and EfficientNetB0 — culminating in a live Flask web application with automated CI/CD deployment.

## Key Features

- Custom CNN model trained on cat vs dog image data
- Hyperparameter tuning and training workflows
- Transfer learning with:
  - VGG16
  - ResNet50
  - EfficientNetB0
- Evaluation using accuracy, loss, and visual model diagnostics
- Flask web app for live image classification
- Automated CI/CD deployment pipeline

## Project Structure

- `data/` - dataset storage and preprocessing resources
- `notebooks/` - exploratory analysis and training notebooks
- `src/` - source code for model definitions, training, evaluation, and Flask application
- `README.md` - project overview and usage instructions

## Getting Started

1. Clone the repository.
2. Install the required Python dependencies.
3. Prepare the cat vs dog dataset in `data/`.
4. Run training scripts or notebooks to build and evaluate models.
5. Launch the Flask app for online inference.

## Usage

- Train a custom CNN from scratch.
- Run transfer learning experiments with VGG16, ResNet50, or EfficientNetB0.
- Use the Flask application to classify new cat and dog images.

## Notes

This project is designed to show the complete workflow for building an image classification solution, from model prototyping and evaluation to deployment.

