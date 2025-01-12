import NeuralNetwork from '../src/NeuralNetwork.js';
import path from 'path';
import chalk from 'chalk';

// Load the neural network model
function loadModel() {
    const nn = new NeuralNetwork(10, 16, 10);
    const modelPath = path.resolve('model', 'model.json');
    nn.loadModel(modelPath);
    console.log(chalk.green(`\nModel loaded: ${modelPath}`));
    return nn;
}

// Log class probabilities
function logClassProbabilities(output) {
    console.log("\nClass probabilities:");
    output.forEach((probability, index) => {
        console.log(`Class ${index}: ${chalk.yellow(probability.toFixed(6))}`);
    });
}

// Log prediction result
function logPredictionResult(predictedClass, predictedProbability) {
    console.log("\nPrediction result:");
    console.log(chalk.blue(`Predicted class: ${predictedClass}`));
    console.log(chalk.blue(`Prediction probability: ${predictedProbability.toFixed(6)}`));

    // Check prediction confidence
    if (predictedProbability < 0.8) {
        console.log(chalk.red("\n⚠️ Warning: The prediction may not be reliable. The probability is low."));
    }
}

// Run inference for a given input
function runInference(nn, input) {
    console.log("\nEvaluated value:");
    console.log(`Input: [${input.join(", ")}] (Expected class: ${input.indexOf(1)})`);

    const output = nn.forward(input).outputActivated;
    logClassProbabilities(output);

    const predictedClass = output.indexOf(Math.max(...output));
    const predictedProbability = output[predictedClass];
    logPredictionResult(predictedClass, predictedProbability);
}

// Input vectors for digits 0–9
const inputVectors = {
    0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
};

// Load the model
const nn = loadModel();
runInference(nn, inputVectors[9]);
