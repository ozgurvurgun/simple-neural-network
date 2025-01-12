import Trainer from '../src/Trainer.js';
import NeuralNetwork from '../src/NeuralNetwork.js';
import path from "path";

// Dataset
const inputs = [
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], // 0
  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], // 1
  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], // 2
  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], // 3
  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], // 4
  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], // 5
  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], // 6
  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], // 7
  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], // 8
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], // 9
];
const targets = inputs;

// Neural network and trainer
const nn = new NeuralNetwork(10, 16, 10);
const trainer = new Trainer(nn, 0.1);

// Start training
trainer.train(inputs, targets, 10000);

const modelPath = path.resolve("model", "model.json");
nn.saveModel(modelPath);
