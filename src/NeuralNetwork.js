import fs from 'fs';

export default class NeuralNetwork {
    constructor(inputNodes, hiddenNodes, outputNodes) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;

        this.weights_input_hidden = Array.from({ length: this.hiddenNodes }, () =>
            Array.from({ length: this.inputNodes }, () => Math.random() * 2 - 1)
        );

        this.weights_hidden_output = Array.from({ length: this.outputNodes }, () =>
            Array.from({ length: this.hiddenNodes }, () => Math.random() * 2 - 1)
        );

        this.bias_hidden = Array.from({ length: this.hiddenNodes }, () => Math.random() * 2 - 1);
        this.bias_output = Array.from({ length: this.outputNodes }, () => Math.random() * 2 - 1);
    }

    forward(input) {
        const hiddenInput = this.matrixMultiply(input, this.weights_input_hidden);
        const hiddenOutput = this.addBias(hiddenInput, this.bias_hidden).map(this.sigmoid);

        const outputInput = this.matrixMultiply(hiddenOutput, this.weights_hidden_output);
        const outputActivated = this.addBias(outputInput, this.bias_output).map(this.sigmoid);

        return { hiddenOutput, outputActivated };
    }

    train(inputs, targets, learningRate = 0.1) {
        inputs.forEach((input, index) => {
            const { hiddenOutput, outputActivated } = this.forward(input);

            const outputErrors = targets[index].map((target, i) => target - outputActivated[i]);

            const outputGradients = outputActivated.map(this.sigmoidDerivative);
            const outputDeltas = outputErrors.map((error, i) => error * outputGradients[i]);

            const hiddenErrors = this.matrixMultiply(outputDeltas, this.transpose(this.weights_hidden_output));
            const hiddenGradients = hiddenOutput.map(this.sigmoidDerivative);
            const hiddenDeltas = hiddenErrors.map((error, i) => error * hiddenGradients[i]);

            this.weights_hidden_output = this.updateWeights(this.weights_hidden_output, outputDeltas, hiddenOutput, learningRate);
            this.weights_input_hidden = this.updateWeights(this.weights_input_hidden, hiddenDeltas, input, learningRate);

            this.bias_output = this.bias_output.map((bias, i) => bias + outputDeltas[i] * learningRate);
            this.bias_hidden = this.bias_hidden.map((bias, i) => bias + hiddenDeltas[i] * learningRate);
        });
    }

    saveModel(filePath) {
        const model = {
            weights_input_hidden: this.weights_input_hidden,
            weights_hidden_output: this.weights_hidden_output,
            bias_hidden: this.bias_hidden,
            bias_output: this.bias_output,
        };
        fs.writeFileSync(filePath, JSON.stringify(model, null, 2));
        console.log(`Model saved: ${filePath}`);
    }

    loadModel(filePath) {
        if (fs.existsSync(filePath)) {
            const model = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
            this.weights_input_hidden = model.weights_input_hidden;
            this.weights_hidden_output = model.weights_hidden_output;
            this.bias_hidden = model.bias_hidden;
            this.bias_output = model.bias_output;
        } else {
            throw new Error(`Model file not found: ${filePath}`);
        }
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    sigmoidDerivative(x) {
        return x * (1 - x);
    }

    matrixMultiply(input, weights) {
        return weights.map(row =>
            row.reduce((sum, weight, i) => sum + input[i] * weight, 0)
        );
    }

    addBias(values, bias) {
        return values.map((value, i) => value + bias[i]);
    }

    transpose(matrix) {
        return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
    }

    updateWeights(weights, deltas, activations, learningRate) {
        return weights.map((row, i) =>
            row.map((weight, j) => weight + deltas[i] * activations[j] * learningRate)
        );
    }
}
