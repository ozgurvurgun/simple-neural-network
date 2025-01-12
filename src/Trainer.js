export default class Trainer {
  constructor(neuralNetwork, learningRate = 0.1) {
    this.neuralNetwork = neuralNetwork;
    this.learningRate = learningRate;
  }

  train(inputs, targets, iterations = 1000) {
    console.time("Training time");
    for (let i = 0; i < iterations; i++) {
      let totalError = 0;

      inputs.forEach((input, index) => {
        // Training step and error calculation
        const { outputActivated } = this.neuralNetwork.forward(input);
        const errors = targets[index].map(
          (target, j) => target - outputActivated[j]
        );
        totalError += errors.reduce((sum, error) => sum + Math.abs(error), 0);

        // Update weights
        this.neuralNetwork.train([input], [targets[index]], this.learningRate);
      });

      // Logging every 100 iterations
      if (i % 100 === 0 || i === iterations - 1) {
        console.log(`\nTraining iteration: ${i + 1}/${iterations}`);
        console.log(`Total error: ${totalError.toFixed(6)}`);

        // Choosing a random test input and performing an intermediate test
        const randomIndex = Math.floor(Math.random() * inputs.length);
        const testInput = inputs[randomIndex];
        const testOutput = this.neuralNetwork.forward(testInput).outputActivated;
        console.log(
          `Training test (Input: ${randomIndex}): Estimated class: ${testOutput.indexOf(
            Math.max(...testOutput)
          )}`
        );
      }
    }
    console.timeEnd("Training time");
    console.log("Training completed.");
  }
}
