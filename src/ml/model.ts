import * as tf from "@tensorflow/tfjs";
import { ACTION_COUNT, BOARD_SIZE } from "../core/state";

export const MODEL_CHANNELS = 12;
export const MODEL_INPUT_SIZE = BOARD_SIZE * BOARD_SIZE * MODEL_CHANNELS;

export interface QuoridorPolicyModel {
  predict(
    encodedState: Float32Array,
    legalMask: Uint8Array
  ): Promise<{
    policy: Float32Array;
    value: number;
  }>;
  predictBatch?(
    inputs: Array<{ encodedState: Float32Array; legalMask: Uint8Array }>
  ): Promise<Array<{
    policy: Float32Array;
    value: number;
  }>>;
}

export function compilePolicyValueModel(model: tf.LayersModel, learningRate = 0.0005, valueLossWeight = 0.5): tf.LayersModel {
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: {
      policy: "categoricalCrossentropy",
      value: "meanSquaredError"
    },
    lossWeights: {
      policy: 1,
      value: valueLossWeight
    }
  } as tf.ModelCompileArgs);
  return model;
}

export function createPolicyValueModel(learningRate = 0.0005): tf.LayersModel {
  const input = tf.input({ shape: [BOARD_SIZE, BOARD_SIZE, MODEL_CHANNELS], name: "state" });
  const conv1 = tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    padding: "same",
    activation: "relu"
  }).apply(input) as tf.SymbolicTensor;
  const conv2 = tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    padding: "same",
    activation: "relu"
  }).apply(conv1) as tf.SymbolicTensor;
  const flat = tf.layers.flatten().apply(conv2) as tf.SymbolicTensor;
  const dense = tf.layers.dense({ units: 128, activation: "relu" }).apply(flat) as tf.SymbolicTensor;
  const policy = tf.layers.dense({ units: ACTION_COUNT, activation: "softmax", name: "policy" }).apply(dense) as tf.SymbolicTensor;
  const value = tf.layers.dense({ units: 1, activation: "tanh", name: "value" }).apply(dense) as tf.SymbolicTensor;
  const model = tf.model({ inputs: input, outputs: [policy, value] });
  return compilePolicyValueModel(model, learningRate);
}

export function maskPolicyLogits(logits: tf.Tensor, legalMask: Uint8Array): tf.Tensor {
  return tf.tidy(() => {
    const mask = tf.tensor1d(Array.from(legalMask), "float32");
    const penalty = tf.mul(tf.sub(1, mask), tf.scalar(-1e9));
    return tf.add(logits, penalty);
  });
}

function maskedProbabilities(policy: Float32Array, legalMask: Uint8Array, temperature = 1): Float64Array {
  const result = new Float64Array(policy.length);
  let sum = 0;
  const t = Math.max(0.05, temperature);
  for (let i = 0; i < policy.length; i++) {
    if (!legalMask[i]) continue;
    const value = Math.pow(Math.max(policy[i], 1e-12), 1 / t);
    result[i] = value;
    sum += value;
  }
  if (sum === 0) {
    for (let i = 0; i < legalMask.length; i++) {
      if (legalMask[i]) {
        result[i] = 1;
        sum++;
      }
    }
  }
  for (let i = 0; i < result.length; i++) result[i] /= sum || 1;
  return result;
}

export function chooseBestLegalAction(policy: Float32Array, legalMask: Uint8Array): number {
  let best = -1;
  let bestValue = -Infinity;
  for (let i = 0; i < policy.length; i++) {
    if (!legalMask[i]) continue;
    const value = Number.isFinite(policy[i]) ? policy[i] : -Infinity;
    if (best < 0 || value > bestValue) {
      bestValue = value;
      best = i;
    }
  }
  if (best < 0) throw new Error("No legal action in mask");
  return best;
}

export function chooseSampledLegalAction(policy: Float32Array, legalMask: Uint8Array, temperature: number): number {
  const probabilities = maskedProbabilities(policy, legalMask, temperature);
  let pick = Math.random();
  for (let i = 0; i < probabilities.length; i++) {
    pick -= probabilities[i];
    if (pick <= 0 && legalMask[i]) return i;
  }
  return chooseBestLegalAction(policy, legalMask);
}

export class TfjsQuoridorPolicyModel implements QuoridorPolicyModel {
  constructor(private model: tf.LayersModel) {}

  async predict(encodedState: Float32Array, legalMask: Uint8Array): Promise<{ policy: Float32Array; value: number }> {
    return (await this.predictBatch([{ encodedState, legalMask }]))[0];
  }

  async predictBatch(inputs: Array<{ encodedState: Float32Array; legalMask: Uint8Array }>): Promise<Array<{ policy: Float32Array; value: number }>> {
    if (inputs.length === 0) return [];
    const batch = inputs.length;
    const flat = new Float32Array(batch * MODEL_INPUT_SIZE);
    inputs.forEach((input, index) => flat.set(input.encodedState, index * MODEL_INPUT_SIZE));
    const [policyTensor, valueTensor] = tf.tidy(() => {
      const input = tf.tensor4d(flat, [batch, BOARD_SIZE, BOARD_SIZE, MODEL_CHANNELS]);
      return this.model.predict(input) as tf.Tensor[];
    });
    const rawPolicy = await policyTensor.data();
    const rawValue = await valueTensor.data();
    policyTensor.dispose();
    valueTensor.dispose();
    return inputs.map((input, batchIndex) => {
      const policy = new Float32Array(ACTION_COUNT);
      let sum = 0;
      const offset = batchIndex * ACTION_COUNT;
      for (let i = 0; i < ACTION_COUNT; i++) {
        const probability = rawPolicy[offset + i];
        if (input.legalMask[i] && Number.isFinite(probability)) {
          policy[i] = probability;
          sum += policy[i];
        }
      }
      if (sum <= 0) {
        for (let i = 0; i < input.legalMask.length; i++) if (input.legalMask[i]) policy[i] = 1;
        sum = policy.reduce((a, b) => a + b, 0);
      }
      for (let i = 0; i < policy.length; i++) policy[i] /= sum;
      return { policy, value: rawValue[batchIndex] };
    });
  }
}
