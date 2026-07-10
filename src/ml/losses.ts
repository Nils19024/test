import * as tf from "@tensorflow/tfjs";
import { ACTION_COUNT, BOARD_SIZE } from "../core/state";
import { MODEL_CHANNELS, MODEL_INPUT_SIZE } from "./model";
import { TrainingSample } from "./replayBuffer";

export function samplesToTensors(samples: TrainingSample[]): {
  xs: tf.Tensor4D;
  policyY: tf.Tensor2D;
  valueY: tf.Tensor2D;
} {
  const xs = new Float32Array(samples.length * MODEL_INPUT_SIZE);
  const policy = new Float32Array(samples.length * ACTION_COUNT);
  const values = new Float32Array(samples.length);
  samples.forEach((sample, i) => {
    xs.set(sample.encodedState, i * MODEL_INPUT_SIZE);
    let policySum = 0;
    if (sample.policyTarget) {
      for (let action = 0; action < ACTION_COUNT; action++) {
        const value = sample.legalMask[action] && Number.isFinite(sample.policyTarget[action]) && sample.policyTarget[action] > 0
          ? sample.policyTarget[action]
          : 0;
        policy[i * ACTION_COUNT + action] = value;
        policySum += value;
      }
    }
    if (policySum > 0) {
      for (let action = 0; action < ACTION_COUNT; action++) policy[i * ACTION_COUNT + action] /= policySum;
    } else {
      policy[i * ACTION_COUNT + sample.actionIndex] = 1;
    }
    values[i] = Number.isFinite(sample.reward) ? sample.reward ?? 0 : 0;
  });
  return {
    xs: tf.tensor4d(xs, [samples.length, BOARD_SIZE, BOARD_SIZE, MODEL_CHANNELS]),
    policyY: tf.tensor2d(policy, [samples.length, ACTION_COUNT]),
    valueY: tf.tensor2d(values, [samples.length, 1])
  };
}
