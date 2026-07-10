import { compilePolicyValueModel, createPolicyValueModel, TfjsQuoridorPolicyModel } from "./model";
import { generateGreedyBestResponseSamples } from "./selfPlay";
import { samplesToTensors } from "./losses";
import { isMainModule } from "../util/isMain";

type DistillationMetrics = {
  prob: number;
  top1: number;
};

async function meanTargetMetrics(policyModel: TfjsQuoridorPolicyModel, samples: Awaited<ReturnType<typeof generateGreedyBestResponseSamples>>): Promise<DistillationMetrics> {
  let probabilitySum = 0;
  let top1 = 0;
  for (const sample of samples) {
    const prediction = await policyModel.predict(sample.encodedState, sample.legalMask);
    probabilitySum += prediction.policy[sample.actionIndex] ?? 0;
    let bestAction = -1;
    let bestProbability = -Infinity;
    for (let action = 0; action < prediction.policy.length; action++) {
      if (!sample.legalMask[action]) continue;
      if (prediction.policy[action] > bestProbability) {
        bestAction = action;
        bestProbability = prediction.policy[action];
      }
    }
    if (bestAction === sample.actionIndex) top1++;
  }
  const denominator = Math.max(1, samples.length);
  return {
    prob: probabilitySum / denominator,
    top1: top1 / denominator
  };
}

export async function runPolicyDistillationCheck(config: {
  games?: number;
  maxMovesPerGame?: number;
  updates?: number;
  batchSize?: number;
  learningRate?: number;
} = {}): Promise<{
  samples: number;
  before: DistillationMetrics;
  after: DistillationMetrics;
  probGain: number;
  top1Gain: number;
}> {
  const games = config.games ?? 1;
  const maxMovesPerGame = config.maxMovesPerGame ?? 80;
  const updates = config.updates ?? 16;
  const batchSize = config.batchSize ?? 32;
  const learningRate = config.learningRate ?? 0.002;
  const model = createPolicyValueModel(learningRate);
  compilePolicyValueModel(model, learningRate, 0.12);
  const policyModel = new TfjsQuoridorPolicyModel(model);
  const samples = await generateGreedyBestResponseSamples({
    games,
    maxMovesPerGame,
    teacherTimeMs: 5,
    teacherMaxDepth: 2,
    explorationRate: 0
  });
  const evalSamples = samples.slice(0, Math.min(samples.length, 48));
  const before = await meanTargetMetrics(policyModel, evalSamples);
  for (let update = 0; update < updates; update++) {
    const batch = samples.slice(0, Math.min(samples.length, batchSize * 2));
    const tensors = samplesToTensors(batch);
    await model.fit(tensors.xs, { policy: tensors.policyY, value: tensors.valueY }, {
      epochs: 1,
      batchSize: Math.min(batchSize, batch.length),
      verbose: 0
    });
    tensors.xs.dispose();
    tensors.policyY.dispose();
    tensors.valueY.dispose();
  }
  const after = await meanTargetMetrics(policyModel, evalSamples);
  model.dispose();
  return {
    samples: samples.length,
    before,
    after,
    probGain: after.prob - before.prob,
    top1Gain: after.top1 - before.top1
  };
}

if (isMainModule(import.meta.url)) {
  const result = await runPolicyDistillationCheck();
  console.log(JSON.stringify(result, null, 2));
  if (result.samples < 8 || result.probGain < 0.05) {
    process.exitCode = 1;
  }
}
