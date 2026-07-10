import * as fs from "node:fs/promises";
import * as tf from "@tensorflow/tfjs";
import { runPrefixStress } from "../benchmark/prefixStress";
import { runStrengthGate } from "../benchmark/strengthGate";
import { GreedyShortestPathAgent } from "../ai/greedyAgent";
import { encodeStateForModel } from "../core/encode";
import { applyKnownLegalMove } from "../core/moves";
import { getModelLegalActionMask } from "../core/modelPerspective";
import { createInitialState, getWinner } from "../core/state";
import { isMainModule } from "../util/isMain";
import { loadModelFromDir, saveModelForBrowser } from "./exportModel";
import { samplesToTensors } from "./losses";
import { compilePolicyValueModel, TfjsQuoridorPolicyModel } from "./model";
import { generatePrefixStressTeacherSamples } from "./prefixStressSamples";
import { TrainingSample } from "./replayBuffer";
import { initializeTrainingBackend, TrainingBackendPreference } from "./tfBackend";

type PrefixFineTuneConfig = {
  sourceDir: string;
  candidateDir: string;
  games: number;
  rolloutPlies: number;
  teacherTimeMs: number;
  teacherMaxDepth: number;
  updates: number;
  batchSize: number;
  learningRate: number;
  anchorGames: number;
  anchorMaxMovesPerGame: number;
  focusGames: number[];
  focusRepeats: number;
  focusTeacherMode: "search" | "best-response";
  focusBestResponseMix: number;
  backend: TrainingBackendPreference;
  skipBenchmarks: boolean;
};

const defaultConfig: PrefixFineTuneConfig = {
  sourceDir: "public/models/latest",
  candidateDir: "models/candidates/prefix-stress",
  games: 8,
  rolloutPlies: 5,
  teacherTimeMs: 120,
  teacherMaxDepth: 5,
  updates: 24,
  batchSize: 64,
  learningRate: 0.00005,
  anchorGames: 4,
  anchorMaxMovesPerGame: 80,
  focusGames: [],
  focusRepeats: 0,
  focusTeacherMode: "search",
  focusBestResponseMix: 0,
  backend: "auto",
  skipBenchmarks: false
};

function parseArgs(): PrefixFineTuneConfig {
  const parsed: Record<string, string> = {};
  for (const arg of process.argv.slice(2)) {
    if (!arg.startsWith("--")) continue;
    const [key, value] = arg.slice(2).split("=", 2);
    parsed[key] = value ?? "true";
  }
  return {
    sourceDir: parsed.sourceDir ?? defaultConfig.sourceDir,
    candidateDir: parsed.candidateDir ?? defaultConfig.candidateDir,
    games: parsed.games ? Number(parsed.games) : defaultConfig.games,
    rolloutPlies: parsed.rolloutPlies ? Number(parsed.rolloutPlies) : defaultConfig.rolloutPlies,
    teacherTimeMs: parsed.teacherTimeMs ? Number(parsed.teacherTimeMs) : defaultConfig.teacherTimeMs,
    teacherMaxDepth: parsed.teacherMaxDepth ? Number(parsed.teacherMaxDepth) : defaultConfig.teacherMaxDepth,
    updates: parsed.updates ? Number(parsed.updates) : defaultConfig.updates,
    batchSize: parsed.batchSize ? Number(parsed.batchSize) : defaultConfig.batchSize,
    learningRate: parsed.learningRate ? Number(parsed.learningRate) : defaultConfig.learningRate,
    anchorGames: parsed.anchorGames ? Number(parsed.anchorGames) : defaultConfig.anchorGames,
    anchorMaxMovesPerGame: parsed.anchorMaxMovesPerGame ? Number(parsed.anchorMaxMovesPerGame) : defaultConfig.anchorMaxMovesPerGame,
    focusGames: parsed.focusGames ? parsed.focusGames.split(",").map((value) => Number(value.trim())).filter(Number.isFinite) : defaultConfig.focusGames,
    focusRepeats: parsed.focusRepeats ? Number(parsed.focusRepeats) : defaultConfig.focusRepeats,
    focusTeacherMode: parsed.focusTeacherMode === "best-response" ? "best-response" : defaultConfig.focusTeacherMode,
    focusBestResponseMix: parsed.focusBestResponseMix ? Number(parsed.focusBestResponseMix) : defaultConfig.focusBestResponseMix,
    backend: (parsed.backend as TrainingBackendPreference | undefined) ?? defaultConfig.backend,
    skipBenchmarks: parsed.skipBenchmarks === "true"
  };
}

function sampleBatch(samples: TrainingSample[], batchSize: number): TrainingSample[] {
  const batch: TrainingSample[] = [];
  for (let index = 0; index < Math.min(batchSize, samples.length); index++) {
    batch.push(samples[Math.floor(Math.random() * samples.length)]);
  }
  return batch;
}

async function generatePolicyAnchorSamples(config: {
  model: TfjsQuoridorPolicyModel;
  games: number;
  maxMovesPerGame: number;
}): Promise<TrainingSample[]> {
  const samples: TrainingSample[] = [];
  for (let game = 0; game < config.games; game++) {
    let state = createInitialState();
    for (let ply = 0; ply < config.maxMovesPerGame && getWinner(state) === null; ply++) {
      const legalMask = getModelLegalActionMask(state);
      const encodedState = encodeStateForModel(state);
      const prediction = await config.model.predict(encodedState, legalMask);
      let actionIndex = 0;
      let bestPolicy = -Infinity;
      const policyTarget = new Float32Array(prediction.policy);
      for (let action = 0; action < policyTarget.length; action++) {
        if (!legalMask[action]) policyTarget[action] = 0;
        if (legalMask[action] && policyTarget[action] > bestPolicy) {
          bestPolicy = policyTarget[action];
          actionIndex = action;
        }
      }
      samples.push({
        encodedState,
        legalMask,
        actionIndex,
        policyTarget,
        reward: Math.max(-1, Math.min(1, prediction.value)),
        priority: 1,
        source: "search",
        terminal: false,
        wallMove: false
      });

      const move = await GreedyShortestPathAgent.selectMove(state);
      state = applyKnownLegalMove(state, move);
      if (ply % 16 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }
  return samples;
}

export async function runPrefixStressFineTune(config: Partial<PrefixFineTuneConfig> = {}) {
  const cfg = { ...defaultConfig, ...config };
  const backend = await initializeTrainingBackend(cfg.backend, (message) => console.log(message));
  const model = await loadModelFromDir(cfg.sourceDir, cfg.learningRate);
  compilePolicyValueModel(model, cfg.learningRate, 0.25);
  const policyModel = new TfjsQuoridorPolicyModel(model);
  const prefixSamples = await generatePrefixStressTeacherSamples({
    games: cfg.games,
    rolloutPlies: cfg.rolloutPlies,
    teacherTimeMs: cfg.teacherTimeMs,
    teacherMaxDepth: cfg.teacherMaxDepth
  });
  const focusSamples = cfg.focusGames.length > 0 && cfg.focusRepeats > 0
    ? await generatePrefixStressTeacherSamples({
      games: 0,
      gamesList: Array.from({ length: cfg.focusRepeats }, () => cfg.focusGames).flat(),
      rolloutPlies: cfg.rolloutPlies,
      teacherTimeMs: cfg.teacherTimeMs,
      teacherMaxDepth: cfg.teacherMaxDepth,
      teacherMode: cfg.focusTeacherMode,
      bestResponseMix: cfg.focusBestResponseMix
    })
    : [];
  const anchorSamples = await generatePolicyAnchorSamples({
    model: policyModel,
    games: cfg.anchorGames,
    maxMovesPerGame: cfg.anchorMaxMovesPerGame
  });
  const samples = [...prefixSamples, ...focusSamples, ...anchorSamples];
  if (samples.length === 0) throw new Error("No prefix stress samples generated");

  const losses: number[] = [];
  for (let update = 0; update < cfg.updates; update++) {
    const batch = sampleBatch(samples, cfg.batchSize);
    const tensors = samplesToTensors(batch);
    const history = await model.fit(tensors.xs, { policy: tensors.policyY, value: tensors.valueY }, {
      epochs: 1,
      batchSize: Math.min(cfg.batchSize, batch.length),
      verbose: 0
    });
    tensors.xs.dispose();
    tensors.policyY.dispose();
    tensors.valueY.dispose();
    const loss = Number(history.history.loss?.at(-1) ?? 0);
    if (Number.isFinite(loss)) losses.push(loss);
    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  await fs.rm(cfg.candidateDir, { recursive: true, force: true });
  await saveModelForBrowser(model, cfg.candidateDir);
  const result: {
    candidateDir: string;
    samples: number;
    prefixSamples: number;
    focusSamples: number;
    anchorSamples: number;
    updates: number;
    backend: unknown;
    firstLoss: number | null;
    lastLoss: number | null;
    prefixStress?: unknown;
    strengthGate?: unknown;
  } = {
    candidateDir: cfg.candidateDir,
    samples: samples.length,
    prefixSamples: prefixSamples.length,
    focusSamples: focusSamples.length,
    anchorSamples: anchorSamples.length,
    updates: cfg.updates,
    backend,
    firstLoss: losses[0] ?? null,
    lastLoss: losses.at(-1) ?? null
  };
  if (!cfg.skipBenchmarks) {
    result.prefixStress = await runPrefixStress(cfg.candidateDir);
    result.strengthGate = await runStrengthGate(cfg.candidateDir);
  }
  return result;
}

async function main(): Promise<void> {
  const result = await runPrefixStressFineTune(parseArgs());
  console.log(JSON.stringify(result, null, 2));
}

if (isMainModule(import.meta.url)) {
  void main();
}
