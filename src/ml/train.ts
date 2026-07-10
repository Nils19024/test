import * as tf from "@tensorflow/tfjs";
import { execFile } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { promisify } from "node:util";
import { alphaBetaAgent } from "../ai/alphabeta";
import { GreedyShortestPathAgent } from "../ai/greedyAgent";
import { RandomAgent } from "../ai/randomAgent";
import { runMatchup } from "../benchmark/benchmark";
import { encodeStateForModel } from "../core/encode";
import { applyMove, getLegalMoves } from "../core/moves";
import { QuoridorState, createInitialState, getWinner } from "../core/state";
import { getModelLegalActionMask, moveToModelActionIndex } from "../core/modelPerspective";
import { moveToActionIndex } from "../core/actions";
import { chooseBestLegalAction, compilePolicyValueModel, createPolicyValueModel, TfjsQuoridorPolicyModel } from "./model";
import { ReplayBuffer, TrainingSample } from "./replayBuffer";
import { samplesToTensors } from "./losses";
import { generateGreedyBestResponseSamples, generateGreedyImitationSamples, runSearchImprovedSelfPlayGame, runSelfPlayGame, selectBestMoveAgainstGreedy } from "./selfPlay";
import { AlphaZeroSelfPlayStats, runAlphaZeroSelfPlayGame, runBatchedAlphaZeroSelfPlayGames } from "./alphaZeroSelfPlay";
import { generateTeacherSamples } from "./supervised";
import { saveModelForBrowser } from "./exportModel";
import { ModelAgent } from "../ai/modelAgent";
import { isMainModule } from "../util/isMain";
import { TrainingBackendInfo, TrainingBackendPreference, initializeTrainingBackend } from "./tfBackend";
import { PersistentReplayStore } from "./persistentReplay";
import { generatePrefixStressTeacherSamples } from "./prefixStressSamples";

const execFileAsync = promisify(execFile);

const MAX_GPU_UTILIZATION_TARGET = 0.7;
const DEFAULT_GPU_UTILIZATION_TARGET = 0.45;
const GPU_POLL_INTERVAL_MS = 750;
const GPU_MAX_WAIT_MS = 15000;

export type TrainingConfig = {
  mode: "supervised" | "selfplay" | "mixed" | "alphazero";
  games: number;
  maxMovesPerGame: number;
  batchSize: number;
  epochs: number;
  learningRate: number;
  teacherTimeMs: number;
  teacherMaxDepth: number;
  saveEveryGames: number;
  replayEveryGames: number;
  evaluateEveryGames: number;
  temperature: number;
  backend: TrainingBackendPreference;
  rlSearchTopK: number;
  rlSearchTimeMs: number;
  rlSearchMaxDepth: number;
  persistentReplay: boolean;
  persistentReplayMaxLoad: number;
  curriculum: "auto" | "off";
  teacherRandomPrefixMax: number;
  teacherValueScale: number;
  explorationNoise: number;
  alphaZeroSimulations: number;
  alphaZeroCpuct: number;
  alphaZeroDirichletAlpha: number;
  alphaZeroDirichletFraction: number;
  alphaZeroTemperatureMoves: number;
  alphaZeroMinTemperature: number;
  alphaZeroPolicySmoothing: number;
  alphaZeroWarmupGames: number;
  alphaZeroFastSimulations: number;
  alphaZeroFullSimulationStart: number;
  alphaZeroHeuristicPriorMix: number;
  alphaZeroDrawValueWeight: number;
  alphaZeroParallelGames: number;
  alphaZeroUpdatesPerGame: number;
  resumeFromCheckpoint: boolean;
  curriculumGameOffset: number;
  gpuUtilizationTarget: number;
  gpuCooldownMs: number;
  bootstrapMaxUpdates: number;
  bootstrapGames: number;
  bootstrapBestResponseGames: number;
  greedyBestResponseGames: number;
  prefixStressGames: number;
  prefixStressRolloutPlies: number;
};

export const defaultTrainingConfig: TrainingConfig = {
  mode: "alphazero",
  games: 1000,
  maxMovesPerGame: 80,
  batchSize: 128,
  epochs: 1,
  learningRate: 0.0005,
  teacherTimeMs: 35,
  teacherMaxDepth: 4,
  saveEveryGames: 100,
  replayEveryGames: 10,
  evaluateEveryGames: 100,
  temperature: 1.0,
  backend: "auto",
  rlSearchTopK: 6,
  rlSearchTimeMs: 8,
  rlSearchMaxDepth: 2,
  persistentReplay: false,
  persistentReplayMaxLoad: 20000,
  curriculum: "auto",
  teacherRandomPrefixMax: 12,
  teacherValueScale: 80,
  explorationNoise: 0.18,
  alphaZeroSimulations: 16,
  alphaZeroCpuct: 1.5,
  alphaZeroDirichletAlpha: 0.3,
  alphaZeroDirichletFraction: 0.35,
  alphaZeroTemperatureMoves: 80,
  alphaZeroMinTemperature: 0.5,
  alphaZeroPolicySmoothing: 0.06,
  alphaZeroWarmupGames: 0,
  alphaZeroFastSimulations: 4,
  alphaZeroFullSimulationStart: 5000,
  alphaZeroHeuristicPriorMix: 0.55,
  alphaZeroDrawValueWeight: 0.7,
  alphaZeroParallelGames: 2,
  alphaZeroUpdatesPerGame: 1,
  resumeFromCheckpoint: true,
  curriculumGameOffset: 0,
  gpuUtilizationTarget: DEFAULT_GPU_UTILIZATION_TARGET,
  gpuCooldownMs: 900,
  bootstrapMaxUpdates: 32,
  bootstrapGames: 2,
  bootstrapBestResponseGames: 5,
  greedyBestResponseGames: 2,
  prefixStressGames: 4,
  prefixStressRolloutPlies: 4
};

export type TrainingProgress = {
  game: number;
  samples: number;
  loss: number | null;
  policyLoss: number | null;
  valueLoss: number | null;
  rewardMean: number | null;
  rewardMin: number | null;
  rewardMax: number | null;
  policyEntropyMean: number | null;
  policyTopProbMean: number | null;
  actionDiversity: number | null;
  avgSamplesPerGame: number | null;
  terminalRate: number | null;
  loopRate: number | null;
  wallMoveRate: number | null;
  forwardProgressRate: number | null;
  pathAdvantageMean: number | null;
  avgGameLength: number | null;
  generationSource: string | null;
  modelPath: string | null;
  backend: TrainingBackendInfo;
  evaluation?: unknown;
};

type GenerationDiagnostics = {
  terminalRate: number | null;
  loopRate: number | null;
  wallMoveRate: number | null;
  forwardProgressRate: number | null;
  pathAdvantageMean: number | null;
  avgGameLength: number | null;
  generationSource: string | null;
};

type GreedyPolicyAgreement = {
  states: number;
  agreement: number;
  top3Agreement: number;
  greedyProbMean: number;
};

type GreedyBestResponseAgreement = {
  states: number;
  agreement: number;
  top3Agreement: number;
  teacherProbMean: number;
};

function mergeTrainingConfig(config: Partial<TrainingConfig>): TrainingConfig {
  const cleaned = Object.fromEntries(Object.entries(config).filter(([, value]) => value !== undefined)) as Partial<TrainingConfig>;
  const merged = { ...defaultTrainingConfig, ...cleaned };
  return {
    ...merged,
    gpuUtilizationTarget: Math.min(MAX_GPU_UTILIZATION_TARGET, Math.max(0.2, merged.gpuUtilizationTarget)),
    gpuCooldownMs: Math.max(0, merged.gpuCooldownMs)
  };
}

function snapshotModel(model: tf.LayersModel, learningRate: number): tf.LayersModel {
  const snapshot = createPolicyValueModel(learningRate);
  const clonedWeights = model.getWeights().map((weight) => weight.clone());
  snapshot.setWeights(clonedWeights);
  return snapshot;
}

async function createOrResumePolicyValueModel(cfg: TrainingConfig, onLog?: (message: string) => void): Promise<tf.LayersModel> {
  const checkpointPath = "models/checkpoints/latest";
  const modelJsonPath = path.join(checkpointPath, "model.json");
  if (!cfg.resumeFromCheckpoint || !fs.existsSync(modelJsonPath)) {
    if (cfg.resumeFromCheckpoint) onLog?.("No checkpoint found, starting with a new model");
    return createPolicyValueModel(cfg.learningRate);
  }
  try {
    const loaded = await tf.loadLayersModel(`file://${path.resolve(modelJsonPath)}`);
    onLog?.(`Resumed model from ${checkpointPath}`);
    return compilePolicyValueModel(loaded, cfg.learningRate);
  } catch (error) {
    onLog?.(`Could not resume checkpoint, starting with a new model: ${error instanceof Error ? error.message : String(error)}`);
    return createPolicyValueModel(cfg.learningRate);
  }
}

function summarizeRewards(samples: TrainingSample[]): { mean: number | null; min: number | null; max: number | null } {
  const rewards = samples
    .map((sample) => sample.reward)
    .filter((reward): reward is number => typeof reward === "number" && Number.isFinite(reward));
  if (rewards.length === 0) return { mean: null, min: null, max: null };
  const sum = rewards.reduce((total, reward) => total + reward, 0);
  return {
    mean: sum / rewards.length,
    min: Math.min(...rewards),
    max: Math.max(...rewards)
  };
}

function summarizePolicyTargets(samples: TrainingSample[], generatedGames: number): {
  entropyMean: number | null;
  topProbMean: number | null;
  actionDiversity: number | null;
  avgSamplesPerGame: number | null;
} {
  if (samples.length === 0) {
    return { entropyMean: null, topProbMean: null, actionDiversity: null, avgSamplesPerGame: null };
  }
  let entropySum = 0;
  let topProbSum = 0;
  let policyCount = 0;
  const actions = new Set<number>();
  for (const sample of samples) {
    actions.add(sample.actionIndex);
    const policy = sample.policyTarget;
    if (!policy) continue;
    let entropy = 0;
    let topProb = 0;
    for (let action = 0; action < policy.length; action++) {
      const probability = sample.legalMask[action] && Number.isFinite(policy[action]) ? policy[action] : 0;
      if (probability <= 0) continue;
      entropy -= probability * Math.log(probability);
      topProb = Math.max(topProb, probability);
    }
    entropySum += entropy;
    topProbSum += topProb;
    policyCount++;
  }
  return {
    entropyMean: policyCount > 0 ? entropySum / policyCount : null,
    topProbMean: policyCount > 0 ? topProbSum / policyCount : null,
    actionDiversity: actions.size / samples.length,
    avgSamplesPerGame: samples.length / Math.max(1, generatedGames)
  };
}

function summarizeGenerationDiagnostics(
  samples: TrainingSample[],
  generatedGames: number,
  source: string | null,
  alphaZeroStats?: AlphaZeroSelfPlayStats
): GenerationDiagnostics {
  if (samples.length === 0) {
    return {
      terminalRate: null,
      loopRate: null,
      wallMoveRate: null,
      forwardProgressRate: null,
      pathAdvantageMean: null,
      avgGameLength: generatedGames > 0 ? 0 : null,
      generationSource: source
    };
  }
  const terminalCount = samples.filter((sample) => sample.terminal === true).length;
  const loopCount = samples.filter((sample) => (sample.loopPenalty ?? 0) > 0).length;
  const wallCount = samples.filter((sample) => sample.wallMove === true).length;
  const pathValues = samples
    .map((sample) => sample.pathAdvantage)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  const progressValues = samples
    .filter((sample) => sample.wallMove !== true)
    .map((sample) => sample.pathProgress)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  const forwardCount = progressValues.filter((value) => value > 0).length;
  return {
    terminalRate: alphaZeroStats ? alphaZeroStats.terminalGames / Math.max(1, alphaZeroStats.terminalGames + alphaZeroStats.drawGames) : terminalCount / samples.length,
    loopRate: alphaZeroStats ? alphaZeroStats.loopCount / Math.max(1, samples.length) : loopCount / samples.length,
    wallMoveRate: alphaZeroStats ? alphaZeroStats.wallMoves / Math.max(1, alphaZeroStats.wallMoves + alphaZeroStats.pawnMoves) : wallCount / samples.length,
    forwardProgressRate: alphaZeroStats
      ? alphaZeroStats.forwardMoves / Math.max(1, alphaZeroStats.pawnMoves)
      : progressValues.length > 0 ? forwardCount / progressValues.length : null,
    pathAdvantageMean: alphaZeroStats?.pathAdvantageMean ?? (pathValues.length > 0 ? pathValues.reduce((sum, value) => sum + value, 0) / pathValues.length : null),
    avgGameLength: alphaZeroStats?.avgGameLength ?? (samples.length / Math.max(1, generatedGames)),
    generationSource: source
  };
}

export function alphaZeroCurriculumSource(game: number, cfg: TrainingConfig): "teacher" | "random" | "greedy" | "alphabeta" | "alphazero" {
  if (cfg.curriculum === "off") return game <= cfg.alphaZeroWarmupGames ? "teacher" : "alphazero";
  if (game <= cfg.alphaZeroWarmupGames) return "teacher";
  const phaseGame = game - cfg.alphaZeroWarmupGames;
  const blockIndex = Math.floor((phaseGame - 1) / Math.max(1, cfg.alphaZeroParallelGames));
  const bucket = blockIndex % 10;
  if (phaseGame <= 120) {
    return "greedy";
  }
  if (phaseGame <= 500) {
    if (bucket < 6) return "greedy";
    if (bucket < 8) return "alphabeta";
    return "random";
  }
  if (phaseGame <= 1800) {
    if (bucket < 5) return "greedy";
    if (bucket < 8) return "alphabeta";
    return "random";
  }
  if (phaseGame <= 8000) {
    if (bucket < 2) return "random";
    if (bucket < 5) return "greedy";
    if (bucket < 7) return "alphabeta";
    return "alphazero";
  }
  if (bucket === 0) return "random";
  if (bucket === 1) return "greedy";
  if (bucket === 2) return "alphabeta";
  return "alphazero";
}

function greedyBestResponseMaxMoves(cfg: TrainingConfig): number {
  return Math.min(120, Math.max(80, cfg.maxMovesPerGame));
}

function crossedIntervalGame(previousGame: number, currentGame: number, interval: number): number | null {
  if (interval <= 0) return null;
  const previousBucket = Math.floor(previousGame / interval);
  const currentBucket = Math.floor(currentGame / interval);
  return currentBucket > previousBucket ? currentBucket * interval : null;
}

function sampleFromArray<T>(items: T[], count: number): T[] {
  if (items.length === 0 || count <= 0) return [];
  const result: T[] = [];
  for (let i = 0; i < count; i++) {
    result.push(items[Math.floor(Math.random() * items.length)]);
  }
  return result;
}

function sampleWeightedSamples(items: TrainingSample[], count: number): TrainingSample[] {
  if (items.length === 0 || count <= 0) return [];
  const weights = items.map((sample) => Math.max(0.05, Math.min(10, sample.priority ?? 1)));
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
  if (!Number.isFinite(totalWeight) || totalWeight <= 0) return sampleFromArray(items, count);
  const result: TrainingSample[] = [];
  for (let i = 0; i < count; i++) {
    let pick = Math.random() * totalWeight;
    let selected = items[items.length - 1];
    for (let index = 0; index < items.length; index++) {
      pick -= weights[index];
      if (pick <= 0) {
        selected = items[index];
        break;
      }
    }
    result.push(selected);
  }
  return result;
}

function greedyAgreementScore(agreement: GreedyPolicyAgreement): number {
  return agreement.agreement + agreement.top3Agreement * 0.2 + agreement.greedyProbMean * 0.25;
}

function bestResponseScore(agreement: GreedyBestResponseAgreement, greedyAgreement?: GreedyPolicyAgreement | null): number {
  const greedySupport = greedyAgreement ? greedyAgreement.top3Agreement * 0.05 : 0;
  return agreement.agreement + agreement.top3Agreement * 0.35 + agreement.teacherProbMean * 0.4 + greedySupport;
}

async function deterministicEvaluationState(index: number): Promise<QuoridorState> {
  let state = createInitialState();
  const prefixLength = index;
  for (let ply = 0; ply < prefixLength && getWinner(state) === null; ply++) {
    if ((index + ply) % 3 !== 1) {
      state = applyMove(state, await GreedyShortestPathAgent.selectMove(state));
      continue;
    }
    const legalMoves = getLegalMoves(state)
      .slice()
      .sort((a, b) => moveToActionIndex(a) - moveToActionIndex(b));
    const pawnMoves = legalMoves.filter((move) => move.type === "pawn");
    const choices = pawnMoves.length > 0 ? pawnMoves : legalMoves;
    state = applyMove(state, choices[(index + ply) % choices.length]);
  }
  return state;
}

async function throttleForGpuTarget(workMs: number, target: number, cooldownMs: number): Promise<number> {
  const clippedTarget = Math.max(0.2, Math.min(MAX_GPU_UTILIZATION_TARGET, target));
  if (clippedTarget >= 0.995 || workMs <= 0) return 0;
  const proportionalSleepMs = Math.max(0, Math.round(workMs * ((1 - clippedTarget) / clippedTarget)));
  const sleepMs = Math.min(4000, Math.max(Math.max(0, cooldownMs), proportionalSleepMs));
  if (sleepMs > 0) await new Promise((resolve) => setTimeout(resolve, sleepMs));
  return sleepMs;
}

async function readNvidiaGpuUtilization(): Promise<number | null> {
  try {
    const { stdout } = await execFileAsync("nvidia-smi", [
      "--query-gpu=utilization.gpu",
      "--format=csv,noheader,nounits"
    ], { timeout: 3000, windowsHide: true });
    const values = stdout
      .split(/\r?\n/)
      .map((line) => Number(line.trim()))
      .filter((value) => Number.isFinite(value));
    if (values.length === 0) return null;
    return Math.max(...values) / 100;
  } catch {
    return null;
  }
}

async function waitForGpuHeadroom(target: number, cooldownMs: number): Promise<number> {
  const clippedTarget = Math.max(0.2, Math.min(MAX_GPU_UTILIZATION_TARGET, target));
  const startedAt = Date.now();
  let waitedMs = 0;
  while (Date.now() - startedAt < GPU_MAX_WAIT_MS) {
    const utilization = await readNvidiaGpuUtilization();
    if (utilization === null || utilization <= clippedTarget) break;
    await new Promise((resolve) => setTimeout(resolve, Math.max(GPU_POLL_INTERVAL_MS, cooldownMs)));
    waitedMs = Date.now() - startedAt;
  }
  return waitedMs;
}

function effectiveFitBatchSize(cfg: TrainingConfig, backend: TrainingBackendInfo, batchLength: number): number {
  if (backend.activeBackend === "cpu") return Math.min(cfg.batchSize, batchLength);
  const target = Math.max(0.2, Math.min(MAX_GPU_UTILIZATION_TARGET, cfg.gpuUtilizationTarget));
  const cap = target <= 0.45 ? 32 : target <= 0.6 ? 48 : 64;
  return Math.max(8, Math.min(cfg.batchSize, cap, batchLength));
}

async function evaluateGreedyPolicyAgreement(
  model: TfjsQuoridorPolicyModel,
  games = 4,
  maxPlies = 18
): Promise<GreedyPolicyAgreement> {
  let states = 0;
  let top1 = 0;
  let top3 = 0;
  let greedyProbabilitySum = 0;
  for (let game = 0; game < games; game++) {
    let state = createInitialState();
    for (let ply = 0; ply < maxPlies && getWinner(state) === null; ply++) {
      const legalMask = getModelLegalActionMask(state);
      const [prediction, greedyMove] = await Promise.all([
        model.predict(encodeStateForModel(state), legalMask),
        GreedyShortestPathAgent.selectMove(state)
      ]);
      const greedyAction = moveToModelActionIndex(state, greedyMove);
      const bestAction = chooseBestLegalAction(prediction.policy, legalMask);
      const topActions = Array.from(prediction.policy.entries())
        .filter(([action]) => legalMask[action] === 1)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([action]) => action);
      states++;
      if (bestAction === greedyAction) top1++;
      if (topActions.includes(greedyAction)) top3++;
      greedyProbabilitySum += prediction.policy[greedyAction] ?? 0;

      const rolloutMove = (game + ply) % 3 === 0
        ? await RandomAgent.selectMove(state)
        : greedyMove;
      state = applyMove(state, rolloutMove);
    }
  }
  const denominator = Math.max(1, states);
  return {
    states,
    agreement: top1 / denominator,
    top3Agreement: top3 / denominator,
    greedyProbMean: greedyProbabilitySum / denominator
  };
}

async function evaluateGreedyBestResponsePolicy(
  model: TfjsQuoridorPolicyModel,
  games = 4,
  maxPlies = 14
): Promise<GreedyBestResponseAgreement> {
  let states = 0;
  let top1 = 0;
  let top3 = 0;
  let teacherProbabilitySum = 0;
  for (let game = 0; game < games; game++) {
    let state = await deterministicEvaluationState(game);
    for (let ply = 0; ply < maxPlies && getWinner(state) === null; ply++) {
      const legalMask = getModelLegalActionMask(state);
      const prediction = await model.predict(encodeStateForModel(state), legalMask);
      const teacherMove = await selectBestMoveAgainstGreedy(state);
      const teacherAction = moveToModelActionIndex(state, teacherMove);
      const bestAction = chooseBestLegalAction(prediction.policy, legalMask);
      const topActions = Array.from(prediction.policy.entries())
        .filter(([action]) => legalMask[action] === 1)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([action]) => action);
      states++;
      if (bestAction === teacherAction) top1++;
      if (topActions.includes(teacherAction)) top3++;
      teacherProbabilitySum += prediction.policy[teacherAction] ?? 0;
      state = applyMove(state, teacherMove);
      if (getWinner(state) !== null) break;
      state = applyMove(state, await GreedyShortestPathAgent.selectMove(state));
      if (ply % 4 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }
  const denominator = Math.max(1, states);
  return {
    states,
    agreement: top1 / denominator,
    top3Agreement: top3 / denominator,
    teacherProbMean: teacherProbabilitySum / denominator
  };
}

export async function train(config: Partial<TrainingConfig> = {}, hooks: {
  shouldStop?: () => boolean;
  onLog?: (message: string) => void;
  onProgress?: (progress: TrainingProgress) => void;
  onCheckpoint?: (checkpoint: { game: number; modelPath: string }) => void;
  onReplayModel?: (replayModel: { game: number; modelPath: string }) => void;
} = {}): Promise<{ model: tf.LayersModel; modelPath: string; samples: number; backend: TrainingBackendInfo }> {
  const cfg = mergeTrainingConfig(config);
  const backend = await initializeTrainingBackend(cfg.backend, hooks.onLog);
  const checkpointModelJsonPath = path.join("models/checkpoints/latest", "model.json");
  const resumedFromCheckpoint = cfg.resumeFromCheckpoint && fs.existsSync(checkpointModelJsonPath);
  const model = await createOrResumePolicyValueModel(cfg, hooks.onLog);
  const policyModel = new TfjsQuoridorPolicyModel(model);
  const buffer = new ReplayBuffer();
  const persistentReplay = cfg.persistentReplay ? new PersistentReplayStore("data/replay", cfg.persistentReplayMaxLoad) : null;
  let persistentReplayAppendQueue: Promise<void> = Promise.resolve();
  let currentLoss: number | null = null;
  let policyLoss: number | null = null;
  let valueLoss: number | null = null;
  let modelPath: string | null = null;
  let completedCheckpointPath: string | null = null;
  let checkpointSaving = false;
  let pendingCheckpoint: { game: number; snapshot: tf.LayersModel } | null = null;
  let checkpointIdle: Promise<void> = Promise.resolve();
  let resolveCheckpointIdle: (() => void) | null = null;
  const rehearsalSamples: TrainingSample[] = [];

  const markCheckpointBusy = () => {
    if (!resolveCheckpointIdle) {
      checkpointIdle = new Promise((resolve) => {
        resolveCheckpointIdle = resolve;
      });
    }
  };

  const markCheckpointIdle = () => {
    if (!checkpointSaving && !pendingCheckpoint && resolveCheckpointIdle) {
      resolveCheckpointIdle();
      resolveCheckpointIdle = null;
    }
  };

  const runCheckpointSave = async (checkpoint: { game: number; snapshot: tf.LayersModel }) => {
    checkpointSaving = true;
    markCheckpointBusy();
    const path = "models/checkpoints/latest";
    try {
      await saveModelForBrowser(checkpoint.snapshot, path);
      modelPath = path;
      completedCheckpointPath = path;
      hooks.onLog?.(`Saved model to ${path} after game ${checkpoint.game}`);
      hooks.onCheckpoint?.({ game: checkpoint.game, modelPath: path });
    } catch (error) {
      hooks.onLog?.(`Checkpoint save failed after game ${checkpoint.game}: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      checkpoint.snapshot.dispose();
      checkpointSaving = false;
      const next = pendingCheckpoint;
      pendingCheckpoint = null;
      if (next) {
        void runCheckpointSave(next);
      } else {
        markCheckpointIdle();
      }
    }
  };

  const queueCheckpointSave = (game: number) => {
    modelPath = "models/checkpoints/latest";
    const snapshot = snapshotModel(model, cfg.learningRate);
    if (checkpointSaving) {
      pendingCheckpoint?.snapshot.dispose();
      pendingCheckpoint = { game, snapshot };
      hooks.onLog?.(`Checkpoint save queued in background after game ${game}`);
      return;
    }
    hooks.onLog?.(`Checkpoint save started in background after game ${game}`);
    void runCheckpointSave({ game, snapshot });
  };

  const queueReplayGeneration = (game: number) => {
    const replayModelPath = completedCheckpointPath;
    if (!replayModelPath) {
      hooks.onLog?.(`Replay skipped after game ${game}: no completed checkpoint available yet`);
      return;
    }
    hooks.onLog?.(`Replay generation queued after game ${game} using ${replayModelPath}`);
    hooks.onReplayModel?.({ game, modelPath: replayModelPath });
  };

  const queuePersistentReplayAppend = (samples: TrainingSample[]) => {
    if (!persistentReplay || samples.length === 0) return;
    persistentReplayAppendQueue = persistentReplayAppendQueue
      .then(() => persistentReplay.append(samples))
      .catch((error) => hooks.onLog?.(`Persistent replay append failed: ${error instanceof Error ? error.message : String(error)}`));
  };

  hooks.onLog?.(`Training started: mode=${cfg.mode}, games=${cfg.games}, backend=${backend.activeBackend}${backend.loadedPackage ? ` (${backend.loadedPackage})` : ""}`);
  hooks.onLog?.(`RL settings: curriculum=${cfg.curriculum}, topK=${cfg.rlSearchTopK}, searchMs=${cfg.rlSearchTimeMs}, searchDepth=${cfg.rlSearchMaxDepth}, explorationNoise=${cfg.explorationNoise}, teacherPrefixMax=${cfg.teacherRandomPrefixMax}, persistentReplay=${cfg.persistentReplay}`);
  hooks.onLog?.(`AlphaZero settings: simulations=${cfg.alphaZeroSimulations}, fastSimulations=${cfg.alphaZeroFastSimulations}, fullStart=${cfg.alphaZeroFullSimulationStart}, warmupGames=${cfg.alphaZeroWarmupGames}, parallelGames=${cfg.alphaZeroParallelGames}, updatesPerGame=${cfg.alphaZeroUpdatesPerGame}, cpuct=${cfg.alphaZeroCpuct}, heuristicPriorMix=${cfg.alphaZeroHeuristicPriorMix}, drawValueWeight=${cfg.alphaZeroDrawValueWeight}, maxMoves=${cfg.maxMovesPerGame}, batch=${cfg.batchSize}, effectiveGpuBatch=${effectiveFitBatchSize(cfg, backend, cfg.batchSize)}, gpuTarget=${Math.round(cfg.gpuUtilizationTarget * 100)}%, gpuCooldown=${cfg.gpuCooldownMs}ms`);
  if (persistentReplay) {
    const loadedSamples = await persistentReplay.load();
    buffer.addMany(loadedSamples);
    hooks.onLog?.(`Loaded ${loadedSamples.length} persistent replay samples`);
  }

  if (cfg.mode === "alphazero" && !resumedFromCheckpoint) {
    const bootstrapStartedAt = Date.now();
    compilePolicyValueModel(model, Math.max(0.002, cfg.learningRate * 4), 0.12);
    const bootstrapImitationSamples = await generateGreedyImitationSamples({
      games: cfg.bootstrapGames,
      maxMovesPerGame: Math.min(cfg.maxMovesPerGame, 48),
      randomMoveRate: 0.55
    });
    const bootstrapBestResponseSamples = cfg.bootstrapBestResponseGames > 0
      ? await generateGreedyBestResponseSamples({
        games: cfg.bootstrapBestResponseGames,
        maxMovesPerGame: greedyBestResponseMaxMoves(cfg),
        teacherTimeMs: Math.max(4, Math.min(25, cfg.teacherTimeMs)),
        teacherMaxDepth: Math.max(2, Math.min(3, cfg.teacherMaxDepth)),
        explorationRate: 0
      })
      : [];
    const bootstrapPrefixStressSamples = cfg.prefixStressGames > 0
      ? await generatePrefixStressTeacherSamples({
        games: cfg.prefixStressGames,
        rolloutPlies: cfg.prefixStressRolloutPlies,
        teacherTimeMs: Math.max(40, cfg.teacherTimeMs),
        teacherMaxDepth: Math.max(4, cfg.teacherMaxDepth)
      })
      : [];
    const bootstrapSamples = bootstrapBestResponseSamples.concat(bootstrapImitationSamples, bootstrapPrefixStressSamples);
    hooks.onLog?.(`Bootstrap samples generated: bestResponse=${bootstrapBestResponseSamples.length}, imitation=${bootstrapImitationSamples.length}, prefixStress=${bootstrapPrefixStressSamples.length}, total=${bootstrapSamples.length}`);
    rehearsalSamples.push(...bootstrapSamples);
    buffer.addMany(bootstrapSamples);
    queuePersistentReplayAppend(bootstrapSamples);
    const bootstrapTargetBestResponse = 0.55;
    const bootstrapMaxUpdates = backend.activeBackend === "cpu"
      ? Math.min(cfg.bootstrapMaxUpdates, 1)
      : cfg.bootstrapMaxUpdates;
    let bootstrapUpdates = 0;
    let bootstrapThrottleMs = 0;
    let greedyAgreement: GreedyPolicyAgreement | null = null;
    let bestResponseAgreement: GreedyBestResponseAgreement | null = null;
    let bestBootstrapAgreement: GreedyPolicyAgreement | null = null;
    let bestBootstrapBestResponse: GreedyBestResponseAgreement | null = null;
    let bestBootstrapScore = -Infinity;
    let bestBootstrapWeights: tf.Tensor[] | null = null;
    while (bootstrapUpdates < bootstrapMaxUpdates) {
      if (hooks.shouldStop?.()) break;
      const updateStartedAt = Date.now();
      const bestResponseCount = Math.min(
        Math.floor(cfg.batchSize * 0.62),
        bootstrapBestResponseSamples.length
      );
      const bestResponseBatch = bestResponseCount > 0 ? sampleWeightedSamples(bootstrapBestResponseSamples, bestResponseCount) : [];
      const prefixCount = Math.min(
        Math.floor(cfg.batchSize * 0.18),
        bootstrapPrefixStressSamples.length,
        cfg.batchSize - bestResponseBatch.length
      );
      const prefixBatch = prefixCount > 0 ? sampleWeightedSamples(bootstrapPrefixStressSamples, prefixCount) : [];
      const remainingCount = Math.min(cfg.batchSize - bestResponseBatch.length - prefixBatch.length, bootstrapSamples.length);
      const batch = bestResponseBatch.concat(prefixBatch, sampleWeightedSamples(bootstrapSamples, remainingCount));
      const tensors = samplesToTensors(batch);
      bootstrapThrottleMs += await waitForGpuHeadroom(cfg.gpuUtilizationTarget, cfg.gpuCooldownMs);
      const history = await model.fit(tensors.xs, { policy: tensors.policyY, value: tensors.valueY }, {
        epochs: cfg.epochs,
        batchSize: effectiveFitBatchSize(cfg, backend, batch.length),
        verbose: 0
      });
      tensors.xs.dispose();
      tensors.policyY.dispose();
      tensors.valueY.dispose();
      const lossValue = Number(history.history.loss?.at(-1) ?? 0);
      const policyLossValue = Number(history.history.policy_loss?.at(-1) ?? history.history.policy_loss ?? 0);
      const valueLossValue = Number(history.history.value_loss?.at(-1) ?? history.history.value_loss ?? 0);
      currentLoss = Number.isFinite(lossValue) ? lossValue : null;
      policyLoss = Number.isFinite(policyLossValue) ? policyLossValue : null;
      valueLoss = Number.isFinite(valueLossValue) ? valueLossValue : null;
      bootstrapUpdates++;
      bootstrapThrottleMs += await throttleForGpuTarget(Date.now() - updateStartedAt, cfg.gpuUtilizationTarget, cfg.gpuCooldownMs);
      bootstrapThrottleMs += await waitForGpuHeadroom(cfg.gpuUtilizationTarget, cfg.gpuCooldownMs);
      if (bootstrapUpdates % Math.min(8, Math.max(1, bootstrapMaxUpdates)) === 0) {
        greedyAgreement = await evaluateGreedyPolicyAgreement(policyModel);
        bestResponseAgreement = await evaluateGreedyBestResponsePolicy(policyModel, 2, 10);
        hooks.onLog?.(`Bootstrap progress: updates=${bootstrapUpdates}, loss=${currentLoss?.toFixed(3) ?? "-"}, greedyAgree=${greedyAgreement.agreement.toFixed(2)}, greedyTop3=${greedyAgreement.top3Agreement.toFixed(2)}, greedyProb=${greedyAgreement.greedyProbMean.toFixed(2)}, bestResp=${bestResponseAgreement.agreement.toFixed(2)}, bestRespTop3=${bestResponseAgreement.top3Agreement.toFixed(2)}, bestRespProb=${bestResponseAgreement.teacherProbMean.toFixed(2)}`);
        hooks.onProgress?.({
          game: 0,
          samples: buffer.size(),
          loss: currentLoss,
          policyLoss,
          valueLoss,
          rewardMean: null,
          rewardMin: null,
          rewardMax: null,
          policyEntropyMean: null,
          policyTopProbMean: null,
          actionDiversity: null,
          avgSamplesPerGame: null,
          terminalRate: null,
          loopRate: null,
          wallMoveRate: null,
          forwardProgressRate: null,
          pathAdvantageMean: null,
          avgGameLength: null,
          generationSource: "bootstrap-greedy",
          modelPath,
          backend,
          evaluation: { greedyAgreement, greedyBestResponse: bestResponseAgreement }
        });
        const agreementScore = bestResponseScore(bestResponseAgreement, greedyAgreement);
        if (!bestBootstrapAgreement || agreementScore > bestBootstrapScore) {
          bestBootstrapWeights?.forEach((weight) => weight.dispose());
          bestBootstrapWeights = model.getWeights().map((weight) => weight.clone());
          bestBootstrapAgreement = greedyAgreement;
          bestBootstrapBestResponse = bestResponseAgreement;
          bestBootstrapScore = agreementScore;
        }
        if (bestResponseAgreement.agreement >= bootstrapTargetBestResponse || bestResponseAgreement.top3Agreement >= 0.85) break;
      }
    }
    if (bestBootstrapWeights && bestBootstrapAgreement) {
      model.setWeights(bestBootstrapWeights);
      greedyAgreement = bestBootstrapAgreement;
      bestResponseAgreement = bestBootstrapBestResponse;
      hooks.onLog?.(`Bootstrap restored best weights: greedyAgree=${greedyAgreement.agreement.toFixed(2)}, greedyTop3=${greedyAgreement.top3Agreement.toFixed(2)}, greedyProb=${greedyAgreement.greedyProbMean.toFixed(2)}, bestResp=${bestResponseAgreement?.agreement.toFixed(2) ?? "-"}, bestRespTop3=${bestResponseAgreement?.top3Agreement.toFixed(2) ?? "-"}, bestRespProb=${bestResponseAgreement?.teacherProbMean.toFixed(2) ?? "-"}`);
    }
    greedyAgreement ??= await evaluateGreedyPolicyAgreement(policyModel);
    bestResponseAgreement ??= await evaluateGreedyBestResponsePolicy(policyModel, 2, 10);
    compilePolicyValueModel(model, cfg.learningRate);
    hooks.onLog?.(`Bootstrap greedy policy: samples=${bootstrapSamples.length}, updates=${bootstrapUpdates}, ms=${Date.now() - bootstrapStartedAt}, throttleMs=${bootstrapThrottleMs}, greedyAgree=${greedyAgreement.agreement.toFixed(2)}, greedyTop3=${greedyAgreement.top3Agreement.toFixed(2)}, greedyProb=${greedyAgreement.greedyProbMean.toFixed(2)}, bestResp=${bestResponseAgreement.agreement.toFixed(2)}, bestRespTop3=${bestResponseAgreement.top3Agreement.toFixed(2)}, bestRespProb=${bestResponseAgreement.teacherProbMean.toFixed(2)}`);
  }

  for (let game = 1; game <= cfg.games; game++) {
    if (hooks.shouldStop?.()) break;
    const previousGame = game - 1;
    let completedGame = game;
    let generatedGames = 1;
    const gameStartedAt = Date.now();
    let generationMs = 0;
    let fitMs = 0;
    let throttleMs = 0;
    let evaluationMs = 0;
    let fitUpdates = 0;
    let generatedSamples: TrainingSample[];
    let alphaZeroStats: AlphaZeroSelfPlayStats | undefined;
    let generationSource: string | null = null;
    const alphaZeroSource = cfg.mode === "alphazero" ? alphaZeroCurriculumSource(game + cfg.curriculumGameOffset, cfg) : null;
    const useTeacher = cfg.mode === "supervised"
      || alphaZeroSource === "teacher"
      || (cfg.mode === "mixed" && (cfg.curriculum === "auto" ? game % 3 !== 0 : game % 2 === 1));
    const generationStartedAt = Date.now();
    if (useTeacher) {
      generationSource = cfg.mode === "alphazero" ? "curriculum-teacher" : "teacher";
      generatedSamples = await generateTeacherSamples({
        games: 1,
        maxMovesPerGame: Math.min(cfg.maxMovesPerGame, 80),
        teacherTimeMs: cfg.teacherTimeMs,
        teacherMaxDepth: cfg.teacherMaxDepth,
        randomPrefixMoves: cfg.teacherRandomPrefixMax,
        valueScale: cfg.teacherValueScale
      });
    } else if (cfg.mode === "alphazero" && (alphaZeroSource === "random" || alphaZeroSource === "greedy" || alphaZeroSource === "alphabeta")) {
      const modelPlayer = Math.random() < 0.5 ? 0 : 1;
      generationSource = `curriculum-${alphaZeroSource}-p${modelPlayer}`;
      const phaseGame = game + cfg.curriculumGameOffset - cfg.alphaZeroWarmupGames;
      if (alphaZeroSource === "greedy" && phaseGame <= 1200) {
        const bestResponseSamples = await generateGreedyBestResponseSamples({
          games: cfg.greedyBestResponseGames,
          maxMovesPerGame: greedyBestResponseMaxMoves(cfg),
          teacherTimeMs: Math.max(4, Math.min(25, cfg.teacherTimeMs)),
          teacherMaxDepth: Math.max(2, Math.min(3, cfg.teacherMaxDepth)),
          explorationRate: 0.03
        });
        const imitationSamples = await generateGreedyImitationSamples({
          games: 1,
          maxMovesPerGame: Math.min(cfg.maxMovesPerGame, 32),
          randomMoveRate: 0.45
        });
        generatedSamples = bestResponseSamples.concat(imitationSamples);
        rehearsalSamples.push(...bestResponseSamples);
        if (rehearsalSamples.length > 8000) rehearsalSamples.splice(0, rehearsalSamples.length - 8000);
        generationSource = "curriculum-greedy-bestresponse+imitation";
      } else {
        generatedSamples = await runSearchImprovedSelfPlayGame({
          model: policyModel,
          maxMovesPerGame: cfg.maxMovesPerGame,
          temperature: Math.max(0.8, cfg.temperature),
          searchTopK: cfg.rlSearchTopK,
          searchTimeMs: cfg.rlSearchTimeMs,
          searchMaxDepth: cfg.rlSearchMaxDepth,
          opponentMix: alphaZeroSource,
          modelPlayer,
          trainOpponentMoves: false,
          rewardShaping: true,
          explorationNoise: Math.max(cfg.explorationNoise, 0.22)
        }).catch(async (error) => {
          hooks.onLog?.(`Curriculum game failed, falling back to AlphaZero self-play: ${error instanceof Error ? error.message : String(error)}`);
          const fallback = await runBatchedAlphaZeroSelfPlayGames({
            model: policyModel,
            games: 1,
            maxMovesPerGame: cfg.maxMovesPerGame,
            simulations: cfg.alphaZeroFastSimulations,
            cpuct: cfg.alphaZeroCpuct,
            dirichletAlpha: cfg.alphaZeroDirichletAlpha,
            dirichletFraction: cfg.alphaZeroDirichletFraction,
            temperatureMoves: cfg.alphaZeroTemperatureMoves,
            minTemperature: cfg.alphaZeroMinTemperature,
            policyTargetSmoothing: cfg.alphaZeroPolicySmoothing,
            heuristicPriorMix: cfg.alphaZeroHeuristicPriorMix,
            drawValueWeight: cfg.alphaZeroDrawValueWeight
          });
          alphaZeroStats = fallback.stats;
          generationSource = "alphazero-fallback";
          return fallback.samples;
        });
      }
    } else if (cfg.mode === "alphazero" || cfg.mode === "mixed") {
      const alphaZeroSimulations = cfg.mode === "alphazero" && game < cfg.alphaZeroFullSimulationStart
        ? cfg.alphaZeroFastSimulations
        : cfg.alphaZeroSimulations;
      if (cfg.mode === "alphazero") {
        generationSource = "alphazero";
        const parallelGames = Math.max(1, Math.min(cfg.alphaZeroParallelGames, cfg.games - game + 1));
        const batch = await runBatchedAlphaZeroSelfPlayGames({
          model: policyModel,
          games: parallelGames,
          maxMovesPerGame: cfg.maxMovesPerGame,
          simulations: alphaZeroSimulations,
          cpuct: cfg.alphaZeroCpuct,
          dirichletAlpha: cfg.alphaZeroDirichletAlpha,
          dirichletFraction: cfg.alphaZeroDirichletFraction,
          temperatureMoves: cfg.alphaZeroTemperatureMoves,
          minTemperature: cfg.alphaZeroMinTemperature,
          policyTargetSmoothing: cfg.alphaZeroPolicySmoothing,
          heuristicPriorMix: cfg.alphaZeroHeuristicPriorMix,
          drawValueWeight: cfg.alphaZeroDrawValueWeight
        });
        generatedSamples = batch.samples;
        generatedGames = batch.games;
        completedGame = game + generatedGames - 1;
        alphaZeroStats = batch.stats;
      } else {
        generationSource = "mixed-alphazero";
        generatedSamples = await runAlphaZeroSelfPlayGame({
          model: policyModel,
          maxMovesPerGame: cfg.maxMovesPerGame,
          simulations: alphaZeroSimulations,
          cpuct: cfg.alphaZeroCpuct,
          dirichletAlpha: cfg.alphaZeroDirichletAlpha,
          dirichletFraction: cfg.alphaZeroDirichletFraction,
          temperatureMoves: cfg.alphaZeroTemperatureMoves,
          minTemperature: cfg.alphaZeroMinTemperature,
          policyTargetSmoothing: cfg.alphaZeroPolicySmoothing,
          heuristicPriorMix: cfg.alphaZeroHeuristicPriorMix,
          drawValueWeight: cfg.alphaZeroDrawValueWeight
        });
      }
    } else {
      const opponentMix = cfg.curriculum === "off"
        ? (game % 6 === 0 ? "greedy" : game % 6 === 3 ? "random" : "self")
        : game <= 300
          ? "random"
          : game <= 900
            ? (game % 6 === 0 ? "greedy" : game % 4 === 0 ? "self" : "random")
            : game % 5 === 0
              ? "greedy"
              : game % 3 === 0
              ? "self"
              : "random";
      generationSource = `selfplay-${opponentMix}`;
      generatedSamples = await runSearchImprovedSelfPlayGame({
        model: policyModel,
        maxMovesPerGame: cfg.maxMovesPerGame,
        temperature: cfg.temperature,
        searchTopK: cfg.rlSearchTopK,
        searchTimeMs: cfg.rlSearchTimeMs,
        searchMaxDepth: cfg.rlSearchMaxDepth,
        opponentMix,
        modelPlayer: Math.random() < 0.5 ? 0 : 1,
        trainOpponentMoves: opponentMix === "self",
        rewardShaping: true,
        explorationNoise: cfg.explorationNoise
      }).catch(async (error) => {
        hooks.onLog?.(`Search-improved self-play failed, falling back to policy self-play: ${error instanceof Error ? error.message : String(error)}`);
        return runSelfPlayGame({ model: policyModel, maxMovesPerGame: cfg.maxMovesPerGame, temperature: cfg.temperature });
      });
    }
    generationMs = Date.now() - generationStartedAt;
    if (cfg.mode === "alphazero" && generationSource?.startsWith("curriculum-greedy-p") === true) {
      const imitationStartedAt = Date.now();
      const imitationSamples = await generateGreedyImitationSamples({
        games: 1,
        maxMovesPerGame: Math.min(cfg.maxMovesPerGame, 64),
        randomMoveRate: 0.22
      });
      generatedSamples = generatedSamples.concat(imitationSamples);
      rehearsalSamples.push(...imitationSamples);
      if (rehearsalSamples.length > 8000) rehearsalSamples.splice(0, rehearsalSamples.length - 8000);
      generationSource = `${generationSource}+greedy-imitation`;
      generationMs += Date.now() - imitationStartedAt;
    }
    if (cfg.mode === "alphazero" && cfg.prefixStressGames > 0 && game % 25 === 0) {
      const prefixStartedAt = Date.now();
      const prefixSamples = await generatePrefixStressTeacherSamples({
        games: Math.min(cfg.prefixStressGames, 3),
        rolloutPlies: cfg.prefixStressRolloutPlies,
        teacherTimeMs: Math.max(30, Math.min(120, cfg.teacherTimeMs)),
        teacherMaxDepth: Math.max(3, Math.min(5, cfg.teacherMaxDepth)),
        seedBase: 2000 + game
      });
      generatedSamples = generatedSamples.concat(prefixSamples);
      rehearsalSamples.push(...prefixSamples);
      if (rehearsalSamples.length > 8000) rehearsalSamples.splice(0, rehearsalSamples.length - 8000);
      generationSource = `${generationSource ?? "unknown"}+prefix-stress`;
      generationMs += Date.now() - prefixStartedAt;
    }
    throttleMs += await throttleForGpuTarget(generationMs, cfg.gpuUtilizationTarget, cfg.gpuCooldownMs);
    const rewards = summarizeRewards(generatedSamples);
    const policyStats = summarizePolicyTargets(generatedSamples, generatedGames);
    const generationDiagnostics = summarizeGenerationDiagnostics(generatedSamples, generatedGames, generationSource, alphaZeroStats);
    buffer.addMany(generatedSamples);
    queuePersistentReplayAppend(generatedSamples);

    if (buffer.size() > 0) {
      const fitStartedAt = Date.now();
      const isGreedyCurriculum = generationSource?.startsWith("curriculum-greedy") === true;
      const updateMultiplier = isGreedyCurriculum ? 8 : 1;
      const updateCount = cfg.mode === "alphazero" ? Math.max(1, generatedGames * cfg.alphaZeroUpdatesPerGame * updateMultiplier) : 1;
      let guardWeights: tf.Tensor[] | null = null;
      let guardScore: number | null = null;
      if (isGreedyCurriculum && cfg.mode === "alphazero") {
        compilePolicyValueModel(model, Math.max(0.001, cfg.learningRate * 2), 0.15);
        const guardAgreement = await evaluateGreedyBestResponsePolicy(policyModel, 2, 10);
        guardScore = bestResponseScore(guardAgreement);
        guardWeights = model.getWeights().map((weight) => weight.clone());
      }
      for (let update = 0; update < updateCount; update++) {
        const updateStartedAt = Date.now();
        const rehearsalFraction = cfg.mode === "alphazero" && rehearsalSamples.length > 0 ? (isGreedyCurriculum ? 0.05 : 0.3) : 0;
        const rehearsalCount = Math.floor(cfg.batchSize * rehearsalFraction);
        const rehearsalBatch = rehearsalCount > 0 ? sampleWeightedSamples(rehearsalSamples, rehearsalCount) : [];
        const freshFraction = isGreedyCurriculum ? 0.9 : 0.45;
        const maxFreshCount = Math.max(0, cfg.batchSize - rehearsalBatch.length);
        const freshCount = cfg.mode === "alphazero" && generatedSamples.length > 0
          ? Math.min(maxFreshCount, Math.floor(cfg.batchSize * freshFraction))
          : 0;
        const freshBatch = freshCount > 0 ? sampleWeightedSamples(generatedSamples, freshCount) : [];
        const replayBatch = buffer.sample(cfg.batchSize - freshBatch.length - rehearsalBatch.length);
        const batch = rehearsalBatch.concat(freshBatch, replayBatch);
        const tensors = samplesToTensors(batch);
        throttleMs += await waitForGpuHeadroom(cfg.gpuUtilizationTarget, cfg.gpuCooldownMs);
        const history = await model.fit(tensors.xs, { policy: tensors.policyY, value: tensors.valueY }, {
          epochs: cfg.epochs,
          batchSize: effectiveFitBatchSize(cfg, backend, batch.length),
          verbose: 0
        });
        tensors.xs.dispose();
        tensors.policyY.dispose();
        tensors.valueY.dispose();
        const lossValue = Number(history.history.loss?.at(-1) ?? 0);
        const policyLossValue = Number(history.history.policy_loss?.at(-1) ?? history.history.policy_loss ?? 0);
        const valueLossValue = Number(history.history.value_loss?.at(-1) ?? history.history.value_loss ?? 0);
        currentLoss = Number.isFinite(lossValue) ? lossValue : null;
        policyLoss = Number.isFinite(policyLossValue) ? policyLossValue : null;
        valueLoss = Number.isFinite(valueLossValue) ? valueLossValue : null;
        fitUpdates++;
        throttleMs += await throttleForGpuTarget(Date.now() - updateStartedAt, cfg.gpuUtilizationTarget, cfg.gpuCooldownMs);
        throttleMs += await waitForGpuHeadroom(cfg.gpuUtilizationTarget, cfg.gpuCooldownMs);
      }
      if (guardWeights && guardScore !== null) {
        const updatedAgreement = await evaluateGreedyBestResponsePolicy(policyModel, 2, 10);
        const updatedScore = bestResponseScore(updatedAgreement);
        if (updatedScore + 0.03 < guardScore) {
          model.setWeights(guardWeights);
          hooks.onLog?.(`Greedy curriculum update reverted: bestRespScore ${updatedScore.toFixed(3)} < ${guardScore.toFixed(3)}, bestResp=${updatedAgreement.agreement.toFixed(2)}, top3=${updatedAgreement.top3Agreement.toFixed(2)}`);
        } else {
          hooks.onLog?.(`Greedy curriculum update accepted: bestRespScore ${updatedScore.toFixed(3)} >= ${guardScore.toFixed(3)}, bestResp=${updatedAgreement.agreement.toFixed(2)}, top3=${updatedAgreement.top3Agreement.toFixed(2)}`);
        }
        guardWeights.forEach((weight) => weight.dispose());
        compilePolicyValueModel(model, cfg.learningRate);
      }
      fitMs = Date.now() - fitStartedAt;
    }

    let evaluation: unknown;
    const evaluationGame = crossedIntervalGame(previousGame, completedGame, cfg.evaluateEveryGames);
    if (evaluationGame !== null || completedGame === cfg.games) {
      const evaluationStartedAt = Date.now();
      const agent = new ModelAgent(policyModel);
      const random = await runMatchup({ agentA: agent, agentB: RandomAgent, games: 8, maxMovesPerGame: 140 });
      const greedy = await runMatchup({ agentA: agent, agentB: GreedyShortestPathAgent, games: 4, maxMovesPerGame: 160 });
      const alpha = await runMatchup({ agentA: agent, agentB: alphaBetaAgent(20, 3), games: 1, maxMovesPerGame: 160 });
      const greedyAgreement = await evaluateGreedyPolicyAgreement(policyModel);
      const greedyBestResponse = await evaluateGreedyBestResponsePolicy(policyModel);
      evaluation = { random, greedy, alpha20: alpha, greedyAgreement, greedyBestResponse };
      hooks.onLog?.(`Evaluation after ${evaluationGame ?? completedGame}: random=${random.winrateA.toFixed(2)}, greedy=${greedy.winrateA.toFixed(2)}, alpha20=${alpha.winrateA.toFixed(2)}, greedyAgree=${greedyAgreement.agreement.toFixed(2)}, greedyTop3=${greedyAgreement.top3Agreement.toFixed(2)}, greedyProb=${greedyAgreement.greedyProbMean.toFixed(2)}, bestResp=${greedyBestResponse.agreement.toFixed(2)}, bestRespTop3=${greedyBestResponse.top3Agreement.toFixed(2)}, bestRespProb=${greedyBestResponse.teacherProbMean.toFixed(2)}`);
      evaluationMs = Date.now() - evaluationStartedAt;
    }

    const checkpointGame = crossedIntervalGame(previousGame, completedGame, cfg.saveEveryGames);
    const replayGame = crossedIntervalGame(previousGame, completedGame, cfg.replayEveryGames);
    if (checkpointGame !== null || completedGame === cfg.games) {
      queueCheckpointSave(checkpointGame ?? completedGame);
    }
    if (replayGame !== null && replayGame !== checkpointGame) {
      queueReplayGeneration(replayGame);
    }
    if (evaluationMs > 0) {
      throttleMs += await throttleForGpuTarget(evaluationMs, cfg.gpuUtilizationTarget, cfg.gpuCooldownMs);
    }

    hooks.onLog?.(`Games ${game}${generatedGames > 1 ? `-${completedGame}` : ""} timings: source=${generationDiagnostics.generationSource ?? "-"}, gen=${generationMs}ms, fit=${fitMs}ms, throttle=${throttleMs}ms, eval=${evaluationMs}ms, total=${Date.now() - gameStartedAt}ms, generatedGames=${generatedGames}, fitUpdates=${fitUpdates}, samples=${generatedSamples.length}, rewardMean=${rewards.mean?.toFixed(3) ?? "-"}, terminal=${generationDiagnostics.terminalRate?.toFixed(2) ?? "-"}, loop=${generationDiagnostics.loopRate?.toFixed(2) ?? "-"}, wall=${generationDiagnostics.wallMoveRate?.toFixed(2) ?? "-"}, forward=${generationDiagnostics.forwardProgressRate?.toFixed(2) ?? "-"}`);

    hooks.onProgress?.({
      game: completedGame,
      samples: buffer.size(),
      loss: currentLoss,
      policyLoss,
      valueLoss,
      rewardMean: rewards.mean,
      rewardMin: rewards.min,
      rewardMax: rewards.max,
      policyEntropyMean: policyStats.entropyMean,
      policyTopProbMean: policyStats.topProbMean,
      actionDiversity: policyStats.actionDiversity,
      avgSamplesPerGame: policyStats.avgSamplesPerGame,
      terminalRate: generationDiagnostics.terminalRate,
      loopRate: generationDiagnostics.loopRate,
      wallMoveRate: generationDiagnostics.wallMoveRate,
      forwardProgressRate: generationDiagnostics.forwardProgressRate,
      pathAdvantageMean: generationDiagnostics.pathAdvantageMean,
      avgGameLength: generationDiagnostics.avgGameLength,
      generationSource: generationDiagnostics.generationSource,
      modelPath,
      backend,
      evaluation
    });
    await new Promise((resolve) => setTimeout(resolve, 0));
    game += generatedGames - 1;
  }

  if (!modelPath) {
    modelPath = "models/checkpoints/latest";
    await saveModelForBrowser(model, modelPath);
  }
  await persistentReplayAppendQueue;
  await checkpointIdle;
  hooks.onLog?.("Training finished");
  return { model, modelPath, samples: buffer.size(), backend };
}

function parseArgs(): Partial<TrainingConfig> {
  const args = process.argv.slice(2);
  const parsed: Record<string, string> = {};
  const value = (...candidates: Array<string | undefined>) =>
    candidates.find((candidate) => candidate !== undefined && candidate !== "" && candidate !== "true");
  for (let i = 0; i < args.length; i++) {
    if (!args[i].startsWith("--")) continue;
    const [key, inlineValue] = args[i].slice(2).split("=", 2);
    parsed[key] = inlineValue ?? args[i + 1];
  }
  const env = process.env as Record<string, string | undefined>;
  const positionalMode = ["mixed", "supervised", "selfplay", "alphazero"].includes(args[0]) ? args[0] : undefined;
  const mode = value(parsed.mode, env.npm_config_mode, positionalMode);
  const games = value(parsed.games, env.npm_config_games, mode === args[0] ? args[1] : undefined);
  const maxMovesPerGame = value(parsed.maxMovesPerGame, env.npm_config_maxmovespergame, env.npm_config_max_moves_per_game, mode === args[0] ? args[2] : undefined);
  const batchSize = value(parsed.batchSize, env.npm_config_batchsize, env.npm_config_batch_size, mode === args[0] ? args[3] : undefined);
  const epochs = value(parsed.epochs, env.npm_config_epochs, mode === args[0] ? args[4] : undefined);
  const learningRate = value(parsed.learningRate, env.npm_config_learningrate, env.npm_config_learning_rate);
  const teacherTimeMs = value(parsed.teacherTimeMs, env.npm_config_teachertimems, env.npm_config_teacher_time_ms);
  const teacherMaxDepth = value(parsed.teacherMaxDepth, env.npm_config_teachermaxdepth, env.npm_config_teacher_max_depth);
  const saveEveryGames = value(parsed.saveEveryGames, env.npm_config_saveeverygames, env.npm_config_save_every_games);
  const replayEveryGames = value(parsed.replayEveryGames, env.npm_config_replayeverygames, env.npm_config_replay_every_games);
  const evaluateEveryGames = value(parsed.evaluateEveryGames, env.npm_config_evaluateeverygames, env.npm_config_evaluate_every_games);
  const temperature = value(parsed.temperature, env.npm_config_temperature);
  const backend = value(parsed.backend, env.npm_config_backend);
  const rlSearchTopK = value(parsed.rlSearchTopK, env.npm_config_rlsearchtopk, env.npm_config_rl_search_top_k);
  const rlSearchTimeMs = value(parsed.rlSearchTimeMs, env.npm_config_rlsearchtimems, env.npm_config_rl_search_time_ms);
  const rlSearchMaxDepth = value(parsed.rlSearchMaxDepth, env.npm_config_rlsearchmaxdepth, env.npm_config_rl_search_max_depth);
  const persistentReplay = value(parsed.persistentReplay, env.npm_config_persistentreplay, env.npm_config_persistent_replay);
  const curriculum = value(parsed.curriculum, env.npm_config_curriculum);
  const teacherRandomPrefixMax = value(parsed.teacherRandomPrefixMax, env.npm_config_teacherrandomprefixmax, env.npm_config_teacher_random_prefix_max);
  const teacherValueScale = value(parsed.teacherValueScale, env.npm_config_teachervaluescale, env.npm_config_teacher_value_scale);
  const explorationNoise = value(parsed.explorationNoise, env.npm_config_explorationnoise, env.npm_config_exploration_noise);
  const alphaZeroSimulations = value(parsed.alphaZeroSimulations, env.npm_config_alphazerosimulations, env.npm_config_alpha_zero_simulations);
  const alphaZeroCpuct = value(parsed.alphaZeroCpuct, env.npm_config_alphazerocpuct, env.npm_config_alpha_zero_cpuct);
  const alphaZeroDirichletAlpha = value(parsed.alphaZeroDirichletAlpha, env.npm_config_alphazerodirichletalpha, env.npm_config_alpha_zero_dirichlet_alpha);
  const alphaZeroDirichletFraction = value(parsed.alphaZeroDirichletFraction, env.npm_config_alphazerodirichletfraction, env.npm_config_alpha_zero_dirichlet_fraction);
  const alphaZeroTemperatureMoves = value(parsed.alphaZeroTemperatureMoves, env.npm_config_alphazerotemperaturemoves, env.npm_config_alpha_zero_temperature_moves);
  const alphaZeroMinTemperature = value(parsed.alphaZeroMinTemperature, env.npm_config_alphazeromintemperature, env.npm_config_alpha_zero_min_temperature);
  const alphaZeroPolicySmoothing = value(parsed.alphaZeroPolicySmoothing, env.npm_config_alphazeropolicysmoothing, env.npm_config_alpha_zero_policy_smoothing);
  const alphaZeroWarmupGames = value(parsed.alphaZeroWarmupGames, env.npm_config_alphazerowarmupgames, env.npm_config_alpha_zero_warmup_games);
  const alphaZeroFastSimulations = value(parsed.alphaZeroFastSimulations, env.npm_config_alphazerofastsimulations, env.npm_config_alpha_zero_fast_simulations);
  const alphaZeroFullSimulationStart = value(parsed.alphaZeroFullSimulationStart, env.npm_config_alphazerofullsimulationstart, env.npm_config_alpha_zero_full_simulation_start);
  const alphaZeroHeuristicPriorMix = value(parsed.alphaZeroHeuristicPriorMix, env.npm_config_alphazeroheuristicpriormix, env.npm_config_alpha_zero_heuristic_prior_mix);
  const alphaZeroDrawValueWeight = value(parsed.alphaZeroDrawValueWeight, env.npm_config_alphazerodrawvalueweight, env.npm_config_alpha_zero_draw_value_weight);
  const alphaZeroParallelGames = value(parsed.alphaZeroParallelGames, env.npm_config_alphazeroparallelgames, env.npm_config_alpha_zero_parallel_games);
  const alphaZeroUpdatesPerGame = value(parsed.alphaZeroUpdatesPerGame, env.npm_config_alphazeroupdatespergame, env.npm_config_alpha_zero_updates_per_game);
  const resumeFromCheckpoint = value(parsed.resumeFromCheckpoint, env.npm_config_resumefromcheckpoint, env.npm_config_resume_from_checkpoint);
  const curriculumGameOffset = value(parsed.curriculumGameOffset, env.npm_config_curriculumgameoffset, env.npm_config_curriculum_game_offset);
  const gpuUtilizationTarget = value(parsed.gpuUtilizationTarget, env.npm_config_gpuutilizationtarget, env.npm_config_gpu_utilization_target);
  const gpuCooldownMs = value(parsed.gpuCooldownMs, env.npm_config_gpucooldownms, env.npm_config_gpu_cooldown_ms);
  const bootstrapMaxUpdates = value(parsed.bootstrapMaxUpdates, env.npm_config_bootstrapmaxupdates, env.npm_config_bootstrap_max_updates);
  const bootstrapGames = value(parsed.bootstrapGames, env.npm_config_bootstrapgames, env.npm_config_bootstrap_games);
  const bootstrapBestResponseGames = value(parsed.bootstrapBestResponseGames, env.npm_config_bootstrapbestresponsegames, env.npm_config_bootstrap_best_response_games);
  const greedyBestResponseGames = value(parsed.greedyBestResponseGames, env.npm_config_greedybestresponsegames, env.npm_config_greedy_best_response_games);
  const prefixStressGames = value(parsed.prefixStressGames, env.npm_config_prefixstressgames, env.npm_config_prefix_stress_games);
  const prefixStressRolloutPlies = value(parsed.prefixStressRolloutPlies, env.npm_config_prefixstressrolloutplies, env.npm_config_prefix_stress_rollout_plies);
  return {
    mode: mode as TrainingConfig["mode"] | undefined,
    games: games ? Number(games) : undefined,
    maxMovesPerGame: maxMovesPerGame ? Number(maxMovesPerGame) : undefined,
    batchSize: batchSize ? Number(batchSize) : undefined,
    epochs: epochs ? Number(epochs) : undefined,
    learningRate: learningRate ? Number(learningRate) : undefined,
    teacherTimeMs: teacherTimeMs ? Number(teacherTimeMs) : undefined,
    teacherMaxDepth: teacherMaxDepth ? Number(teacherMaxDepth) : undefined,
    saveEveryGames: saveEveryGames ? Number(saveEveryGames) : undefined,
    replayEveryGames: replayEveryGames ? Number(replayEveryGames) : undefined,
    evaluateEveryGames: evaluateEveryGames ? Number(evaluateEveryGames) : undefined,
    temperature: temperature ? Number(temperature) : undefined,
    backend: backend as TrainingBackendPreference | undefined,
    rlSearchTopK: rlSearchTopK ? Number(rlSearchTopK) : undefined,
    rlSearchTimeMs: rlSearchTimeMs ? Number(rlSearchTimeMs) : undefined,
    rlSearchMaxDepth: rlSearchMaxDepth ? Number(rlSearchMaxDepth) : undefined,
    persistentReplay: persistentReplay ? persistentReplay !== "false" : undefined,
    curriculum: curriculum as TrainingConfig["curriculum"] | undefined,
    teacherRandomPrefixMax: teacherRandomPrefixMax ? Number(teacherRandomPrefixMax) : undefined,
    teacherValueScale: teacherValueScale ? Number(teacherValueScale) : undefined,
    explorationNoise: explorationNoise ? Number(explorationNoise) : undefined,
    alphaZeroSimulations: alphaZeroSimulations ? Number(alphaZeroSimulations) : undefined,
    alphaZeroCpuct: alphaZeroCpuct ? Number(alphaZeroCpuct) : undefined,
    alphaZeroDirichletAlpha: alphaZeroDirichletAlpha ? Number(alphaZeroDirichletAlpha) : undefined,
    alphaZeroDirichletFraction: alphaZeroDirichletFraction ? Number(alphaZeroDirichletFraction) : undefined,
    alphaZeroTemperatureMoves: alphaZeroTemperatureMoves ? Number(alphaZeroTemperatureMoves) : undefined,
    alphaZeroMinTemperature: alphaZeroMinTemperature ? Number(alphaZeroMinTemperature) : undefined,
    alphaZeroPolicySmoothing: alphaZeroPolicySmoothing ? Number(alphaZeroPolicySmoothing) : undefined,
    alphaZeroWarmupGames: alphaZeroWarmupGames ? Number(alphaZeroWarmupGames) : undefined,
    alphaZeroFastSimulations: alphaZeroFastSimulations ? Number(alphaZeroFastSimulations) : undefined,
    alphaZeroFullSimulationStart: alphaZeroFullSimulationStart ? Number(alphaZeroFullSimulationStart) : undefined,
    alphaZeroHeuristicPriorMix: alphaZeroHeuristicPriorMix ? Number(alphaZeroHeuristicPriorMix) : undefined,
    alphaZeroDrawValueWeight: alphaZeroDrawValueWeight ? Number(alphaZeroDrawValueWeight) : undefined,
    alphaZeroParallelGames: alphaZeroParallelGames ? Number(alphaZeroParallelGames) : undefined,
    alphaZeroUpdatesPerGame: alphaZeroUpdatesPerGame ? Number(alphaZeroUpdatesPerGame) : undefined,
    resumeFromCheckpoint: resumeFromCheckpoint ? resumeFromCheckpoint !== "false" : undefined,
    curriculumGameOffset: curriculumGameOffset ? Number(curriculumGameOffset) : undefined,
    gpuUtilizationTarget: gpuUtilizationTarget ? Number(gpuUtilizationTarget) : undefined,
    gpuCooldownMs: gpuCooldownMs ? Number(gpuCooldownMs) : undefined,
    bootstrapMaxUpdates: bootstrapMaxUpdates ? Number(bootstrapMaxUpdates) : undefined,
    bootstrapGames: bootstrapGames ? Number(bootstrapGames) : undefined,
    bootstrapBestResponseGames: bootstrapBestResponseGames ? Number(bootstrapBestResponseGames) : undefined,
    greedyBestResponseGames: greedyBestResponseGames ? Number(greedyBestResponseGames) : undefined,
    prefixStressGames: prefixStressGames ? Number(prefixStressGames) : undefined,
    prefixStressRolloutPlies: prefixStressRolloutPlies ? Number(prefixStressRolloutPlies) : undefined
  };
}

if (isMainModule(import.meta.url)) {
  await train(parseArgs(), {
    onLog: console.log,
    onProgress: (progress) => console.log(JSON.stringify(progress))
  });
}
