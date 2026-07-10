import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { createInitialState } from "../core/state";
import { encodeStateForModel } from "../core/encode";
import { applyKnownLegalMove, getLegalActionMask, getLegalMoves } from "../core/moves";
import { shortestPathLength } from "../core/pathfinding";
import { chooseSampledLegalAction, createPolicyValueModel, MODEL_INPUT_SIZE, QuoridorPolicyModel, TfjsQuoridorPolicyModel } from "../ml/model";
import { samplesToTensors } from "../ml/losses";
import { TrainingSample } from "../ml/replayBuffer";
import { loadModelFromDir, saveModelForBrowser } from "../ml/exportModel";
import { runAlphaZeroSelfPlayGame, searchBestMoveMcts } from "../ml/alphaZeroSelfPlay";
import { modelActionIndexToMove } from "../core/modelPerspective";
import { generateGreedyBestResponseSamples, runSearchImprovedSelfPlayGame, selectBestMoveAgainstGreedy } from "../ml/selfPlay";
import { createPrefixStressState, generatePrefixStressTeacherSamples } from "../ml/prefixStressSamples";
import { moveToActionIndex } from "../core/actions";
import { runMatchup } from "../benchmark/benchmark";
import { GreedyShortestPathAgent } from "../ai/greedyAgent";
import { alphaZeroCurriculumSource, defaultTrainingConfig } from "../ml/train";

function usefulPawnMass(sample: TrainingSample): number {
  if (!sample.policyTarget) return 0;
  const state = createInitialState();
  let mass = 0;
  for (let action = 0; action < sample.policyTarget.length; action++) {
    if (!sample.legalMask[action]) continue;
    const move = modelActionIndexToMove(state, action);
    if (move.type === "pawn" && move.row === 7 && move.col === 4) mass += sample.policyTarget[action];
  }
  return mass;
}

const wallBiasedModel: QuoridorPolicyModel = {
  async predict(_encodedState, legalMask) {
    const policy = new Float32Array(legalMask.length);
    let sum = 0;
    const state = createInitialState();
    for (let action = 0; action < legalMask.length; action++) {
      if (!legalMask[action]) continue;
      const move = modelActionIndexToMove(state, action);
      policy[action] = move.type === "pawn" ? 0.001 : 1;
      sum += policy[action];
    }
    for (let action = 0; action < policy.length; action++) policy[action] /= Math.max(1e-6, sum);
    return { policy, value: 0 };
  }
};

describe("training utilities", () => {
  it("encodes states and samples only legal actions", () => {
    const state = createInitialState();
    const encoded = encodeStateForModel(state);
    const mask = getLegalActionMask(state);
    expect(encoded).toHaveLength(MODEL_INPUT_SIZE);
    const policy = new Float32Array(209).fill(1);
    const action = chooseSampledLegalAction(policy, mask, 1);
    expect(mask[action]).toBe(1);
  });

  it("runs a tiny training step and saves/loads model", async () => {
    const state = createInitialState();
    const mask = getLegalActionMask(state);
    const actionIndex = mask.findIndex((value) => value === 1);
    const sample: TrainingSample = {
      encodedState: encodeStateForModel(state),
      legalMask: mask,
      actionIndex,
      reward: 0
    };
    const model = createPolicyValueModel(0.001);
    const tensors = samplesToTensors([sample]);
    await model.fit(tensors.xs, { policy: tensors.policyY, value: tensors.valueY }, { epochs: 1, verbose: 0 });
    tensors.xs.dispose();
    tensors.policyY.dispose();
    tensors.valueY.dispose();

    const dir = await mkdtemp(path.join(tmpdir(), "quoridor-model-"));
    await saveModelForBrowser(model, dir);
    const loaded = await loadModelFromDir(dir);
    const prediction = await new TfjsQuoridorPolicyModel(loaded).predict(sample.encodedState, mask);
    expect(prediction.policy).toHaveLength(209);
    expect(Number.isFinite(prediction.value)).toBe(true);
    await rm(dir, { recursive: true, force: true });
  }, 20000);

  it("generates alpha zero self-play samples with visit-count policies", async () => {
    const model = new TfjsQuoridorPolicyModel(createPolicyValueModel(0.001));
    const samples = await runAlphaZeroSelfPlayGame({
      model,
      maxMovesPerGame: 2,
      simulations: 2,
      cpuct: 1.5,
      dirichletAlpha: 0.3,
      dirichletFraction: 0.25,
      temperatureMoves: 2
    });
    expect(samples.length).toBeGreaterThan(0);
    expect(samples[0].policyTarget).toBeDefined();
    expect(samples[0].legalMask[samples[0].actionIndex]).toBe(1);
    expect(usefulPawnMass(samples[0])).toBeGreaterThanOrEqual(0.34);
  }, 20000);

  it("returns a legal move from runtime MCTS search", async () => {
    const state = createInitialState();
    const model: QuoridorPolicyModel = {
      async predict(_encodedState, legalMask) {
        const policy = new Float32Array(legalMask.length);
        let legalCount = 0;
        for (let action = 0; action < legalMask.length; action++) if (legalMask[action]) legalCount++;
        for (let action = 0; action < legalMask.length; action++) {
          if (legalMask[action]) policy[action] = 1 / Math.max(1, legalCount);
        }
        return { policy, value: 0 };
      }
    };
    const result = await searchBestMoveMcts({
      state,
      model,
      simulations: 4,
      cpuct: 1.25,
      timeMs: 50,
      heuristicPriorMix: 0.4,
      valueHeuristicMix: 0.6
    });
    expect(getLegalMoves(state).some((move) => moveToActionIndex(move) === moveToActionIndex(result.move))).toBe(true);
    expect(result.simulations).toBeGreaterThan(0);
  });

  it("keeps useful pawn moves in curriculum targets even when the model is wall-biased", async () => {
    const samples = await runSearchImprovedSelfPlayGame({
      model: wallBiasedModel,
      maxMovesPerGame: 1,
      temperature: 0.2,
      searchTopK: 6,
      searchTimeMs: 1,
      searchMaxDepth: 1,
      opponentMix: "random",
      modelPlayer: 0,
      trainOpponentMoves: false,
      rewardShaping: true,
      explorationNoise: 0
    });
    expect(samples.length).toBe(1);
    expect(samples[0].policyTarget).toBeDefined();
    expect(usefulPawnMass(samples[0])).toBeGreaterThanOrEqual(0.34);
  }, 20000);

  it("generates legal greedy best-response samples", async () => {
    const samples = await generateGreedyBestResponseSamples({
      games: 1,
      maxMovesPerGame: 8,
      teacherTimeMs: 2,
      teacherMaxDepth: 1,
      explorationRate: 0
    });
    expect(samples.length).toBeGreaterThan(0);
    for (const sample of samples) {
      expect(sample.legalMask[sample.actionIndex]).toBe(1);
      expect(sample.policyTarget?.[sample.actionIndex]).toBeGreaterThan(0);
      expect(sample.source).toBe("search");
    }
  }, 20000);

  it("keeps greedy best-response as the label when rollout exploration is enabled", async () => {
    const initialState = createInitialState();
    const expectedMove = await selectBestMoveAgainstGreedy(initialState);
    const samples = await generateGreedyBestResponseSamples({
      games: 1,
      maxMovesPerGame: 2,
      teacherTimeMs: 2,
      teacherMaxDepth: 1,
      explorationRate: 1
    });
    expect(samples.length).toBeGreaterThan(0);
    expect(samples[0].actionIndex).toBe(moveToActionIndex(expectedMove));
    const expectedProgress = shortestPathLength(initialState, 0) - shortestPathLength(applyKnownLegalMove(initialState, expectedMove), 0);
    expect(samples[0].pathProgress).toBe(expectedProgress);
  }, 20000);

  it("selects a legal move against greedy", async () => {
    const state = createInitialState();
    const move = await selectBestMoveAgainstGreedy(state);
    const legal = new Set(getLegalMoves(state).map(moveToActionIndex));
    expect(legal.has(moveToActionIndex(move))).toBe(true);
  }, 20000);

  it("uses greedy best-response curriculum from the early phase", () => {
    const firstTwenty = Array.from({ length: 20 }, (_, index) => alphaZeroCurriculumSource(index + 1, {
      ...defaultTrainingConfig,
      alphaZeroWarmupGames: 0,
      alphaZeroParallelGames: 2
    }));
    expect(firstTwenty[0]).toBe("greedy");
    expect(firstTwenty.every((source) => source === "greedy")).toBe(true);
  });

  it("starts the default alphazero curriculum with greedy best-response instead of teacher warmup", () => {
    const firstTwenty = Array.from({ length: 20 }, (_, index) => alphaZeroCurriculumSource(index + 1, defaultTrainingConfig));
    expect(firstTwenty[0]).toBe("greedy");
    expect(firstTwenty).not.toContain("teacher");
    expect(firstTwenty.every((source) => source === "greedy")).toBe(true);
    expect(defaultTrainingConfig.persistentReplay).toBe(false);
  });

  it("adds alphabeta opponents after the greedy foundation phase", () => {
    const sources = Array.from({ length: 80 }, (_, index) => alphaZeroCurriculumSource(121 + index, defaultTrainingConfig));
    expect(sources).toContain("greedy");
    expect(sources).toContain("alphabeta");
    expect(sources).toContain("random");
  });

  it("creates deterministic prefix stress states", () => {
    const a = createPrefixStressState(2);
    const b = createPrefixStressState(2);
    expect(a).toEqual(b);
    expect(a.moveNumber).toBeGreaterThanOrEqual(6);
  });

  it("generates legal prefix stress teacher samples", async () => {
    const samples = await generatePrefixStressTeacherSamples({
      games: 1,
      rolloutPlies: 2,
      teacherTimeMs: 5,
      teacherMaxDepth: 2
    });
    expect(samples.length).toBeGreaterThan(0);
    for (const sample of samples) {
      expect(sample.legalMask[sample.actionIndex]).toBe(1);
      expect(sample.policyTarget?.[sample.actionIndex]).toBe(1);
      expect(sample.source).toBe("search");
    }
  }, 20000);

  it("generates mirrored prefix stress samples for both player roles", async () => {
    const samples = await generatePrefixStressTeacherSamples({
      games: 1,
      rolloutPlies: 2,
      teacherTimeMs: 5,
      teacherMaxDepth: 2
    });
    const oneSidedSamples = await generatePrefixStressTeacherSamples({
      games: 1,
      rolloutPlies: 2,
      teacherTimeMs: 5,
      teacherMaxDepth: 2,
      mirrored: false
    });
    expect(samples.length).toBeGreaterThan(oneSidedSamples.length);
    expect(samples.every((sample) => sample.legalMask[sample.actionIndex] === 1)).toBe(true);
  }, 20000);

  it("generates focused prefix stress samples from an explicit game list", async () => {
    const samples = await generatePrefixStressTeacherSamples({
      games: 0,
      gamesList: [0, 0, 3],
      rolloutPlies: 1,
      teacherTimeMs: 5,
      teacherMaxDepth: 2
    });
    expect(samples.length).toBeGreaterThanOrEqual(3);
    expect(samples.every((sample) => sample.legalMask[sample.actionIndex] === 1)).toBe(true);
  }, 20000);

  it("generates focused prefix stress samples with best-response teacher", async () => {
    const samples = await generatePrefixStressTeacherSamples({
      games: 0,
      gamesList: [0, 3],
      rolloutPlies: 1,
      teacherTimeMs: 5,
      teacherMaxDepth: 2,
      teacherMode: "best-response"
    });
    expect(samples.length).toBeGreaterThan(0);
    expect(samples.every((sample) => sample.legalMask[sample.actionIndex] === 1)).toBe(true);
  }, 20000);

  it("generates soft best-response prefix targets", async () => {
    const samples = await generatePrefixStressTeacherSamples({
      games: 0,
      gamesList: [0],
      rolloutPlies: 1,
      teacherTimeMs: 5,
      teacherMaxDepth: 2,
      teacherMode: "best-response",
      bestResponseMix: 0.25
    });
    expect(samples.length).toBeGreaterThan(0);
    for (const sample of samples) {
      const policy = sample.policyTarget;
      expect(policy).toBeDefined();
      expect(sample.legalMask[sample.actionIndex]).toBe(1);
      const policySum = Array.from(policy ?? []).reduce((sum, value) => sum + value, 0);
      expect(policySum).toBeCloseTo(1, 5);
      expect(policy?.[sample.actionIndex]).toBeGreaterThan(0);
    }
  }, 20000);

  it("uses a strong enough default best-response bootstrap for GPU training", () => {
    expect(defaultTrainingConfig.bootstrapMaxUpdates).toBeGreaterThanOrEqual(24);
    expect(defaultTrainingConfig.bootstrapBestResponseGames).toBeGreaterThanOrEqual(4);
    expect(defaultTrainingConfig.greedyBestResponseGames).toBeGreaterThanOrEqual(2);
  });

  it("lets greedy best-response samples reach terminal wins when given enough moves", async () => {
    const samples = await generateGreedyBestResponseSamples({
      games: 1,
      maxMovesPerGame: 80,
      teacherTimeMs: 5,
      teacherMaxDepth: 2,
      explorationRate: 0
    });
    expect(samples.length).toBeGreaterThan(0);
    expect(samples.some((sample) => sample.terminal === true)).toBe(true);
    const rewardMean = samples.reduce((sum, sample) => sum + (sample.reward ?? 0), 0) / samples.length;
    expect(rewardMean).toBeGreaterThan(0.2);
  }, 30000);

  it("greedy best-response teacher beats greedy in a short matchup", async () => {
    const bestResponseAgent = {
      name: "GreedyBestResponseTeacher",
      selectMove: selectBestMoveAgainstGreedy
    };
    const result = await runMatchup({
      agentA: bestResponseAgent,
      agentB: GreedyShortestPathAgent,
      games: 2,
      maxMovesPerGame: 100
    });
    expect(result.illegalMoves).toBe(0);
    expect(result.winrateA).toBeGreaterThanOrEqual(0.5);
  }, 30000);
});
