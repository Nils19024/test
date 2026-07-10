import { encodeStateForModel } from "../core/encode";
import { applyKnownLegalMove, getLegalMoves } from "../core/moves";
import { getModelLegalActionMask, modelActionIndexToMove, moveToModelActionIndex } from "../core/modelPerspective";
import { ACTION_COUNT, Move, PlayerIndex, QuoridorState, createInitialState, getWinner } from "../core/state";
import { shortestPathLength } from "../core/pathfinding";
import { RandomAgent } from "../ai/randomAgent";
import { QuoridorPolicyModel, chooseBestLegalAction, chooseSampledLegalAction } from "./model";
import { TrainingSample } from "./replayBuffer";
import { searchBestMove } from "../ai/alphabeta";
import { evaluateState } from "../ai/evaluate";
import { GreedyShortestPathAgent } from "../ai/greedyAgent";
import { selectBestMoveAgainstGreedy } from "../ai/greedyBestResponse";

export { selectBestMoveAgainstGreedy };

type PendingSample = TrainingSample & { player: PlayerIndex; shapedReward?: number; trainable?: boolean };
type PlayerPosition = { row: number; col: number };

export async function runSelfPlayGame(config: {
  model?: QuoridorPolicyModel;
  maxMovesPerGame: number;
  temperature: number;
}): Promise<TrainingSample[]> {
  let state = createInitialState();
  const samples: PendingSample[] = [];

  for (let ply = 0; ply < config.maxMovesPerGame && getWinner(state) === null; ply++) {
    const legalMask = getModelLegalActionMask(state);
    let actionIndex: number;
    let move;
    if (config.model) {
      const prediction = await config.model.predict(encodeStateForModel(state), legalMask);
      actionIndex = chooseSampledLegalAction(prediction.policy, legalMask, config.temperature);
      move = modelActionIndexToMove(state, actionIndex);
    } else {
      move = await RandomAgent.selectMove(state);
      actionIndex = moveToModelActionIndex(state, move);
    }
    samples.push({
      encodedState: encodeStateForModel(state),
      legalMask,
      actionIndex,
      player: state.turn,
      source: "selfplay",
      wallMove: move.type !== "pawn"
    });
    state = applyKnownLegalMove(state, move);
    if (ply % 20 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
  }

  const winner = getWinner(state);
  return samples.filter((sample) => sample.trainable !== false).map((sample) => ({
    encodedState: sample.encodedState,
    legalMask: sample.legalMask,
    actionIndex: sample.actionIndex,
    reward: winner === null ? -0.1 : winner === sample.player ? 1 : -1,
    source: sample.source,
    terminal: winner !== null,
    wallMove: sample.wallMove
  }));
}

function softmaxPolicy(scores: Array<{ actionIndex: number; score: number }>, temperature: number): Float32Array {
  const policy = new Float32Array(ACTION_COUNT);
  const finiteScores = scores.map((item) => ({
    actionIndex: item.actionIndex,
    score: Number.isFinite(item.score) ? item.score : item.score > 0 ? 100000 : -100000
  }));
  const max = Math.max(...finiteScores.map((item) => item.score));
  let sum = 0;
  for (const item of finiteScores) {
    const value = Math.exp((item.score - max) / Math.max(0.05, temperature));
    policy[item.actionIndex] = value;
    sum += value;
  }
  if (sum === 0) return policy;
  for (const item of finiteScores) policy[item.actionIndex] /= sum;
  return policy;
}

function sampleFromPolicy(policy: Float32Array, fallbackActionIndex: number): number {
  let pick = Math.random();
  let fallback = -1;
  for (let i = 0; i < policy.length; i++) {
    if (policy[i] > 0 && fallback < 0) fallback = i;
    pick -= policy[i];
    if (pick <= 0 && policy[i] > 0) return i;
  }
  if (fallback >= 0) return fallback;
  return fallbackActionIndex;
}

function searchMoveCorrectionScore(state: QuoridorState, move: Move, avoidPositions: PlayerPosition[]): number {
  const player = state.turn;
  const opponent = player === 0 ? 1 : 0;
  const ownBefore = shortestPathLength(state, player);
  const oppBefore = shortestPathLength(state, opponent);
  const next = applyKnownLegalMove(state, move);
  const ownAfter = shortestPathLength(next, player);
  const oppAfter = shortestPathLength(next, opponent);
  const pathProgress = Number.isFinite(ownBefore) && Number.isFinite(ownAfter) ? ownBefore - ownAfter : 0;
  const blockProgress = Number.isFinite(oppBefore) && Number.isFinite(oppAfter) ? oppAfter - oppBefore : 0;
  const afterPosition = next.players[player];
  const repeatedPosition = move.type === "pawn" && avoidPositions.some((position) => position.row === afterPosition.row && position.col === afterPosition.col);

  if (getWinner(next) === player) return 1000;
  if (move.type === "pawn") {
    return pathProgress * 1.8 - (pathProgress < 0 ? 1.4 : 0) - (pathProgress === 0 ? 0.2 : 0) - (repeatedPosition ? 2.6 : 0);
  }
  return blockProgress * 1.0 - 0.3 - (blockProgress <= 0 ? 0.8 : 0);
}

async function searchImprovedAction(config: {
  state: QuoridorState;
  model: QuoridorPolicyModel;
  topK: number;
  searchTimeMs: number;
  searchMaxDepth: number;
  temperature: number;
  explorationNoise?: number;
  avoidPositions?: PlayerPosition[];
}): Promise<{ actionIndex: number; policyTarget: Float32Array }> {
  const legalMask = getModelLegalActionMask(config.state);
  const prediction = await config.model.predict(encodeStateForModel(config.state), legalMask);
  const legalMoves = getLegalMoves(config.state);
  const avoidPositions = config.avoidPositions ?? [];
  const rankedByModel = Array.from(prediction.policy.entries())
    .filter(([index]) => legalMask[index])
    .sort((a, b) => b[1] - a[1])
    .slice(0, config.topK)
    .map(([index]) => index);
  const tactical = legalMoves
    .filter((move) => move.type === "pawn")
    .sort((a, b) => searchMoveCorrectionScore(config.state, b, avoidPositions) - searchMoveCorrectionScore(config.state, a, avoidPositions))
    .map((move) => moveToModelActionIndex(config.state, move))
    .slice(0, Math.min(4, config.topK));
  const greedyCandidate = moveToModelActionIndex(config.state, await GreedyShortestPathAgent.selectMove(config.state));
  const candidates = Array.from(new Set([...tactical, greedyCandidate, ...rankedByModel])).slice(0, Math.max(1, config.topK + tactical.length + 1));
  const scored: Array<{ actionIndex: number; score: number }> = [];

  for (const actionIndex of candidates) {
    const move = modelActionIndexToMove(config.state, actionIndex);
    const next = applyKnownLegalMove(config.state, move);
    let score: number;
    if (getWinner(next) === config.state.turn) score = 100000;
    else {
      const result = searchBestMove(next, {
        timeMs: config.searchTimeMs,
        maxDepth: config.searchMaxDepth,
        mode: "candidate"
      });
      score = -result.score
        + evaluateState(next, config.state.turn) * 0.05
        + (prediction.policy[actionIndex] ?? 0) * 10
        + searchMoveCorrectionScore(config.state, move, avoidPositions) * 36;
    }
    scored.push({ actionIndex, score });
    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  if (scored.length === 0) {
    const actionIndex = chooseBestLegalAction(prediction.policy, legalMask);
    const policyTarget = new Float32Array(ACTION_COUNT);
    policyTarget[actionIndex] = 1;
    return { actionIndex, policyTarget };
  }

  const candidateActions = scored.map((item) => item.actionIndex);
  const policyTarget = keepUsefulPawnMovesAlive(
    config.state,
    mixExplorationNoise(softmaxPolicy(scored, config.temperature), candidateActions, config.explorationNoise ?? 0),
    candidateActions
  );
  const best = scored.slice().sort((a, b) => b.score - a.score)[0].actionIndex;
  if (policyTarget.every((value) => value === 0 || !Number.isFinite(value))) policyTarget[best] = 1;
  return { actionIndex: sampleFromPolicy(policyTarget, best), policyTarget };
}

function mixExplorationNoise(policy: Float32Array, candidateActions: number[], noise: number): Float32Array {
  const amount = Math.max(0, Math.min(0.5, noise));
  if (amount <= 0 || candidateActions.length === 0) return policy;
  const mixed = new Float32Array(policy);
  const randomWeights = candidateActions.map(() => -Math.log(Math.max(1e-6, Math.random())));
  const randomSum = randomWeights.reduce((sum, value) => sum + value, 0);
  for (let i = 0; i < mixed.length; i++) mixed[i] *= 1 - amount;
  for (let i = 0; i < candidateActions.length; i++) {
    mixed[candidateActions[i]] += amount * (randomWeights[i] / Math.max(1e-6, randomSum));
  }
  return mixed;
}

function keepUsefulPawnMovesAlive(state: QuoridorState, policy: Float32Array, candidateActions: number[], minPawnMass = 0.35): Float32Array {
  const player = state.turn;
  const ownBefore = shortestPathLength(state, player);
  if (!Number.isFinite(ownBefore)) return policy;
  const usefulPawns = candidateActions
    .map((actionIndex) => ({ actionIndex, move: modelActionIndexToMove(state, actionIndex) }))
    .filter(({ move }) => move.type === "pawn")
    .map(({ actionIndex, move }) => {
      const next = applyKnownLegalMove(state, move);
      const ownAfter = shortestPathLength(next, player);
      return {
        actionIndex,
        progress: Number.isFinite(ownAfter) ? ownBefore - ownAfter : 0,
        terminal: getWinner(next) === player
      };
    })
    .filter((item) => item.terminal || item.progress > 0);
  if (usefulPawns.length === 0) return policy;
  const currentPawnMass = usefulPawns.reduce((sum, item) => sum + Math.max(0, policy[item.actionIndex] ?? 0), 0);
  if (currentPawnMass >= minPawnMass) return policy;

  const usefulPawnActions = new Set(usefulPawns.map((item) => item.actionIndex));
  const adjusted = new Float32Array(policy);
  const scale = (1 - minPawnMass) / Math.max(1e-6, 1 - currentPawnMass);
  for (let action = 0; action < adjusted.length; action++) {
    if (!usefulPawnActions.has(action)) adjusted[action] *= scale;
  }
  const extraMass = minPawnMass - currentPawnMass;
  const weights = usefulPawns.map((item) => item.terminal ? 4 : Math.max(1, item.progress));
  const weightSum = weights.reduce((sum, value) => sum + value, 0);
  usefulPawns.forEach((item, index) => {
    adjusted[item.actionIndex] += extraMass * (weights[index] / Math.max(1e-6, weightSum));
  });
  return adjusted;
}

export async function runSearchImprovedSelfPlayGame(config: {
  model: QuoridorPolicyModel;
  maxMovesPerGame: number;
  temperature: number;
  searchTopK: number;
  searchTimeMs: number;
  searchMaxDepth: number;
  opponentMix: "self" | "greedy" | "random" | "alphabeta";
  modelPlayer?: PlayerIndex;
  trainOpponentMoves?: boolean;
  rewardShaping?: boolean;
  explorationNoise?: number;
}): Promise<TrainingSample[]> {
  let state = createInitialState();
  const samples: PendingSample[] = [];
  const recentPositions: PlayerPosition[][] = [[], []];
  const modelPlayer = config.modelPlayer ?? 0;

  for (let ply = 0; ply < config.maxMovesPerGame && getWinner(state) === null; ply++) {
    const legalMask = getModelLegalActionMask(state);
    const encodedState = encodeStateForModel(state);
    const actingPlayer = state.turn;
    const ownBefore = shortestPathLength(state, actingPlayer);
    const oppBefore = shortestPathLength(state, actingPlayer === 0 ? 1 : 0);
    let actionIndex: number;
    let policyTarget: Float32Array | undefined;
    const useOpponent = config.opponentMix !== "self" && state.turn !== modelPlayer;

    if (useOpponent && config.opponentMix === "greedy") {
      actionIndex = moveToModelActionIndex(state, await GreedyShortestPathAgent.selectMove(state));
    } else if (useOpponent && config.opponentMix === "random") {
      actionIndex = moveToModelActionIndex(state, await RandomAgent.selectMove(state));
    } else if (useOpponent && config.opponentMix === "alphabeta") {
      actionIndex = moveToModelActionIndex(state, searchBestMove(state, {
        timeMs: Math.max(8, config.searchTimeMs * 3),
        maxDepth: Math.max(3, config.searchMaxDepth + 1),
        mode: "candidate"
      }).move);
    } else {
      const improved = await searchImprovedAction({
        state,
        model: config.model,
        topK: config.searchTopK,
        searchTimeMs: config.searchTimeMs,
        searchMaxDepth: config.searchMaxDepth,
        temperature: config.temperature,
        explorationNoise: config.explorationNoise,
        avoidPositions: recentPositions[actingPlayer]
      });
      actionIndex = improved.actionIndex;
      policyTarget = improved.policyTarget;
    }

    const move = modelActionIndexToMove(state, actionIndex);
    const next = applyKnownLegalMove(state, move);
    const ownAfter = shortestPathLength(next, actingPlayer);
    const oppAfter = shortestPathLength(next, actingPlayer === 0 ? 1 : 0);
    const pathProgress = Number.isFinite(ownBefore) && Number.isFinite(ownAfter) ? ownBefore - ownAfter : 0;
    const blockProgress = Number.isFinite(oppBefore) && Number.isFinite(oppAfter) ? oppAfter - oppBefore : 0;
    const pathAdvantage = Number.isFinite(ownAfter) && Number.isFinite(oppAfter) ? oppAfter - ownAfter : 0;
    const afterPosition = next.players[actingPlayer];
    const repeatedPosition = move.type === "pawn" && recentPositions[actingPlayer].some((position) => position.row === afterPosition.row && position.col === afterPosition.col);
    const wallCost = move.type === "pawn" ? 0 : -0.02;
    const repetitionPenalty = repeatedPosition ? -0.18 : 0;
    const shapedReward = config.rewardShaping === false
      ? 0
      : Math.max(-0.45, Math.min(0.45, pathProgress * 0.12 + blockProgress * 0.06 + wallCost + repetitionPenalty));

    samples.push({
      encodedState,
      legalMask,
      actionIndex,
      policyTarget,
      player: actingPlayer,
      shapedReward,
      trainable: !useOpponent || config.trainOpponentMoves === true,
      source: "search",
      wallMove: move.type !== "pawn",
      pathAdvantage,
      pathProgress
    });
    recentPositions[actingPlayer].push({ row: afterPosition.row, col: afterPosition.col });
    recentPositions[actingPlayer] = recentPositions[actingPlayer].slice(-6);
    state = next;
  }

  const winner = getWinner(state);
  return samples.filter((sample) => sample.trainable !== false).map((sample) => {
    const terminalReward = winner === null ? -0.05 : winner === sample.player ? 1 : -1;
    const reward = Math.max(-1, Math.min(1, terminalReward * 0.65 + (sample.shapedReward ?? 0)));
    const priority = 1 + Math.min(1, Math.abs(reward)) + (winner === null ? 0 : 0.8) + Math.min(0.4, Math.abs(sample.shapedReward ?? 0) * 2);
    return {
      encodedState: sample.encodedState,
      legalMask: sample.legalMask,
      actionIndex: sample.actionIndex,
      policyTarget: sample.policyTarget,
      reward,
      priority,
      source: sample.source,
      terminal: winner !== null,
      wallMove: sample.wallMove,
      pathAdvantage: sample.pathAdvantage,
      pathProgress: sample.pathProgress
    };
  });
}

export async function generateGreedyImitationSamples(config: {
  games: number;
  maxMovesPerGame: number;
  randomMoveRate?: number;
}): Promise<TrainingSample[]> {
  const samples: TrainingSample[] = [];
  const randomMoveRate = Math.max(0, Math.min(1, config.randomMoveRate ?? 0.2));

  for (let game = 0; game < config.games; game++) {
    let state = createInitialState();
    for (let ply = 0; ply < config.maxMovesPerGame && getWinner(state) === null; ply++) {
      const actingPlayer = state.turn;
      const legalMask = getModelLegalActionMask(state);
      const ownBefore = shortestPathLength(state, actingPlayer);
      const greedyMove = await GreedyShortestPathAgent.selectMove(state);
      const greedyAction = moveToModelActionIndex(state, greedyMove);
      const target = new Float32Array(ACTION_COUNT);
      target[greedyAction] = 1;
      const nextGreedy = applyKnownLegalMove(state, greedyMove);
      const ownAfterGreedy = shortestPathLength(nextGreedy, actingPlayer);
      const oppAfterGreedy = shortestPathLength(nextGreedy, actingPlayer === 0 ? 1 : 0);
      const pathProgress = Number.isFinite(ownBefore) && Number.isFinite(ownAfterGreedy) ? ownBefore - ownAfterGreedy : 0;
      const pathAdvantage = Number.isFinite(ownAfterGreedy) && Number.isFinite(oppAfterGreedy) ? oppAfterGreedy - ownAfterGreedy : 0;

      samples.push({
        encodedState: encodeStateForModel(state),
        legalMask,
        actionIndex: greedyAction,
        policyTarget: target,
        reward: Math.max(-1, Math.min(1, pathAdvantage / 10)),
        priority: 2,
        source: "search",
        terminal: getWinner(nextGreedy) !== null,
        wallMove: greedyMove.type !== "pawn",
        pathAdvantage,
        pathProgress
      });

      const rolloutMove = Math.random() < randomMoveRate ? await RandomAgent.selectMove(state) : greedyMove;
      state = applyKnownLegalMove(state, rolloutMove);
      if (ply % 20 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  return samples;
}

export async function generateGreedyBestResponseSamples(config: {
  games: number;
  maxMovesPerGame: number;
  teacherTimeMs: number;
  teacherMaxDepth: number;
  explorationRate?: number;
}): Promise<TrainingSample[]> {
  const samples: TrainingSample[] = [];
  const explorationRate = Math.max(0, Math.min(0.35, config.explorationRate ?? 0.08));

  for (let game = 0; game < config.games; game++) {
    let state = createInitialState();
    const pending: PendingSample[] = [];
    const recentPositions: PlayerPosition[][] = [[], []];
    const modelPlayer: PlayerIndex = game % 2 === 0 ? 0 : 1;
    for (let ply = 0; ply < config.maxMovesPerGame && getWinner(state) === null; ply++) {
      const actingPlayer = state.turn;
      const legalMask = getModelLegalActionMask(state);
      const ownBefore = shortestPathLength(state, actingPlayer);
      const oppBefore = shortestPathLength(state, actingPlayer === 0 ? 1 : 0);
      const isModelSide = actingPlayer === modelPlayer;
      const targetMove = isModelSide
        ? await selectBestMoveAgainstGreedy(state, recentPositions[actingPlayer])
        : await GreedyShortestPathAgent.selectMove(state);
      const move = isModelSide && Math.random() < explorationRate
        ? await RandomAgent.selectMove(state)
        : targetMove;
      const actionIndex = moveToModelActionIndex(state, targetMove);
      const next = applyKnownLegalMove(state, move);
      const labeledNext = applyKnownLegalMove(state, targetMove);
      const ownAfter = shortestPathLength(labeledNext, actingPlayer);
      const oppAfter = shortestPathLength(labeledNext, actingPlayer === 0 ? 1 : 0);
      const pathProgress = Number.isFinite(ownBefore) && Number.isFinite(ownAfter) ? ownBefore - ownAfter : 0;
      const blockProgress = Number.isFinite(oppBefore) && Number.isFinite(oppAfter) ? oppAfter - oppBefore : 0;
      const pathAdvantage = Number.isFinite(ownAfter) && Number.isFinite(oppAfter) ? oppAfter - ownAfter : 0;
      const target = new Float32Array(ACTION_COUNT);
      target[actionIndex] = 1;

      pending.push({
        encodedState: encodeStateForModel(state),
        legalMask,
        actionIndex,
        policyTarget: target,
        player: actingPlayer,
        trainable: isModelSide,
        shapedReward: Math.max(-0.45, Math.min(0.45, pathProgress * 0.14 + blockProgress * 0.08 + pathAdvantage * 0.025)),
        source: "search",
        wallMove: targetMove.type !== "pawn",
        pathAdvantage,
        pathProgress
      });

      state = next;
      recentPositions[actingPlayer].push({ row: next.players[actingPlayer].row, col: next.players[actingPlayer].col });
      recentPositions[actingPlayer] = recentPositions[actingPlayer].slice(-6);
      if (ply % 12 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
    }

    const winner = getWinner(state);
    for (const sample of pending.filter((item) => item.trainable !== false)) {
      const terminalReward = winner === null ? 0 : winner === sample.player ? 1 : -1;
      const reward = Math.max(-1, Math.min(1, terminalReward * 0.45 + (sample.shapedReward ?? 0)));
      samples.push({
        encodedState: sample.encodedState,
        legalMask: sample.legalMask,
        actionIndex: sample.actionIndex,
        policyTarget: sample.policyTarget,
        reward,
        priority: 6 + Math.max(0, reward) + (sample.terminal ? 1 : 0),
        source: sample.source,
        terminal: winner !== null,
        wallMove: sample.wallMove,
        pathAdvantage: sample.pathAdvantage,
        pathProgress: sample.pathProgress
      });
    }
  }

  return samples;
}
