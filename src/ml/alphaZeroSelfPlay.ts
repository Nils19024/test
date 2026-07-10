import { encodeStateForModel } from "../core/encode";
import { applyKnownLegalMove, applyMove, getLegalMoves } from "../core/moves";
import { getModelLegalActionMask, modelActionIndexToMove, moveToModelActionIndex } from "../core/modelPerspective";
import { ACTION_COUNT, Move, PlayerIndex, QuoridorState, createInitialState, getWinner } from "../core/state";
import { shortestPathLength } from "../core/pathfinding";
import { evaluateState } from "../ai/evaluate";
import { QuoridorPolicyModel } from "./model";
import { TrainingSample } from "./replayBuffer";

type MctsNode = {
  state: QuoridorState;
  toPlay: PlayerIndex;
  prior: number;
  visits: number;
  valueSum: number;
  expanded: boolean;
  children: Map<number, MctsNode>;
};

type PendingSample = TrainingSample & { player: PlayerIndex; pathProgress?: number };

type PredictionInput = { encodedState: Float32Array; legalMask: Uint8Array; legalMoves: Move[] };

export type AlphaZeroSelfPlayStats = {
  terminalGames: number;
  drawGames: number;
  loopCount: number;
  wallMoves: number;
  pawnMoves: number;
  forwardMoves: number;
  avgGameLength: number;
  pathAdvantageMean: number | null;
};

export type AlphaZeroSelfPlayResult = {
  samples: TrainingSample[];
  games: number;
  stats: AlphaZeroSelfPlayStats;
};

export type AlphaZeroSearchResult = {
  move: Move;
  actionIndex: number;
  visits: number;
  simulations: number;
  ms: number;
  value: number;
};

type SearchPath = {
  nodes: MctsNode[];
  rootNoise?: { alpha: number; fraction: number };
};

type ActiveGame = {
  state: QuoridorState;
  samples: PendingSample[];
  visited: Map<string, number>;
  loopPenalty: Map<number, number>;
  recentPositions: Array<Array<{ row: number; col: number }>>;
  loopCount: number;
  wallMoves: number;
  pawnMoves: number;
  forwardMoves: number;
};

type SearchCache = {
  legalMoves: Map<string, Move[]>;
  legalMasks: Map<string, Uint8Array>;
  pathLengths: Map<string, number>;
};

function createNode(state: QuoridorState, prior = 1): MctsNode {
  return {
    state,
    toPlay: state.turn,
    prior,
    visits: 0,
    valueSum: 0,
    expanded: false,
    children: new Map()
  };
}

function stateKey(state: QuoridorState): string {
  const h = state.hWalls.map((row) => row.map((value) => value ? "1" : "0").join("")).join("");
  const v = state.vWalls.map((row) => row.map((value) => value ? "1" : "0").join("")).join("");
  return `${state.turn}|${state.players[0].row},${state.players[0].col},${state.players[0].wallsLeft}|${state.players[1].row},${state.players[1].col},${state.players[1].wallsLeft}|${h}|${v}`;
}

function createSearchCache(): SearchCache {
  return {
    legalMoves: new Map(),
    legalMasks: new Map(),
    pathLengths: new Map()
  };
}

function cachedLegalMoves(state: QuoridorState, cache?: SearchCache): Move[] {
  if (!cache) return getLegalMoves(state);
  const key = stateKey(state);
  const cached = cache.legalMoves.get(key);
  if (cached) return cached;
  const legalMoves = getLegalMoves(state);
  cache.legalMoves.set(key, legalMoves);
  return legalMoves;
}

function cachedLegalMask(state: QuoridorState, legalMoves: Move[] | undefined, cache?: SearchCache): Uint8Array {
  if (!cache) return getModelLegalActionMask(state);
  const key = stateKey(state);
  const cached = cache.legalMasks.get(key);
  if (cached) return cached;
  const mask = getModelLegalActionMask(state);
  cache.legalMasks.set(key, mask);
  return mask;
}

function cachedShortestPathLength(state: QuoridorState, player: PlayerIndex, cache?: SearchCache): number {
  if (!cache) return shortestPathLength(state, player);
  const key = `${stateKey(state)}|p${player}`;
  const cached = cache.pathLengths.get(key);
  if (cached !== undefined) return cached;
  const length = shortestPathLength(state, player);
  cache.pathLengths.set(key, length);
  return length;
}

function nodeValue(node: MctsNode): number {
  return node.visits > 0 ? node.valueSum / node.visits : 0;
}

function terminalValue(state: QuoridorState, perspective: PlayerIndex): number | null {
  const winner = getWinner(state);
  if (winner === null) return null;
  return winner === perspective ? 1 : -1;
}

function sampleGamma(alpha: number): number {
  // Marsaglia-Tsang, with Johnk's boost for alpha < 1.
  if (alpha < 1) return sampleGamma(alpha + 1) * Math.pow(Math.max(1e-12, Math.random()), 1 / alpha);
  const d = alpha - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x = 0;
    let y = 0;
    let r = 0;
    do {
      x = Math.random() * 2 - 1;
      y = Math.random() * 2 - 1;
      r = x * x + y * y;
    } while (r === 0 || r >= 1);
    const normal = x * Math.sqrt((-2 * Math.log(r)) / r);
    const v = Math.pow(1 + c * normal, 3);
    if (v <= 0) continue;
    const u = Math.random();
    if (u < 1 - 0.0331 * Math.pow(normal, 4)) return d * v;
    if (Math.log(u) < 0.5 * normal * normal + d * (1 - v + Math.log(v))) return d * v;
  }
}

function addDirichletNoise(node: MctsNode, alpha: number, fraction: number): void {
  const children = Array.from(node.children.values());
  if (children.length === 0 || fraction <= 0) return;
  const noise = children.map(() => sampleGamma(alpha));
  const sum = noise.reduce((total, value) => total + value, 0);
  if (!Number.isFinite(sum) || sum <= 0) return;
  children.forEach((child, index) => {
    child.prior = child.prior * (1 - fraction) + (noise[index] / sum) * fraction;
  });
}

async function expandNode(config: {
  node: MctsNode;
  model: QuoridorPolicyModel;
  rootNoise?: { alpha: number; fraction: number };
  heuristicPriorMix?: number;
  valueHeuristicMix?: number;
  cache?: SearchCache;
}): Promise<number> {
  const legalMoves = cachedLegalMoves(config.node.state, config.cache);
  const legalMask = cachedLegalMask(config.node.state, legalMoves, config.cache);
  const prediction = await config.model.predict(encodeStateForModel(config.node.state), legalMask);
  const player = config.node.state.turn;
  const opponent = player === 0 ? 1 : 0;
  const ownPathBefore = cachedShortestPathLength(config.node.state, player, config.cache);
  const oppPathBefore = cachedShortestPathLength(config.node.state, opponent, config.cache);
  let modelPriorSum = 0;
  let heuristicPriorSum = 0;
  const priors = legalMoves.map((move) => {
    const actionIndex = moveToModelActionIndex(config.node.state, move);
    const modelPrior = Number.isFinite(prediction.policy[actionIndex]) && prediction.policy[actionIndex] > 0
      ? prediction.policy[actionIndex]
      : 0;
    const heuristicPrior = Math.exp(heuristicMoveScore(config.node.state, move, ownPathBefore, oppPathBefore, config.cache));
    modelPriorSum += modelPrior;
    heuristicPriorSum += heuristicPrior;
    return { move, actionIndex, modelPrior, heuristicPrior };
  });
  const fallbackPrior = priors.length > 0 ? 1 / priors.length : 0;
  const heuristicMix = Math.max(0, Math.min(0.8, config.heuristicPriorMix ?? 0.35));
  for (const item of priors) {
    const modelPrior = modelPriorSum > 0 ? item.modelPrior / modelPriorSum : fallbackPrior;
    const heuristicPrior = heuristicPriorSum > 0 ? item.heuristicPrior / heuristicPriorSum : fallbackPrior;
    const prior = modelPrior * (1 - heuristicMix) + heuristicPrior * heuristicMix;
    config.node.children.set(item.actionIndex, createNode(applyKnownLegalMove(config.node.state, item.move), prior));
  }
  config.node.expanded = true;
  if (config.rootNoise) addDirichletNoise(config.node, config.rootNoise.alpha, config.rootNoise.fraction);
  const modelValue = Math.max(-1, Math.min(1, Number.isFinite(prediction.value) ? prediction.value : 0));
  const valueMix = Math.max(0, Math.min(1, config.valueHeuristicMix ?? 0));
  const heuristicValue = Math.tanh(evaluateState(config.node.state, config.node.toPlay) / 120);
  return modelValue * (1 - valueMix) + heuristicValue * valueMix;
}

function expandNodeWithPrediction(config: {
  node: MctsNode;
  prediction: { policy: Float32Array; value: number };
  legalMoves: Move[];
  rootNoise?: { alpha: number; fraction: number };
  heuristicPriorMix?: number;
  valueHeuristicMix?: number;
  cache?: SearchCache;
}): number {
  const player = config.node.state.turn;
  const opponent = player === 0 ? 1 : 0;
  const ownPathBefore = cachedShortestPathLength(config.node.state, player, config.cache);
  const oppPathBefore = cachedShortestPathLength(config.node.state, opponent, config.cache);
  let modelPriorSum = 0;
  let heuristicPriorSum = 0;
  const priors = config.legalMoves.map((move) => {
    const actionIndex = moveToModelActionIndex(config.node.state, move);
    const modelPrior = Number.isFinite(config.prediction.policy[actionIndex]) && config.prediction.policy[actionIndex] > 0
      ? config.prediction.policy[actionIndex]
      : 0;
    const heuristicPrior = Math.exp(heuristicMoveScore(config.node.state, move, ownPathBefore, oppPathBefore, config.cache));
    modelPriorSum += modelPrior;
    heuristicPriorSum += heuristicPrior;
    return { move, actionIndex, modelPrior, heuristicPrior };
  });
  const fallbackPrior = priors.length > 0 ? 1 / priors.length : 0;
  const heuristicMix = Math.max(0, Math.min(0.8, config.heuristicPriorMix ?? 0.35));
  for (const item of priors) {
    const modelPrior = modelPriorSum > 0 ? item.modelPrior / modelPriorSum : fallbackPrior;
    const heuristicPrior = heuristicPriorSum > 0 ? item.heuristicPrior / heuristicPriorSum : fallbackPrior;
    const prior = modelPrior * (1 - heuristicMix) + heuristicPrior * heuristicMix;
    config.node.children.set(item.actionIndex, createNode(applyKnownLegalMove(config.node.state, item.move), prior));
  }
  config.node.expanded = true;
  if (config.rootNoise) addDirichletNoise(config.node, config.rootNoise.alpha, config.rootNoise.fraction);
  const modelValue = Math.max(-1, Math.min(1, Number.isFinite(config.prediction.value) ? config.prediction.value : 0));
  const valueMix = Math.max(0, Math.min(1, config.valueHeuristicMix ?? 0));
  const heuristicValue = Math.tanh(evaluateState(config.node.state, config.node.toPlay) / 120);
  return modelValue * (1 - valueMix) + heuristicValue * valueMix;
}

function heuristicMoveScore(state: QuoridorState, move: Move, ownBefore: number, oppBefore: number, cache?: SearchCache): number {
  const player = state.turn;
  const opponent = player === 0 ? 1 : 0;
  const next = applyKnownLegalMove(state, move);
  const ownAfter = cachedShortestPathLength(next, player, cache);
  const oppAfter = cachedShortestPathLength(next, opponent, cache);
  const pathProgress = Number.isFinite(ownBefore) && Number.isFinite(ownAfter) ? ownBefore - ownAfter : 0;
  const blockProgress = Number.isFinite(oppBefore) && Number.isFinite(oppAfter) ? oppAfter - oppBefore : 0;
  const wallCost = move.type === "pawn" ? 0 : -0.1;
  return Math.max(-2, Math.min(2, pathProgress * 1.2 + blockProgress * 0.65 + wallCost));
}

function correctedMoveScore(config: {
  state: QuoridorState;
  move: Move;
  cache: SearchCache;
  recentPositions?: Array<{ row: number; col: number }>;
}): { score: number; pathProgress: number; repeatedPosition: boolean } {
  const player = config.state.turn;
  const opponent = player === 0 ? 1 : 0;
  const ownBefore = cachedShortestPathLength(config.state, player, config.cache);
  const oppBefore = cachedShortestPathLength(config.state, opponent, config.cache);
  const next = applyKnownLegalMove(config.state, config.move);
  const ownAfter = cachedShortestPathLength(next, player, config.cache);
  const oppAfter = cachedShortestPathLength(next, opponent, config.cache);
  const pathProgress = Number.isFinite(ownBefore) && Number.isFinite(ownAfter) ? ownBefore - ownAfter : 0;
  const blockProgress = Number.isFinite(oppBefore) && Number.isFinite(oppAfter) ? oppAfter - oppBefore : 0;
  const afterPosition = next.players[player];
  const repeatedPosition = config.move.type === "pawn" && (config.recentPositions ?? []).some((position) => position.row === afterPosition.row && position.col === afterPosition.col);
  const terminal = getWinner(next) === player ? 3 : 0;
  let score = terminal;
  if (config.move.type === "pawn") {
    score += pathProgress * 1.7;
    if (pathProgress < 0) score -= 1.2;
    if (pathProgress === 0) score -= 0.15;
    if (repeatedPosition) score -= 2.2;
  } else {
    score += blockProgress * 1.1 - 0.3;
    if (blockProgress <= 0) score -= 0.85;
    if (Number.isFinite(ownBefore) && ownBefore <= 3) score -= 0.5;
  }
  return { score: Math.max(-4, Math.min(4, score)), pathProgress, repeatedPosition };
}

function correctPolicyTarget(config: {
  state: QuoridorState;
  policyTarget: Float32Array;
  legalMoves: Move[];
  cache: SearchCache;
  recentPositions: Array<{ row: number; col: number }>;
}): Float32Array {
  const corrected = new Float32Array(ACTION_COUNT);
  const scored = config.legalMoves.map((move) => {
    const actionIndex = moveToModelActionIndex(config.state, move);
    const quality = correctedMoveScore({ state: config.state, move, cache: config.cache, recentPositions: config.recentPositions });
    return { actionIndex, move, quality };
  });
  if (scored.length === 0) return corrected;
  const maxScore = Math.max(...scored.map((item) => item.quality.score));
  let sum = 0;
  for (const item of scored) {
    const original = Math.max(0, Number.isFinite(config.policyTarget[item.actionIndex]) ? config.policyTarget[item.actionIndex] : 0);
    const heuristic = Math.exp((item.quality.score - maxScore) * 1.15);
    const badPawn = item.quality.repeatedPosition || item.quality.pathProgress < 0;
    const originalWeight = badPawn ? original * 0.08 : original;
    const weight = originalWeight * 0.55 + heuristic * 0.45;
    corrected[item.actionIndex] = weight;
    sum += weight;
  }
  if (!Number.isFinite(sum) || sum <= 0) {
    const best = scored.reduce((a, b) => b.quality.score > a.quality.score ? b : a);
    corrected[best.actionIndex] = 1;
    return corrected;
  }
  for (const item of scored) corrected[item.actionIndex] /= sum;
  keepUsefulPawnMass(scored, corrected);
  return corrected;
}

function keepUsefulPawnMass(
  scored: Array<{ actionIndex: number; move: Move; quality: { pathProgress: number; score: number } }>,
  policy: Float32Array,
  minPawnMass = 0.35
): void {
  const usefulPawns = scored.filter((item) => item.move.type === "pawn" && item.quality.pathProgress > 0);
  if (usefulPawns.length === 0) return;
  const currentPawnMass = usefulPawns.reduce((sum, item) => sum + Math.max(0, policy[item.actionIndex] ?? 0), 0);
  if (currentPawnMass >= minPawnMass) return;

  const usefulPawnActions = new Set(usefulPawns.map((item) => item.actionIndex));
  const scale = (1 - minPawnMass) / Math.max(1e-6, 1 - currentPawnMass);
  for (let action = 0; action < policy.length; action++) {
    if (!usefulPawnActions.has(action)) policy[action] *= scale;
  }
  const extraMass = minPawnMass - currentPawnMass;
  const minScore = Math.min(...usefulPawns.map((item) => item.quality.score));
  const weights = usefulPawns.map((item) => Math.exp(item.quality.score - minScore));
  const weightSum = weights.reduce((sum, value) => sum + value, 0);
  usefulPawns.forEach((item, index) => {
    policy[item.actionIndex] += extraMass * (weights[index] / Math.max(1e-6, weightSum));
  });
}

function selectChild(node: MctsNode, cpuct: number): MctsNode {
  let best: MctsNode | null = null;
  let bestScore = -Infinity;
  const parentVisits = Math.max(1, node.visits);
  for (const child of node.children.values()) {
    const q = -nodeValue(child);
    const u = cpuct * child.prior * Math.sqrt(parentVisits) / (1 + child.visits);
    const score = q + u;
    if (score > bestScore) {
      bestScore = score;
      best = child;
    }
  }
  if (!best) throw new Error("MCTS node has no children");
  return best;
}

async function simulate(config: {
  node: MctsNode;
  model: QuoridorPolicyModel;
  cpuct: number;
  rootNoise?: { alpha: number; fraction: number };
  heuristicPriorMix?: number;
  valueHeuristicMix?: number;
  cache?: SearchCache;
}): Promise<number> {
  const terminal = terminalValue(config.node.state, config.node.toPlay);
  if (terminal !== null) {
    config.node.visits++;
    config.node.valueSum += terminal;
    return terminal;
  }

  if (!config.node.expanded) {
    const value = await expandNode(config);
    config.node.visits++;
    config.node.valueSum += value;
    return value;
  }

  const child = selectChild(config.node, config.cpuct);
  const childValue = await simulate({
    node: child,
    model: config.model,
    cpuct: config.cpuct,
      heuristicPriorMix: config.heuristicPriorMix,
      valueHeuristicMix: config.valueHeuristicMix,
      cache: config.cache
  });
  const value = -childValue;
  config.node.visits++;
  config.node.valueSum += value;
  return value;
}

function selectSearchPath(root: MctsNode, cpuct: number, rootNoise?: { alpha: number; fraction: number }): SearchPath {
  const nodes = [root];
  let node = root;
  for (;;) {
    const terminal = terminalValue(node.state, node.toPlay);
    if (terminal !== null || !node.expanded) return { nodes, rootNoise: node === root ? rootNoise : undefined };
    node = selectChild(node, cpuct);
    nodes.push(node);
  }
}

function backupPath(path: MctsNode[], leafValue: number): void {
  let value = leafValue;
  for (let i = path.length - 1; i >= 0; i--) {
    path[i].visits++;
    path[i].valueSum += value;
    value = -value;
  }
}

async function predictMany(model: QuoridorPolicyModel, inputs: PredictionInput[]): Promise<Array<{ policy: Float32Array; value: number }>> {
  if (inputs.length === 0) return [];
  if (model.predictBatch) return model.predictBatch(inputs);
  return Promise.all(inputs.map((input) => model.predict(input.encodedState, input.legalMask)));
}

async function runBatchedMcts(config: {
  roots: MctsNode[];
  model: QuoridorPolicyModel;
  simulations: number;
  cpuct: number;
  dirichletAlpha: number;
  dirichletFraction: number;
  heuristicPriorMix?: number;
  valueHeuristicMix?: number;
  cache?: SearchCache;
}): Promise<void> {
  for (let simulation = 0; simulation < config.simulations; simulation++) {
    const pending: Array<{ path: SearchPath; input: PredictionInput }> = [];
    for (const root of config.roots) {
      const path = selectSearchPath(
        root,
        config.cpuct,
        simulation === 0 ? { alpha: config.dirichletAlpha, fraction: config.dirichletFraction } : undefined
      );
      const leaf = path.nodes[path.nodes.length - 1];
      const terminal = terminalValue(leaf.state, leaf.toPlay);
      if (terminal !== null) {
        backupPath(path.nodes, terminal);
        continue;
      }
      const legalMoves = cachedLegalMoves(leaf.state, config.cache);
      pending.push({
        path,
        input: {
          encodedState: encodeStateForModel(leaf.state),
          legalMask: cachedLegalMask(leaf.state, legalMoves, config.cache),
          legalMoves
        }
      });
    }
    const predictions = await predictMany(config.model, pending.map((item) => item.input));
    predictions.forEach((prediction, index) => {
      const path = pending[index].path;
      const leaf = path.nodes[path.nodes.length - 1];
      const value = expandNodeWithPrediction({
        node: leaf,
        prediction,
        legalMoves: pending[index].input.legalMoves,
        rootNoise: path.rootNoise,
        heuristicPriorMix: config.heuristicPriorMix,
        valueHeuristicMix: config.valueHeuristicMix,
        cache: config.cache
      });
      backupPath(path.nodes, value);
    });
    if (simulation % 16 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

function policyFromVisits(root: MctsNode, temperature: number, smoothing = 0.04): Float32Array {
  const policy = new Float32Array(ACTION_COUNT);
  const entries = Array.from(root.children.entries());
  if (entries.length === 0) return policy;
  if (temperature <= 0.05) {
    const [bestAction] = entries.reduce((best, current) => current[1].visits > best[1].visits ? current : best);
    policy[bestAction] = 1;
  } else {
    let sum = 0;
    for (const [actionIndex, child] of entries) {
      const value = Math.pow(Math.max(0, child.visits), 1 / temperature);
      policy[actionIndex] = value;
      sum += value;
    }
    if (sum <= 0) {
      const uniform = 1 / entries.length;
      for (const [actionIndex] of entries) policy[actionIndex] = uniform;
    } else {
      for (const [actionIndex] of entries) policy[actionIndex] /= sum;
    }
  }
  const clippedSmoothing = Math.max(0, Math.min(0.25, smoothing));
  if (clippedSmoothing > 0 && entries.length > 1) {
    const uniform = clippedSmoothing / entries.length;
    for (const [actionIndex] of entries) policy[actionIndex] = policy[actionIndex] * (1 - clippedSmoothing) + uniform;
  }
  return policy;
}

function sampleFromPolicy(policy: Float32Array): number {
  let pick = Math.random();
  let fallback = -1;
  for (let i = 0; i < policy.length; i++) {
    if (policy[i] > 0 && fallback < 0) fallback = i;
    pick -= policy[i];
    if (pick <= 0 && policy[i] > 0) return i;
  }
  if (fallback >= 0) return fallback;
  throw new Error("Cannot sample from empty policy");
}

export async function searchBestMoveMcts(config: {
  state: QuoridorState;
  model: QuoridorPolicyModel;
  simulations: number;
  cpuct: number;
  heuristicPriorMix?: number;
  timeMs?: number;
  valueHeuristicMix?: number;
}): Promise<AlphaZeroSearchResult> {
  const started = performance.now();
  const deadline = started + Math.max(1, config.timeMs ?? Number.POSITIVE_INFINITY);
  const cache = createSearchCache();
  const root = createNode(config.state);
  let completed = 0;
  for (let simulation = 0; simulation < config.simulations; simulation++) {
    if (performance.now() >= deadline) break;
    await simulate({
      node: root,
      model: config.model,
      cpuct: config.cpuct,
      heuristicPriorMix: config.heuristicPriorMix,
      valueHeuristicMix: config.valueHeuristicMix,
      cache
    });
    completed++;
    if (simulation % 16 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
  }

  if (!root.expanded) {
    await simulate({
      node: root,
      model: config.model,
      cpuct: config.cpuct,
      heuristicPriorMix: config.heuristicPriorMix,
      valueHeuristicMix: config.valueHeuristicMix,
      cache
    });
    completed++;
  }

  const children = Array.from(root.children.entries());
  if (children.length === 0) throw new Error("MCTS root has no children");
  const [actionIndex, child] = children.reduce((best, current) => current[1].visits > best[1].visits ? current : best);
  return {
    move: modelActionIndexToMove(config.state, actionIndex),
    actionIndex,
    visits: child.visits,
    simulations: completed,
    ms: performance.now() - started,
    value: -nodeValue(child)
  };
}

export async function runAlphaZeroSelfPlayGame(config: {
  model: QuoridorPolicyModel;
  maxMovesPerGame: number;
  simulations: number;
  cpuct: number;
  dirichletAlpha: number;
  dirichletFraction: number;
  temperatureMoves: number;
  minTemperature?: number;
  policyTargetSmoothing?: number;
  heuristicPriorMix?: number;
  drawValueWeight?: number;
}): Promise<TrainingSample[]> {
  return (await runBatchedAlphaZeroSelfPlayGames({ ...config, games: 1 })).samples;
}

export async function runBatchedAlphaZeroSelfPlayGames(config: {
  model: QuoridorPolicyModel;
  games: number;
  maxMovesPerGame: number;
  simulations: number;
  cpuct: number;
  dirichletAlpha: number;
  dirichletFraction: number;
  temperatureMoves: number;
  minTemperature?: number;
  policyTargetSmoothing?: number;
  heuristicPriorMix?: number;
  drawValueWeight?: number;
}): Promise<AlphaZeroSelfPlayResult> {
  const cache = createSearchCache();
  const games: ActiveGame[] = Array.from({ length: Math.max(1, config.games) }, () => ({
    state: createInitialState(),
    samples: [],
    visited: new Map(),
    loopPenalty: new Map(),
    recentPositions: [[], []],
    loopCount: 0,
    wallMoves: 0,
    pawnMoves: 0,
    forwardMoves: 0
  }));

  for (let ply = 0; ply < config.maxMovesPerGame; ply++) {
    const active = games.filter((game) => getWinner(game.state) === null);
    if (active.length === 0) break;
    const roots = active.map((game) => createNode(game.state));
    await runBatchedMcts({
      roots,
      model: config.model,
      simulations: config.simulations,
      cpuct: config.cpuct,
      dirichletAlpha: config.dirichletAlpha,
      dirichletFraction: config.dirichletFraction,
      heuristicPriorMix: config.heuristicPriorMix,
      cache
    });
    active.forEach((game, index) => {
      const rawPolicyTarget = policyFromVisits(
        roots[index],
        ply < config.temperatureMoves ? 1 : Math.max(0.35, config.minTemperature ?? 0.5),
        config.policyTargetSmoothing ?? 0.04
      );
      const legalMoves = cachedLegalMoves(game.state, cache);
      const policyTarget = correctPolicyTarget({
        state: game.state,
        policyTarget: rawPolicyTarget,
        legalMoves,
        cache,
        recentPositions: game.recentPositions[game.state.turn]
      });
      const actionIndex = sampleFromPolicy(policyTarget);
      const move = modelActionIndexToMove(game.state, actionIndex);
      const player = game.state.turn;
      const opponent = player === 0 ? 1 : 0;
      const ownPath = cachedShortestPathLength(game.state, player, cache);
      const oppPath = cachedShortestPathLength(game.state, opponent, cache);
      const pathAdvantage = Number.isFinite(ownPath) && Number.isFinite(oppPath) ? oppPath - ownPath : 0;
      const nextState = applyKnownLegalMove(game.state, move);
      const ownPathAfter = cachedShortestPathLength(nextState, player, cache);
      const pathProgress = Number.isFinite(ownPath) && Number.isFinite(ownPathAfter) ? ownPath - ownPathAfter : 0;
      const afterPosition = nextState.players[player];
      const repeatedPosition = move.type === "pawn" && game.recentPositions[player].some((position) => position.row === afterPosition.row && position.col === afterPosition.col);
      const previousVisits = game.visited.get(stateKey(game.state)) ?? 0;
      if (previousVisits > 0) {
        game.loopPenalty.set(game.samples.length, Math.min(0.35, 0.12 * previousVisits));
        game.loopCount++;
      }
      if (repeatedPosition) {
        const existingPenalty = game.loopPenalty.get(game.samples.length) ?? 0;
        game.loopPenalty.set(game.samples.length, Math.min(0.55, existingPenalty + 0.22));
        game.loopCount++;
      }
      game.visited.set(stateKey(game.state), previousVisits + 1);
      if (move.type === "pawn") {
        game.pawnMoves++;
        if (pathProgress > 0) game.forwardMoves++;
      }
      else game.wallMoves++;
      game.samples.push({
        encodedState: encodeStateForModel(game.state),
        legalMask: cachedLegalMask(game.state, undefined, cache),
        actionIndex,
        policyTarget,
        player,
        source: "alphazero",
        wallMove: move.type !== "pawn",
        pathAdvantage,
        pathProgress
      });
      game.recentPositions[player].push({ row: afterPosition.row, col: afterPosition.col });
      game.recentPositions[player] = game.recentPositions[player].slice(-5);
      game.state = nextState;
    });
  }

  let terminalGames = 0;
  let drawGames = 0;
  let loopCount = 0;
  let wallMoves = 0;
  let pawnMoves = 0;
  let forwardMoves = 0;
  let totalGameLength = 0;
  let pathAdvantageSum = 0;
  let pathAdvantageCount = 0;
  const samples = games.flatMap((game) => {
    const winner = getWinner(game.state);
    if (winner === null) drawGames++;
    else terminalGames++;
    loopCount += game.loopPenalty.size;
    wallMoves += game.wallMoves;
    pawnMoves += game.pawnMoves;
    forwardMoves += game.forwardMoves;
    totalGameLength += game.samples.length;
    return game.samples.map((sample, index) => {
      const loopPenalty = game.loopPenalty.get(index) ?? 0;
      if (typeof sample.pathAdvantage === "number" && Number.isFinite(sample.pathAdvantage)) {
        pathAdvantageSum += sample.pathAdvantage;
        pathAdvantageCount++;
      }
      const baseReward = winner === null ? unfinishedValue(game.state, sample.player, config.drawValueWeight ?? 0.45) : winner === sample.player ? 1 : -1;
      const progressReward = Math.max(-0.08, Math.min(0.08, (sample.pathProgress ?? 0) * 0.06));
      const reward = Math.max(-1, Math.min(1, baseReward + progressReward - loopPenalty));
      return {
        encodedState: sample.encodedState,
        legalMask: sample.legalMask,
        actionIndex: sample.actionIndex,
        policyTarget: sample.policyTarget,
        reward,
        priority: 1 + Math.abs(reward) + loopPenalty + Math.max(0, sample.pathProgress ?? 0) * 0.2 + (winner === null ? 0 : 0.8),
        source: sample.source,
        terminal: winner !== null,
        wallMove: sample.wallMove,
        loopPenalty,
        pathAdvantage: sample.pathAdvantage
      };
    });
  });

  return {
    games: games.length,
    samples,
    stats: {
      terminalGames,
      drawGames,
      loopCount,
      wallMoves,
      pawnMoves,
      forwardMoves,
      avgGameLength: totalGameLength / Math.max(1, games.length),
      pathAdvantageMean: pathAdvantageCount > 0 ? pathAdvantageSum / pathAdvantageCount : null
    }
  };
}

async function runSequentialAlphaZeroSelfPlayGame(config: {
  model: QuoridorPolicyModel;
  maxMovesPerGame: number;
  simulations: number;
  cpuct: number;
  dirichletAlpha: number;
  dirichletFraction: number;
  temperatureMoves: number;
  minTemperature?: number;
  policyTargetSmoothing?: number;
  heuristicPriorMix?: number;
  drawValueWeight?: number;
}): Promise<TrainingSample[]> {
  let state = createInitialState();
  const samples: PendingSample[] = [];

  for (let ply = 0; ply < config.maxMovesPerGame && getWinner(state) === null; ply++) {
    const root = createNode(state);
    for (let simulation = 0; simulation < config.simulations; simulation++) {
      await simulate({
        node: root,
        model: config.model,
        cpuct: config.cpuct,
        heuristicPriorMix: config.heuristicPriorMix,
        rootNoise: simulation === 0
          ? { alpha: config.dirichletAlpha, fraction: config.dirichletFraction }
          : undefined
      });
      if (simulation % 16 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
    }
    const policyTarget = policyFromVisits(root, ply < config.temperatureMoves ? 1 : Math.max(0.35, config.minTemperature ?? 0.5), config.policyTargetSmoothing ?? 0.04);
    const actionIndex = sampleFromPolicy(policyTarget);
    samples.push({
      encodedState: encodeStateForModel(state),
      legalMask: getModelLegalActionMask(state),
      actionIndex,
      policyTarget,
      player: state.turn,
      source: "alphazero",
      wallMove: modelActionIndexToMove(state, actionIndex).type !== "pawn"
    });
    state = applyMove(state, modelActionIndexToMove(state, actionIndex));
  }

  const winner = getWinner(state);
  return samples.map((sample) => {
    const reward = winner === null ? unfinishedValue(state, sample.player, config.drawValueWeight ?? 0.45) : winner === sample.player ? 1 : -1;
    return {
      encodedState: sample.encodedState,
      legalMask: sample.legalMask,
      actionIndex: sample.actionIndex,
      policyTarget: sample.policyTarget,
      reward,
      priority: 1 + Math.abs(reward),
      source: sample.source,
      terminal: winner !== null,
      wallMove: sample.wallMove
    };
  });
}

function unfinishedValue(finalState: QuoridorState, player: PlayerIndex, weight: number): number {
  const opponent = player === 0 ? 1 : 0;
  const ownPath = shortestPathLength(finalState, player);
  const oppPath = shortestPathLength(finalState, opponent);
  if (!Number.isFinite(ownPath) || !Number.isFinite(oppPath)) return -0.1;
  return Math.max(-0.6, Math.min(0.6, Math.tanh((oppPath - ownPath) / 4) * Math.max(0, Math.min(1, weight))));
}
