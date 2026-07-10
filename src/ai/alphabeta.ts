import { moveToActionIndex } from "../core/actions";
import { applyKnownLegalMove, getLegalMoves } from "../core/moves";
import { moveToModelActionIndex } from "../core/modelPerspective";
import { BOARD_SIZE, Move, QuoridorState, cloneState, getWinner, otherPlayer } from "../core/state";
import { distanceMapToGoal, shortestPathLength } from "../core/pathfinding";
import { canMoveBetween } from "../core/walls";
import { evaluateState, moveOrderingScore } from "./evaluate";

export type SearchOptions = {
  timeMs: number;
  maxDepth: number;
  mode?: "full" | "candidate";
  perspective?: 0 | 1;
  rootPolicyPrior?: Float32Array;
  rootTopK?: number;
  tacticalReplies?: "auto" | "always" | "off";
};

export type SearchResult = {
  move: Move;
  actionIndex: number;
  score: number;
  depth: number;
  nodes: number;
  ms: number;
  principalVariation: Move[];
  tacticalRepliesUsed?: boolean;
};

type TTEntry = {
  depth: number;
  score: number;
  bound: "exact" | "lower" | "upper";
};

function stateKey(state: QuoridorState): string {
  const h = state.hWalls.map((row) => row.map(Number).join("")).join("");
  const v = state.vWalls.map((row) => row.map(Number).join("")).join("");
  const p0 = state.players[0];
  const p1 = state.players[1];
  return `${state.turn}|${p0.row},${p0.col},${p0.wallsLeft}|${p1.row},${p1.col},${p1.wallsLeft}|${h}|${v}`;
}

function uniqueMoves(moves: Move[]): Move[] {
  const seen = new Set<number>();
  const result: Move[] = [];
  for (const move of moves) {
    const index = moveToActionIndex(move);
    if (seen.has(index)) continue;
    seen.add(index);
    result.push(move);
  }
  return result;
}

const PATH_DIRS = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1]
] as const;

function routeBlockingWalls(state: QuoridorState, player: 0 | 1, legalWallMoves: Move[]): Move[] {
  const legalByIndex = new Map(legalWallMoves.map((move) => [moveToActionIndex(move), move]));
  const distances = distanceMapToGoal(state, player);
  let row = state.players[player].row;
  let col = state.players[player].col;
  const result: Move[] = [];

  while (distances[row * BOARD_SIZE + col] > 0) {
    const currentDistance = distances[row * BOARD_SIZE + col];
    const next = PATH_DIRS
      .map(([dr, dc]) => ({ row: row + dr, col: col + dc }))
      .find((position) => canMoveBetween(state, row, col, position.row, position.col)
        && distances[position.row * BOARD_SIZE + position.col] === currentDistance - 1);
    if (!next) break;

    const blockers: Move[] = row !== next.row
      ? [
          { type: "hwall", row: Math.min(row, next.row), col: col - 1 },
          { type: "hwall", row: Math.min(row, next.row), col }
        ]
      : [
          { type: "vwall", row: row - 1, col: Math.min(col, next.col) },
          { type: "vwall", row, col: Math.min(col, next.col) }
        ];
    for (const blocker of blockers) {
      if (blocker.row < 0 || blocker.row >= BOARD_SIZE - 1 || blocker.col < 0 || blocker.col >= BOARD_SIZE - 1) continue;
      const legal = legalByIndex.get(moveToActionIndex(blocker));
      if (legal) result.push(legal);
    }
    row = next.row;
    col = next.col;
  }
  return uniqueMoves(result);
}

function wallBlockScore(
  state: QuoridorState,
  move: Move,
  ownBefore = shortestPathLength(state, state.turn),
  oppBefore = shortestPathLength(state, otherPlayer(state.turn))
): number {
  if (move.type === "pawn") return -Infinity;
  const player = state.turn;
  const opponent = otherPlayer(player);
  const next = applyKnownLegalMove(state, move);
  const ownAfter = shortestPathLength(next, player);
  const oppAfter = shortestPathLength(next, opponent);
  const blockProgress = Number.isFinite(oppBefore) && Number.isFinite(oppAfter) ? oppAfter - oppBefore : 0;
  const ownDamage = Number.isFinite(ownBefore) && Number.isFinite(ownAfter) ? ownAfter - ownBefore : 0;
  return blockProgress * 3 - ownDamage * 2 - (ownBefore <= 2 ? 1 : 0);
}

function hasSevereWallThreat(state: QuoridorState, minimumPathIncrease = 4): boolean {
  const threatenedPlayer = state.turn;
  const opponent = otherPlayer(threatenedPlayer);
  if (state.players[opponent].wallsLeft <= 0) return false;
  const before = shortestPathLength(state, threatenedPlayer);
  if (!Number.isFinite(before)) return false;
  const opponentState = cloneState(state);
  opponentState.turn = opponent;
  for (const move of getLegalMoves(opponentState)) {
    if (move.type === "pawn") continue;
    const after = shortestPathLength(applyKnownLegalMove(opponentState, move), threatenedPlayer);
    if (Number.isFinite(after) && after - before >= minimumPathIncrease) return true;
  }
  return false;
}

function candidateMoves(
  state: QuoridorState,
  mode: "full" | "candidate",
  rootPolicyPrior?: Float32Array,
  rootTopK = 0,
  includeTacticalWalls = false
): Move[] {
  const all = getLegalMoves(state);
  if (mode === "full") return all;
  const pawnMoves = all.filter((move) => move.type === "pawn");
  const allWallMoves = all.filter((move) => move.type !== "pawn");
  const wallMoves = all
    .filter((move) => move.type !== "pawn")
    .sort((a, b) => moveOrderingScore(state, b) - moveOrderingScore(state, a))
    .slice(0, includeTacticalWalls ? 8 : rootPolicyPrior ? 10 : 18);
  const includeBlockingWalls = rootPolicyPrior !== undefined || includeTacticalWalls;
  const ownBefore = includeBlockingWalls ? shortestPathLength(state, state.turn) : 0;
  const oppBefore = includeBlockingWalls ? shortestPathLength(state, otherPlayer(state.turn)) : 0;
  const blockingPool = includeTacticalWalls
    ? uniqueMoves([
        ...routeBlockingWalls(state, state.turn, allWallMoves),
        ...routeBlockingWalls(state, otherPlayer(state.turn), allWallMoves)
      ])
    : allWallMoves;
  const blockingWallMoves = includeBlockingWalls
    ? blockingPool
      .slice()
      .sort((a, b) => wallBlockScore(state, b, ownBefore, oppBefore) - wallBlockScore(state, a, ownBefore, oppBefore))
      .slice(0, includeTacticalWalls ? 8 : 6)
    : [];
  const policyMoves = rootPolicyPrior && rootTopK > 0
    ? all
      .slice()
      .sort((a, b) => (rootPolicyPrior[moveToModelActionIndex(state, b)] ?? 0) - (rootPolicyPrior[moveToModelActionIndex(state, a)] ?? 0))
      .slice(0, rootTopK)
    : [];
  return uniqueMoves([...pawnMoves, ...policyMoves, ...blockingWallMoves, ...wallMoves]);
}

function orderedMoves(state: QuoridorState, moves: Move[], perspective: 0 | 1, rootPolicyPrior?: Float32Array): Move[] {
  return moves.slice().sort((a, b) => {
    const priorA = rootPolicyPrior?.[moveToModelActionIndex(state, a)] ?? 0;
    const priorB = rootPolicyPrior?.[moveToModelActionIndex(state, b)] ?? 0;
    return (moveOrderingScore(state, b, perspective) + priorB * 80)
      - (moveOrderingScore(state, a, perspective) + priorA * 80);
  });
}

export function searchBestMove(state: QuoridorState, options: SearchOptions): SearchResult {
  const started = performance.now();
  const deadline = started + Math.max(1, options.timeMs);
  const perspective = options.perspective ?? state.turn;
  const tt = new Map<string, TTEntry>();
  const moveCache = new Map<string, Move[]>();
  const evalCache = new Map<string, number>();
  const useTacticalReplies = options.mode !== "full" && (
    options.tacticalReplies === "always"
    || (options.tacticalReplies !== "off" && hasSevereWallThreat(state))
  );
  let nodes = 0;

  function cachedEvaluate(current: QuoridorState): number {
    const key = `${stateKey(current)}|eval|${perspective}`;
    const hit = evalCache.get(key);
    if (hit !== undefined) return hit;
    const score = evaluateState(current, perspective);
    evalCache.set(key, score);
    return score;
  }

  function cachedCandidateMoves(current: QuoridorState, withRootPolicy = false, includeTacticalWalls = false): Move[] {
    const key = `${stateKey(current)}|${options.mode ?? "candidate"}|${withRootPolicy ? "root" : includeTacticalWalls ? "tactical" : "node"}`;
    const hit = moveCache.get(key);
    if (hit) return hit;
    const moves = candidateMoves(
      current,
      options.mode ?? "candidate",
      withRootPolicy ? options.rootPolicyPrior : undefined,
      withRootPolicy ? options.rootTopK : 0,
      includeTacticalWalls
    );
    moveCache.set(key, moves);
    return moves;
  }

  let bestMove = cachedCandidateMoves(state, true)[0];
  if (!bestMove) throw new Error("No legal move");
  let bestScore = cachedEvaluate(applyKnownLegalMove(state, bestMove));
  let completedDepth = 0;
  let pv: Move[] = [bestMove];

  const timedOut = () => performance.now() >= deadline;

  function minimax(current: QuoridorState, depth: number, alpha: number, beta: number, line: Move[], plyFromRoot: number): number {
    nodes++;
    const winner = getWinner(current);
    if (depth === 0 || winner !== null || timedOut()) return cachedEvaluate(current);
    const key = stateKey(current);
    const hit = tt.get(key);
    if (hit && hit.depth >= depth) {
      if (hit.bound === "exact") return hit.score;
      if (hit.bound === "lower") alpha = Math.max(alpha, hit.score);
      else beta = Math.min(beta, hit.score);
      if (alpha >= beta) return hit.score;
    }
    const originalAlpha = alpha;
    const originalBeta = beta;

    const moves = cachedCandidateMoves(current, false, useTacticalReplies && plyFromRoot === 1).slice().sort(
      (a, b) => moveOrderingScore(current, b, current.turn) - moveOrderingScore(current, a, current.turn)
    );

    if (current.turn === perspective) {
      let value = -Infinity;
      for (const move of moves) {
        const next = applyKnownLegalMove(current, move);
        const childLine: Move[] = [];
        const score = minimax(next, depth - 1, alpha, beta, childLine, plyFromRoot + 1);
        if (score > value) {
          value = score;
          line.length = 0;
          line.push(move, ...childLine);
        }
        alpha = Math.max(alpha, value);
        if (alpha >= beta || timedOut()) break;
      }
      if (!timedOut()) {
        const bound = value <= originalAlpha ? "upper" : value >= originalBeta ? "lower" : "exact";
        tt.set(key, { depth, score: value, bound });
      }
      return value;
    }

    let value = Infinity;
    for (const move of moves) {
      const next = applyKnownLegalMove(current, move);
      const childLine: Move[] = [];
      const score = minimax(next, depth - 1, alpha, beta, childLine, plyFromRoot + 1);
      if (score < value) {
        value = score;
        line.length = 0;
        line.push(move, ...childLine);
      }
      beta = Math.min(beta, value);
      if (alpha >= beta || timedOut()) break;
    }
    if (!timedOut()) {
      const bound = value <= originalAlpha ? "upper" : value >= originalBeta ? "lower" : "exact";
      tt.set(key, { depth, score: value, bound });
    }
    return value;
  }

  for (let depth = 1; depth <= options.maxDepth; depth++) {
    if (timedOut()) break;
    let depthBest = bestMove;
    let depthBestScore = -Infinity;
    let depthPv: Move[] = [];
    const moves = orderedMoves(
      state,
      cachedCandidateMoves(state, true),
      perspective,
      options.rootPolicyPrior
    );
    for (const move of moves) {
      if (timedOut()) break;
      const line: Move[] = [];
      const score = minimax(applyKnownLegalMove(state, move), depth - 1, -Infinity, Infinity, line, 1);
      if (score > depthBestScore) {
        depthBestScore = score;
        depthBest = move;
        depthPv = [move, ...line];
      }
    }
    if (!timedOut()) {
      bestMove = depthBest;
      bestScore = depthBestScore;
      completedDepth = depth;
      pv = depthPv;
    }
  }

  return {
    move: bestMove,
    actionIndex: moveToActionIndex(bestMove),
    score: bestScore,
    depth: completedDepth,
    nodes: Math.max(1, nodes),
    ms: performance.now() - started,
    principalVariation: pv,
    tacticalRepliesUsed: useTacticalReplies
  };
}

export function alphaBetaAgent(timeMs: number, maxDepth: number, mode: "full" | "candidate" = "candidate") {
  return {
    name: `AlphaBeta ${timeMs}ms`,
    async selectMove(state: QuoridorState) {
      return searchBestMove(state, { timeMs, maxDepth, mode }).move;
    }
  };
}

export const DIFFICULTIES = {
  easy: { timeMs: 10, maxDepth: 3 },
  medium: { timeMs: 80, maxDepth: 4 },
  hard: { timeMs: 200, maxDepth: 6 },
  pro: { timeMs: 650, maxDepth: 7 },
  elite: { timeMs: 650, maxDepth: 7, maxTimeMs: 9500, criticalTimeMs: 1200, criticalMaxDepth: 10, mode: "candidate" }
} as const;
