import { moveToActionIndex } from "../core/actions";
import { encodeStateForModel } from "../core/encode";
import { applyKnownLegalMove, applyMove, getLegalMoves } from "../core/moves";
import { getModelLegalActionMask, modelActionIndexToMove } from "../core/modelPerspective";
import { Move, QuoridorState, getWinner, otherPlayer } from "../core/state";
import { shortestPathLength } from "../core/pathfinding";
import { chooseBestLegalAction, QuoridorPolicyModel } from "../ml/model";
import { SearchOptions, SearchResult, searchBestMove } from "./alphabeta";
import { evaluateState } from "./evaluate";
import { GreedyShortestPathAgent } from "./greedyAgent";

function immediateWinningMove(state: QuoridorState): Move | null {
  for (const move of getLegalMoves(state)) {
    if (getWinner(applyKnownLegalMove(state, move)) === state.turn) return move;
  }
  return null;
}

function allowsImmediateOpponentWin(next: QuoridorState, player: 0 | 1): boolean {
  const opponent = otherPlayer(player);
  if (next.turn !== opponent) return false;
  for (const reply of getLegalMoves(next)) {
    if (getWinner(applyKnownLegalMove(next, reply)) === opponent) return true;
  }
  return false;
}

function bestImmediateThreatBlock(state: QuoridorState): Move | null {
  let bestMove: Move | null = null;
  let bestScore = -Infinity;
  for (const move of getLegalMoves(state)) {
    const next = applyKnownLegalMove(state, move);
    if (allowsImmediateOpponentWin(next, state.turn)) continue;
    const score = evaluateState(next, state.turn);
    if (score > bestScore) {
      bestScore = score;
      bestMove = move;
    }
  }
  return bestMove;
}

function shouldRunTrapRecovery(state: QuoridorState, searchScore: number): boolean {
  return searchScore <= -70 && isPathTrouble(state);
}

function isPathTrouble(state: QuoridorState): boolean {
  const player = state.turn;
  const opponent = otherPlayer(player);
  const myPath = shortestPathLength(state, player);
  const oppPath = shortestPathLength(state, opponent);
  if (!Number.isFinite(myPath) || !Number.isFinite(oppPath)) return false;
  const pathDeficit = myPath - oppPath;
  return pathDeficit >= 5 || (myPath >= 17 && pathDeficit >= 3);
}

function sameMove(a: Move, b: Move): boolean {
  return moveToActionIndex(a) === moveToActionIndex(b);
}

function trapRecoveryCandidates(state: QuoridorState, requiredMove: Move): Move[] {
  const legalMoves = getLegalMoves(state);
  const candidates: Move[] = [requiredMove];
  const pawnMoves = legalMoves.filter((move) => move.type === "pawn");
  for (const move of pawnMoves) {
    if (!candidates.some((candidate) => sameMove(candidate, move))) candidates.push(move);
  }
  return candidates;
}

async function scoreAfterGreedyReply(state: QuoridorState, move: Move): Promise<number> {
  const player = state.turn;
  const next = applyKnownLegalMove(state, move);
  if (getWinner(next) === player) return 100000;
  const reply = await GreedyShortestPathAgent.selectMove(next);
  return evaluateState(applyKnownLegalMove(next, reply), player);
}

function worstImmediateReplyScore(state: QuoridorState, move: Move): number {
  const player = state.turn;
  const next = applyKnownLegalMove(state, move);
  if (getWinner(next) === player) return 100000;
  let worstScore = Infinity;
  for (const reply of getLegalMoves(next)) {
    const score = evaluateState(applyKnownLegalMove(next, reply), player);
    if (score < worstScore) worstScore = score;
  }
  return worstScore === Infinity ? evaluateState(next, player) : worstScore;
}

async function trapRecoveredResult(state: QuoridorState, result: SearchResult): Promise<SearchResult> {
  if (!shouldRunTrapRecovery(state, result.score)) return result;
  let bestMove = result.move;
  let bestScore = await scoreAfterGreedyReply(state, result.move);
  const originalScore = bestScore;
  for (const move of trapRecoveryCandidates(state, result.move)) {
    const score = await scoreAfterGreedyReply(state, move);
    if (score > bestScore) {
      bestScore = score;
      bestMove = move;
    }
  }
  if (sameMove(bestMove, result.move) || bestScore < originalScore + 45) return result;
  return {
    ...result,
    move: bestMove,
    actionIndex: moveToActionIndex(bestMove),
    score: bestScore,
    principalVariation: [bestMove]
  };
}

export async function searchBestMoveHybrid(
  state: QuoridorState,
  model: QuoridorPolicyModel,
  options: {
    timeMs: number;
    maxDepth: number;
    topKFromModel: number;
    modelConfidenceThreshold?: number;
    searchMode?: "candidate" | "full";
    troubleSearchMode?: "candidate" | "full";
    troubleMaxDepth?: number;
    forceSearchInTrouble?: boolean;
    maxTimeMs?: number;
    criticalTimeMs?: number;
    criticalMaxDepth?: number;
    runSearch?: (state: QuoridorState, options: SearchOptions) => Promise<SearchResult>;
  }
): Promise<SearchResult> {
  const started = performance.now();
  const deadline = started + Math.max(1, options.timeMs);
  const runSearch = options.runSearch
    ?? ((searchState: QuoridorState, searchOptions: SearchOptions) => Promise.resolve(searchBestMove(searchState, searchOptions)));
  const legalMask = getModelLegalActionMask(state);
  const prediction = await model.predict(encodeStateForModel(state), legalMask);
  const modelAction = chooseBestLegalAction(prediction.policy, legalMask);
  const modelMove = modelActionIndexToMove(state, modelAction);
  const modelConfidence = prediction.policy[modelAction] ?? 0;
  const winningMove = immediateWinningMove(state);
  if (winningMove) {
    return {
      move: winningMove,
      actionIndex: moveToActionIndex(winningMove),
      score: 100000,
      depth: 0,
      nodes: 1,
      ms: performance.now() - started,
      principalVariation: [winningMove]
    };
  }
  const confidenceThreshold = options.modelConfidenceThreshold ?? 0.85;
  const modelNext = applyMove(state, modelMove);
  const modelPathRegression = shortestPathLength(modelNext, state.turn) > shortestPathLength(state, state.turn);
  const noWallsRemain = state.players[0].wallsLeft === 0 && state.players[1].wallsLeft === 0;
  const unsafePathRegression = modelPathRegression && (noWallsRemain || isPathTrouble(state));
  const modelAllowsImmediateLoss = allowsImmediateOpponentWin(modelNext, state.turn);
  if (modelAllowsImmediateLoss) {
    const block = bestImmediateThreatBlock(state);
    if (block) {
      return {
        move: block,
        actionIndex: moveToActionIndex(block),
        score: evaluateState(applyKnownLegalMove(state, block), state.turn),
        depth: 0,
        nodes: 1,
        ms: performance.now() - started,
        principalVariation: [block]
      };
    }
  }

  const forceSearch = options.forceSearchInTrouble === true && isPathTrouble(state);
  if (modelConfidence >= confidenceThreshold && !modelAllowsImmediateLoss && !unsafePathRegression && !forceSearch) {
    const score = evaluateState(modelNext, state.turn) + modelConfidence * 80;
    return {
      move: modelMove,
      actionIndex: moveToActionIndex(modelMove),
      score,
      depth: 0,
      nodes: 1,
      ms: performance.now() - started,
      principalVariation: [modelMove]
    };
  }
  const remaining = deadline - performance.now();
  if (remaining < 4) {
    const score = evaluateState(applyMove(state, modelMove), state.turn) + (prediction.policy[modelAction] ?? 0) * 80;
    return {
      move: modelMove,
      actionIndex: moveToActionIndex(modelMove),
      score,
      depth: 0,
      nodes: 1,
      ms: performance.now() - started,
      principalVariation: [modelMove]
    };
  }

  let result = await runSearch(state, {
    timeMs: Math.max(1, remaining),
    maxDepth: options.maxDepth,
    mode: options.searchMode ?? "candidate",
    perspective: state.turn,
    rootPolicyPrior: prediction.policy,
    rootTopK: Math.max(1, Math.min(12, options.topKFromModel))
  });
  const troubleRemaining = deadline - performance.now();
  if (options.troubleSearchMode && shouldRunTrapRecovery(state, result.score) && troubleRemaining > 40) {
    const troubleResult = await runSearch(state, {
      timeMs: Math.max(1, troubleRemaining),
      maxDepth: options.troubleMaxDepth ?? options.maxDepth,
      mode: options.troubleSearchMode,
      perspective: state.turn,
      rootPolicyPrior: prediction.policy,
      rootTopK: Math.max(1, Math.min(12, options.topKFromModel))
    });
    if (troubleResult.score > result.score + 10) result = troubleResult;
  }
  let recovered = options.timeMs >= 100 ? await trapRecoveredResult(state, result) : result;
  const maxTimeMs = Math.max(options.timeMs, options.maxTimeMs ?? options.timeMs);
  if (maxTimeMs > options.timeMs && recovered.depth > 0) {
    const player = state.turn;
    const opponent = otherPlayer(player);
    const myPath = shortestPathLength(state, player);
    const opponentPath = shortestPathLength(state, opponent);
    const modelDisagrees = !sameMove(modelMove, recovered.move);
    const tacticalPosition = recovered.tacticalRepliesUsed === true;
    const pathDeficit = myPath - opponentPath;
    const emergency = tacticalPosition && (
      (pathDeficit >= 8 && myPath >= 18 && recovered.score <= -120 && recovered.depth <= 2)
      || (opponentPath <= 3 && state.players[player].wallsLeft > 0 && myPath >= opponentPath)
    );
    const critical = tacticalPosition
      && recovered.depth <= 3
      && (modelDisagrees || Math.abs(recovered.score) <= 60);
    const targetTimeMs = emergency
      ? maxTimeMs
      : critical
        ? Math.min(maxTimeMs, Math.max(options.timeMs, options.criticalTimeMs ?? 3200))
        : 0;
    const extensionBudget = targetTimeMs - (performance.now() - started);
    if (extensionBudget > 40) {
      const extended = await runSearch(state, {
        timeMs: extensionBudget,
        maxDepth: options.criticalMaxDepth ?? Math.max(options.maxDepth, 10),
        mode: options.searchMode ?? "candidate",
        perspective: state.turn,
        rootPolicyPrior: prediction.policy,
        rootTopK: Math.max(1, Math.min(12, options.topKFromModel)),
        tacticalReplies: "always"
      });
      if (extended.depth >= recovered.depth) {
        const sameDecision = sameMove(extended.move, recovered.move);
        const baseSafety = sameDecision ? 0 : worstImmediateReplyScore(state, recovered.move);
        const extendedSafety = sameDecision ? 0 : worstImmediateReplyScore(state, extended.move);
        const clearlySafer = extendedSafety >= baseSafety + 8;
        const clearlyBetterScore = extended.score >= recovered.score + 30 && extendedSafety >= baseSafety - 2;
        const verifiedEmergency = emergency
          && extended.depth >= recovered.depth + 2
          && extendedSafety >= baseSafety;
        if (sameDecision || clearlySafer || clearlyBetterScore || verifiedEmergency) recovered = extended;
      }
    }
  }
  return {
    ...recovered,
    ms: performance.now() - started
  };
}
