import { searchBestMove } from "../ai/alphabeta";
import { moveOrderingScore } from "../ai/evaluate";
import { selectBestMoveAgainstGreedy } from "../ai/greedyBestResponse";
import { GreedyShortestPathAgent } from "../ai/greedyAgent";
import { ACTION_COUNT, QuoridorState, createInitialState, getWinner } from "../core/state";
import { applyKnownLegalMove, applyMove, getLegalMoves } from "../core/moves";
import { moveToModelActionIndex, getModelLegalActionMask } from "../core/modelPerspective";
import { encodeStateForModel } from "../core/encode";
import { shortestPathLength } from "../core/pathfinding";
import { TrainingSample } from "./replayBuffer";

function createSeededRandom(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

export function createPrefixStressState(game: number, seedBase = 1000): QuoridorState {
  const random = createSeededRandom(seedBase + game);
  let state = createInitialState();
  const plies = 6 + (game % 5);
  for (let ply = 0; ply < plies && getWinner(state) === null; ply++) {
    const moves = getLegalMoves(state);
    if (moves.length === 0) break;
    const ranked = moves
      .slice()
      .sort((a, b) => moveOrderingScore(state, b, state.turn) - moveOrderingScore(state, a, state.turn));
    const pool = ranked.slice(0, Math.min(ranked.length, 8));
    state = applyMove(state, pool[Math.floor(random() * pool.length)]);
  }
  return state;
}

export async function generatePrefixStressTeacherSamples(config: {
  games: number;
  rolloutPlies: number;
  teacherTimeMs: number;
  teacherMaxDepth: number;
  seedBase?: number;
  mirrored?: boolean;
  gamesList?: number[];
  teacherMode?: "search" | "best-response";
  bestResponseMix?: number;
}): Promise<TrainingSample[]> {
  const samples: TrainingSample[] = [];
  const targetPlayers: Array<0 | 1> = config.mirrored === false ? [0] : [0, 1];
  const games = config.gamesList ?? Array.from({ length: config.games }, (_, game) => game);
  for (const game of games) {
    for (const targetPlayer of targetPlayers) {
      let state = createPrefixStressState(game, config.seedBase);
      for (let ply = 0; ply < config.rolloutPlies && getWinner(state) === null; ply++) {
        if (state.turn !== targetPlayer) {
          const opponentMove = await GreedyShortestPathAgent.selectMove(state);
          state = applyKnownLegalMove(state, opponentMove);
          continue;
        }

        const actingPlayer = state.turn;
        const legalMask = getModelLegalActionMask(state);
        const ownBefore = shortestPathLength(state, actingPlayer);
        const oppBefore = shortestPathLength(state, actingPlayer === 0 ? 1 : 0);
        const searchMove = searchBestMove(state, {
            timeMs: config.teacherTimeMs,
            maxDepth: config.teacherMaxDepth,
            mode: "candidate"
          }).move;
        const bestResponseMix = config.teacherMode === "best-response"
          ? Math.max(0, Math.min(1, config.bestResponseMix ?? 1))
          : 0;
        const teacherMove = bestResponseMix > 0
          ? await selectBestMoveAgainstGreedy(state)
          : searchMove;
        const searchActionIndex = moveToModelActionIndex(state, searchMove);
        const bestResponseActionIndex = moveToModelActionIndex(state, teacherMove);
        const policyTarget = new Float32Array(ACTION_COUNT);
        if (bestResponseMix > 0) {
          policyTarget[searchActionIndex] += 1 - bestResponseMix;
          policyTarget[bestResponseActionIndex] += bestResponseMix;
        } else {
          policyTarget[searchActionIndex] = 1;
        }
        const actionIndex = policyTarget[bestResponseActionIndex] > policyTarget[searchActionIndex]
          ? bestResponseActionIndex
          : searchActionIndex;
        const selectedMove = actionIndex === bestResponseActionIndex ? teacherMove : searchMove;
        const next = applyKnownLegalMove(state, selectedMove);
        const ownAfter = shortestPathLength(next, actingPlayer);
        const oppAfter = shortestPathLength(next, actingPlayer === 0 ? 1 : 0);
        const pathProgress = Number.isFinite(ownBefore) && Number.isFinite(ownAfter) ? ownBefore - ownAfter : 0;
        const pathAdvantage = Number.isFinite(ownAfter) && Number.isFinite(oppAfter) ? oppAfter - ownAfter : 0;
        samples.push({
          encodedState: encodeStateForModel(state),
          legalMask,
          actionIndex,
          policyTarget,
          reward: Math.max(-1, Math.min(1, pathAdvantage / 10)),
          priority: 4 + Math.max(0, pathProgress) * 0.25,
          source: "search",
          terminal: getWinner(next) !== null,
          wallMove: selectedMove.type !== "pawn",
          pathAdvantage,
          pathProgress
        });

        state = next;
        if (getWinner(state) !== null) break;
        if (ply % 4 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }
  }
  return samples;
}
