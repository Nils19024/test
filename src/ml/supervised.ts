import { encodeStateForModel } from "../core/encode";
import { applyMove, getLegalMoves } from "../core/moves";
import { getModelLegalActionMask, moveToModelActionIndex } from "../core/modelPerspective";
import { QuoridorState, createInitialState, getWinner } from "../core/state";
import { searchBestMove } from "../ai/alphabeta";
import { TrainingSample } from "./replayBuffer";

const DEFAULT_VALUE_SCALE = 80;

export async function generateTeacherSamples(config: {
  games: number;
  maxMovesPerGame: number;
  teacherTimeMs: number;
  teacherMaxDepth: number;
  randomPrefixMoves?: number;
  valueScale?: number;
  onProgress?: (message: string) => void;
}): Promise<TrainingSample[]> {
  const samples: TrainingSample[] = [];
  const valueScale = config.valueScale ?? DEFAULT_VALUE_SCALE;
  for (let game = 0; game < config.games; game++) {
    let state: QuoridorState = createInitialState();
    state = applyRandomLegalPrefix(state, Math.floor(Math.random() * ((config.randomPrefixMoves ?? 0) + 1)));
    for (let ply = 0; ply < config.maxMovesPerGame && getWinner(state) === null; ply++) {
      const legalMask = getModelLegalActionMask(state);
      const result = searchBestMove(state, {
        timeMs: config.teacherTimeMs,
        maxDepth: config.teacherMaxDepth,
        mode: "candidate"
      });
      const valueTarget = scoreToValueTarget(result.score, valueScale);
      samples.push({
        encodedState: encodeStateForModel(state),
        legalMask,
        actionIndex: moveToModelActionIndex(state, result.move),
        reward: valueTarget,
        priority: 1.4 + Math.abs(valueTarget) * 0.8,
        source: "teacher",
        wallMove: result.move.type !== "pawn"
      });
      state = applyMove(state, result.move);
      if (ply % 25 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
    }
    config.onProgress?.(`Teacher game ${game + 1}/${config.games}, samples=${samples.length}`);
  }
  return samples;
}

export function teacherMoveSample(state: QuoridorState, teacherTimeMs: number, teacherMaxDepth: number, valueScale = DEFAULT_VALUE_SCALE): TrainingSample {
  const result = searchBestMove(state, { timeMs: teacherTimeMs, maxDepth: teacherMaxDepth, mode: "candidate" });
  const valueTarget = scoreToValueTarget(result.score, valueScale);
  return {
    encodedState: encodeStateForModel(state),
    legalMask: getModelLegalActionMask(state),
    actionIndex: moveToModelActionIndex(state, result.move),
    reward: valueTarget,
    priority: 1.4 + Math.abs(valueTarget) * 0.8,
    source: "teacher",
    wallMove: result.move.type !== "pawn"
  };
}

function scoreToValueTarget(score: number, valueScale: number): number {
  if (!Number.isFinite(score)) return 0;
  return Math.max(-1, Math.min(1, Math.tanh(score / Math.max(1, valueScale))));
}

function applyRandomLegalPrefix(initialState: QuoridorState, moves: number): QuoridorState {
  let state = initialState;
  for (let i = 0; i < moves && getWinner(state) === null; i++) {
    const legalMoves = getLegalMoves(state);
    if (legalMoves.length === 0) break;
    state = applyMove(state, legalMoves[Math.floor(Math.random() * legalMoves.length)]);
  }
  return state;
}
