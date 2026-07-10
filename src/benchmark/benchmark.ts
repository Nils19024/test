import { Agent } from "../ai/agents";
import { applyMove, getLegalMoves } from "../core/moves";
import { Move, QuoridorState, createInitialState, getWinner } from "../core/state";
import { moveToActionIndex } from "../core/actions";

export type BenchmarkResult = {
  matchup: string;
  games: number;
  winsA: number;
  winsB: number;
  draws: number;
  winrateA: number;
  avgMoveMsA: number;
  avgMoveMsB: number;
  avgGameLength: number;
  illegalMoves: number;
};

export type MatchupGameDetail = {
  game: number;
  aIsPlayer: 0 | 1;
  winner: 0 | 1 | null;
  aWon: boolean;
  draw: boolean;
  illegalMove: boolean;
  moveNumber: number;
  avgMoveMsA: number;
  avgMoveMsB: number;
};

function isLegalMove(move: Move, legalMoves: Move[]): boolean {
  const index = moveToActionIndex(move);
  return legalMoves.some((legal) => moveToActionIndex(legal) === index);
}

export async function runMatchup(config: {
  agentA: Agent;
  agentB: Agent;
  games: number;
  maxMovesPerGame: number;
  initialState?: (game: number) => QuoridorState | Promise<QuoridorState>;
}): Promise<BenchmarkResult> {
  let winsA = 0;
  let winsB = 0;
  let draws = 0;
  let illegalMoves = 0;
  let movesA = 0;
  let movesB = 0;
  let msA = 0;
  let msB = 0;
  let totalLength = 0;

  for (let game = 0; game < config.games; game++) {
    let state = config.initialState ? await config.initialState(game) : createInitialState();
    const aIsPlayer = (game % 2) as 0 | 1;
    for (let ply = 0; ply < config.maxMovesPerGame && getWinner(state) === null; ply++) {
      const agent = state.turn === aIsPlayer ? config.agentA : config.agentB;
      const started = performance.now();
      const move = await agent.selectMove(state);
      const elapsed = performance.now() - started;
      const legalMoves = getLegalMoves(state);
      if (!isLegalMove(move, legalMoves)) {
        illegalMoves++;
        break;
      }
      if (state.turn === aIsPlayer) {
        msA += elapsed;
        movesA++;
      } else {
        msB += elapsed;
        movesB++;
      }
      state = applyMove(state, move);
    }
    totalLength += state.moveNumber;
    const winner = getWinner(state);
    if (winner === null) draws++;
    else if (winner === aIsPlayer) winsA++;
    else winsB++;
  }

  return {
    matchup: `${config.agentA.name} vs ${config.agentB.name}`,
    games: config.games,
    winsA,
    winsB,
    draws,
    winrateA: winsA / Math.max(1, config.games),
    avgMoveMsA: msA / Math.max(1, movesA),
    avgMoveMsB: msB / Math.max(1, movesB),
    avgGameLength: totalLength / Math.max(1, config.games),
    illegalMoves
  };
}

export async function runMirroredMatchup(config: {
  agentA: Agent;
  agentB: Agent;
  games: number;
  maxMovesPerGame: number;
  initialState: (game: number) => QuoridorState | Promise<QuoridorState>;
}): Promise<BenchmarkResult> {
  let winsA = 0;
  let winsB = 0;
  let draws = 0;
  let illegalMoves = 0;
  let movesA = 0;
  let movesB = 0;
  let msA = 0;
  let msB = 0;
  let totalLength = 0;

  for (let game = 0; game < config.games; game++) {
    for (const aIsPlayer of [0, 1] as const) {
      let state = await config.initialState(game);
      for (let ply = 0; ply < config.maxMovesPerGame && getWinner(state) === null; ply++) {
        const agent = state.turn === aIsPlayer ? config.agentA : config.agentB;
        const started = performance.now();
        const move = await agent.selectMove(state);
        const elapsed = performance.now() - started;
        const legalMoves = getLegalMoves(state);
        if (!isLegalMove(move, legalMoves)) {
          illegalMoves++;
          break;
        }
        if (state.turn === aIsPlayer) {
          msA += elapsed;
          movesA++;
        } else {
          msB += elapsed;
          movesB++;
        }
        state = applyMove(state, move);
      }
      totalLength += state.moveNumber;
      const winner = getWinner(state);
      if (winner === null) draws++;
      else if (winner === aIsPlayer) winsA++;
      else winsB++;
    }
  }

  const playedGames = config.games * 2;
  return {
    matchup: `${config.agentA.name} vs ${config.agentB.name} mirrored`,
    games: playedGames,
    winsA,
    winsB,
    draws,
    winrateA: winsA / Math.max(1, playedGames),
    avgMoveMsA: msA / Math.max(1, movesA),
    avgMoveMsB: msB / Math.max(1, movesB),
    avgGameLength: totalLength / Math.max(1, playedGames),
    illegalMoves
  };
}

export async function runMirroredMatchupDetails(config: {
  agentA: Agent;
  agentB: Agent;
  games: number;
  maxMovesPerGame: number;
  initialState: (game: number) => QuoridorState | Promise<QuoridorState>;
}): Promise<MatchupGameDetail[]> {
  const details: MatchupGameDetail[] = [];
  for (let game = 0; game < config.games; game++) {
    for (const aIsPlayer of [0, 1] as const) {
      let state = await config.initialState(game);
      let illegalMove = false;
      let movesA = 0;
      let movesB = 0;
      let msA = 0;
      let msB = 0;
      for (let ply = 0; ply < config.maxMovesPerGame && getWinner(state) === null; ply++) {
        const agent = state.turn === aIsPlayer ? config.agentA : config.agentB;
        const started = performance.now();
        const move = await agent.selectMove(state);
        const elapsed = performance.now() - started;
        const legalMoves = getLegalMoves(state);
        if (!isLegalMove(move, legalMoves)) {
          illegalMove = true;
          break;
        }
        if (state.turn === aIsPlayer) {
          msA += elapsed;
          movesA++;
        } else {
          msB += elapsed;
          movesB++;
        }
        state = applyMove(state, move);
      }
      const winner = getWinner(state);
      details.push({
        game,
        aIsPlayer,
        winner,
        aWon: winner === aIsPlayer,
        draw: winner === null,
        illegalMove,
        moveNumber: state.moveNumber,
        avgMoveMsA: msA / Math.max(1, movesA),
        avgMoveMsB: msB / Math.max(1, movesB)
      });
    }
  }
  return details;
}
