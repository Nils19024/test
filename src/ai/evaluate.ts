import { Move, QuoridorState, getWinner, otherPlayer } from "../core/state";
import { shortestPathLength } from "../core/pathfinding";

export function evaluateState(state: QuoridorState, perspective = state.turn): number {
  const winner = getWinner(state);
  if (winner === perspective) return 100000;
  if (winner === otherPlayer(perspective)) return -100000;
  const opponent = otherPlayer(perspective);
  const myPath = shortestPathLength(state, perspective);
  const oppPath = shortestPathLength(state, opponent);
  const myWalls = state.players[perspective].wallsLeft;
  const oppWalls = state.players[opponent].wallsLeft;
  // One useful wall commonly changes a shortest path by about one step, so its
  // remaining option value belongs on the same scale as path distance.
  let score = oppPath * 18 - myPath * 22 + myWalls * 18 - oppWalls * 18;
  if (myPath <= 2) score += 40;
  if (oppPath <= 2) score -= 45;
  return score;
}

export function moveOrderingScore(state: QuoridorState, move: Move, perspective = state.turn): number {
  if (move.type === "pawn") {
    const target = perspective === 0 ? 0 : 8;
    return 100 - Math.abs(move.row - target) * 8;
  }
  return 20 - Math.abs(move.row - 3.5) - Math.abs(move.col - 3.5);
}
