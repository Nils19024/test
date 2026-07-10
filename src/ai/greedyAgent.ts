import { Agent } from "./agents";
import { applyKnownLegalMove, getLegalMoves } from "../core/moves";
import { shortestPathLength } from "../core/pathfinding";
import { otherPlayer } from "../core/state";

export const GreedyShortestPathAgent: Agent = {
  name: "GreedyShortestPath",
  async selectMove(state) {
    const player = state.turn;
    const opponent = otherPlayer(player);
    const legalMoves = getLegalMoves(state);
    let best = legalMoves[0];
    let bestScore = -Infinity;
    for (const move of legalMoves) {
      const next = applyKnownLegalMove(state, move);
      const score = shortestPathLength(next, opponent) * 1.2 - shortestPathLength(next, player);
      if (score > bestScore) {
        best = move;
        bestScore = score;
      }
    }
    return best;
  }
};
