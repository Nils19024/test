import { Agent } from "./agents";
import { getLegalMoves } from "../core/moves";

export const RandomAgent: Agent = {
  name: "Random",
  async selectMove(state) {
    const moves = getLegalMoves(state);
    if (moves.length === 0) throw new Error("No legal moves");
    return moves[Math.floor(Math.random() * moves.length)];
  }
};
