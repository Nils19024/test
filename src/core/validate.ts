import { getLegalMoves } from "./moves";
import { QuoridorState } from "./state";

export function assertValidState(state: QuoridorState): void {
  if (state.players[0].row === state.players[1].row && state.players[0].col === state.players[1].col) {
    throw new Error("Players occupy the same square");
  }
  getLegalMoves(state);
}
