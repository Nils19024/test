import { Move, QuoridorState } from "../core/state";

export interface Agent {
  name: string;
  selectMove(state: QuoridorState): Promise<Move>;
}
