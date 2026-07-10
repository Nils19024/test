import { Move, QuoridorState, inBoard, inWallGrid } from "./state";

export function isBlockedBetween(state: QuoridorState, r1: number, c1: number, r2: number, c2: number): boolean {
  if (!inBoard(r1, c1) || !inBoard(r2, c2)) return true;
  const dr = r2 - r1;
  const dc = c2 - c1;
  if (Math.abs(dr) + Math.abs(dc) !== 1) return true;

  if (dr !== 0) {
    const top = Math.min(r1, r2);
    const col = c1;
    if (col > 0 && state.hWalls[top][col - 1]) return true;
    if (col < 8 && state.hWalls[top][col]) return true;
  } else {
    const left = Math.min(c1, c2);
    const row = r1;
    if (row > 0 && state.vWalls[row - 1][left]) return true;
    if (row < 8 && state.vWalls[row][left]) return true;
  }
  return false;
}

export function canMoveBetween(state: QuoridorState, r1: number, c1: number, r2: number, c2: number): boolean {
  return inBoard(r2, c2) && !isBlockedBetween(state, r1, c1, r2, c2);
}

export function isWallPlacementFree(state: QuoridorState, move: Move): boolean {
  if (move.type === "pawn") return false;
  const { row, col } = move;
  if (!inWallGrid(row, col)) return false;
  if (move.type === "hwall") {
    if (state.hWalls[row][col] || state.vWalls[row][col]) return false;
    if (col > 0 && state.hWalls[row][col - 1]) return false;
    if (col < 7 && state.hWalls[row][col + 1]) return false;
    return true;
  }
  if (state.vWalls[row][col] || state.hWalls[row][col]) return false;
  if (row > 0 && state.vWalls[row - 1][col]) return false;
  if (row < 7 && state.vWalls[row + 1][col]) return false;
  return true;
}
