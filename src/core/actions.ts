import { ACTION_COUNT, Move } from "./state";

export function pawnActionIndex(row: number, col: number): number {
  return row * 9 + col;
}

export function hWallActionIndex(row: number, col: number): number {
  return 81 + row * 8 + col;
}

export function vWallActionIndex(row: number, col: number): number {
  return 145 + row * 8 + col;
}

export function actionIndexToMove(index: number): Move {
  if (index < 0 || index >= ACTION_COUNT) throw new Error(`Action index out of range: ${index}`);
  if (index < 81) return { type: "pawn", row: Math.floor(index / 9), col: index % 9 };
  if (index < 145) {
    const offset = index - 81;
    return { type: "hwall", row: Math.floor(offset / 8), col: offset % 8 };
  }
  const offset = index - 145;
  return { type: "vwall", row: Math.floor(offset / 8), col: offset % 8 };
}

export function moveToActionIndex(move: Move): number {
  if (move.type === "pawn") return pawnActionIndex(move.row, move.col);
  if (move.type === "hwall") return hWallActionIndex(move.row, move.col);
  return vWallActionIndex(move.row, move.col);
}
