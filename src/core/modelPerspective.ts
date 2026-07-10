import { actionIndexToMove, moveToActionIndex } from "./actions";
import { getLegalMoves } from "./moves";
import { ACTION_COUNT, Move, QuoridorState } from "./state";

function rotateBoardIndex(value: number): number {
  return 8 - value;
}

function rotateWallIndex(value: number): number {
  return 7 - value;
}

export function normalizeMoveForModel(state: QuoridorState, move: Move): Move {
  if (state.turn === 0) return move;
  if (move.type === "pawn") return { type: "pawn", row: rotateBoardIndex(move.row), col: rotateBoardIndex(move.col) };
  if (move.type === "hwall") return { type: "hwall", row: rotateWallIndex(move.row), col: rotateWallIndex(move.col) };
  return { type: "vwall", row: rotateWallIndex(move.row), col: rotateWallIndex(move.col) };
}

export function denormalizeMoveFromModel(state: QuoridorState, move: Move): Move {
  return normalizeMoveForModel(state, move);
}

export function moveToModelActionIndex(state: QuoridorState, move: Move): number {
  return moveToActionIndex(normalizeMoveForModel(state, move));
}

export function modelActionIndexToMove(state: QuoridorState, actionIndex: number): Move {
  return denormalizeMoveFromModel(state, actionIndexToMove(actionIndex));
}

export function getModelLegalActionMask(state: QuoridorState): Uint8Array {
  const mask = new Uint8Array(ACTION_COUNT);
  for (const move of getLegalMoves(state)) mask[moveToModelActionIndex(state, move)] = 1;
  return mask;
}
