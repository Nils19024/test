import { describe, expect, it } from "vitest";
import { actionIndexToMove, moveToActionIndex } from "../core/actions";
import { encodeStateForModel } from "../core/encode";
import { applyMove, getLegalMoves, getLegalPawnMoves, isWallLegal } from "../core/moves";
import { getModelLegalActionMask, modelActionIndexToMove, moveToModelActionIndex } from "../core/modelPerspective";
import { hasPathToGoal, shortestPathLength } from "../core/pathfinding";
import { Move, QuoridorState, createInitialState, getWinner } from "../core/state";
import { canMoveBetween } from "../core/walls";
import { MODEL_INPUT_SIZE } from "../ml/model";

function hasMove(moves: Move[], move: Move): boolean {
  const index = moveToActionIndex(move);
  return moves.some((candidate) => moveToActionIndex(candidate) === index);
}

describe("quoridor core", () => {
  it("creates the official start position", () => {
    const state = createInitialState();
    expect(state.players[0]).toEqual({ row: 8, col: 4, wallsLeft: 10 });
    expect(state.players[1]).toEqual({ row: 0, col: 4, wallsLeft: 10 });
    expect(state.turn).toBe(0);
  });

  it("has legal start moves", () => {
    const moves = getLegalMoves(createInitialState());
    expect(hasMove(moves, { type: "pawn", row: 7, col: 4 })).toBe(true);
    expect(hasMove(moves, { type: "pawn", row: 8, col: 3 })).toBe(true);
    expect(hasMove(moves, { type: "pawn", row: 8, col: 5 })).toBe(true);
    expect(moves.length).toBe(131);
  });

  it("applies movement and detects winners", () => {
    let state = createInitialState();
    state = applyMove(state, { type: "pawn", row: 7, col: 4 });
    expect(state.players[0].row).toBe(7);
    expect(state.turn).toBe(1);
    state.players[0].row = 0;
    expect(getWinner(state)).toBe(0);
  });

  it("blocks movement with horizontal and vertical walls", () => {
    const state = createInitialState();
    state.hWalls[7][4] = true;
    expect(canMoveBetween(state, 8, 4, 7, 4)).toBe(false);
    const state2 = createInitialState();
    state2.vWalls[7][4] = true;
    expect(canMoveBetween(state2, 8, 4, 8, 5)).toBe(false);
  });

  it("supports jumps and diagonal sidesteps", () => {
    const state = createInitialState();
    state.players[0] = { row: 4, col: 4, wallsLeft: 10 };
    state.players[1] = { row: 3, col: 4, wallsLeft: 10 };
    let pawns = getLegalPawnMoves(state);
    expect(hasMove(pawns, { type: "pawn", row: 2, col: 4 })).toBe(true);

    state.hWalls[2][4] = true;
    pawns = getLegalPawnMoves(state);
    expect(hasMove(pawns, { type: "pawn", row: 3, col: 3 })).toBe(true);
    expect(hasMove(pawns, { type: "pawn", row: 3, col: 5 })).toBe(true);
  });

  it("rejects occupied, overlapping and crossing walls", () => {
    const state = createInitialState();
    state.hWalls[2][2] = true;
    expect(isWallLegal(state, { type: "hwall", row: 2, col: 2 })).toBe(false);
    expect(isWallLegal(state, { type: "hwall", row: 2, col: 3 })).toBe(false);
    expect(isWallLegal(state, { type: "vwall", row: 2, col: 2 })).toBe(false);
    expect(isWallLegal(state, { type: "hwall", row: -1, col: 0 })).toBe(false);
  });

  it("computes paths and model encoding", () => {
    const state = createInitialState();
    expect(shortestPathLength(state, 0)).toBe(8);
    expect(shortestPathLength(state, 1)).toBe(8);
    expect(hasPathToGoal(state, 0)).toBe(true);
    expect(encodeStateForModel(state)).toHaveLength(MODEL_INPUT_SIZE);
  });

  it("keeps action moves legal after decoding", () => {
    const state = createInitialState();
    const move = actionIndexToMove(moveToActionIndex(getLegalMoves(state)[0]));
    expect(() => applyMove(state, move)).not.toThrow();
  });

  it("normalizes model actions from the current player's perspective", () => {
    let state = createInitialState();
    state = applyMove(state, { type: "pawn", row: 7, col: 4 });
    const realMove: Move = { type: "pawn", row: 1, col: 4 };
    const modelAction = moveToModelActionIndex(state, realMove);
    expect(actionIndexToMove(modelAction)).toEqual({ type: "pawn", row: 7, col: 4 });
    expect(modelActionIndexToMove(state, modelAction)).toEqual(realMove);
    expect(getModelLegalActionMask(state)[modelAction]).toBe(1);
  });
});
