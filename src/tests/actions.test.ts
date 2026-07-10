import { describe, expect, it } from "vitest";
import { actionIndexToMove, hWallActionIndex, moveToActionIndex, pawnActionIndex, vWallActionIndex } from "../core/actions";
import { createInitialState } from "../core/state";
import { getLegalActionMask, getLegalMoves } from "../core/moves";

describe("action space", () => {
  it("maps pawn and wall actions", () => {
    expect(pawnActionIndex(8, 4)).toBe(76);
    expect(hWallActionIndex(0, 0)).toBe(81);
    expect(hWallActionIndex(7, 7)).toBe(144);
    expect(vWallActionIndex(0, 0)).toBe(145);
    expect(vWallActionIndex(7, 7)).toBe(208);
  });

  it("is reversible for all 209 actions", () => {
    for (let index = 0; index < 209; index++) {
      expect(moveToActionIndex(actionIndexToMove(index))).toBe(index);
    }
  });

  it("creates a legal mask of length 209", () => {
    const state = createInitialState();
    const mask = getLegalActionMask(state);
    expect(mask).toHaveLength(209);
    const legalIndexes = new Set(getLegalMoves(state).map(moveToActionIndex));
    mask.forEach((value, index) => {
      expect(value).toBe(legalIndexes.has(index) ? 1 : 0);
    });
  });
});
