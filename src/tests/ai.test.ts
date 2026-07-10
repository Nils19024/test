import { describe, expect, it } from "vitest";
import { alphaBetaAgent, searchBestMove } from "../ai/alphabeta";
import { GreedyShortestPathAgent } from "../ai/greedyAgent";
import { searchBestMoveHybrid } from "../ai/hybrid";
import { ModelAgent } from "../ai/modelAgent";
import { RandomAgent } from "../ai/randomAgent";
import { evaluateState } from "../ai/evaluate";
import { moveToActionIndex } from "../core/actions";
import { applyMove, getLegalActionMask, getLegalMoves } from "../core/moves";
import { createInitialState, getWinner } from "../core/state";
import { QuoridorPolicyModel } from "../ml/model";

function expectLegal(state = createInitialState(), actionIndex: number) {
  const legal = new Set(getLegalMoves(state).map(moveToActionIndex));
  expect(legal.has(actionIndex)).toBe(true);
}

describe("agents", () => {
  it("random, greedy and alphabeta return legal moves", async () => {
    const state = createInitialState();
    expectLegal(state, moveToActionIndex(await RandomAgent.selectMove(state)));
    expectLegal(state, moveToActionIndex(await GreedyShortestPathAgent.selectMove(state)));
    expectLegal(state, moveToActionIndex(await alphaBetaAgent(10, 2).selectMove(state)));
  });

  it("search returns metadata and remains near the time budget", () => {
    const result = searchBestMove(createInitialState(), { timeMs: 20, maxDepth: 3 });
    expectLegal(createInitialState(), result.actionIndex);
    expect(result.nodes).toBeGreaterThan(0);
    expect(result.ms).toBeLessThan(90);
  });

  it("search prefers the direct opening progress from the initial state", () => {
    const result = searchBestMove(createInitialState(), { timeMs: 50, maxDepth: 3 });
    expect(result.move).toEqual({ type: "pawn", row: 7, col: 4 });
  });

  it("search matches exact minimax when transpositions and cutoffs occur", () => {
    const state = createInitialState();
    state.players[0] = { row: 5, col: 4, wallsLeft: 0 };
    state.players[1] = { row: 3, col: 4, wallsLeft: 0 };
    state.turn = 0;

    function exactMinimax(current: typeof state, depth: number): number {
      if (depth === 0 || getWinner(current) !== null) return evaluateState(current, state.turn);
      const scores = getLegalMoves(current).map((move) => exactMinimax(applyMove(current, move), depth - 1));
      return current.turn === state.turn ? Math.max(...scores) : Math.min(...scores);
    }

    const result = searchBestMove(state, { timeMs: 1000, maxDepth: 4, mode: "full", perspective: state.turn });
    const exactScore = Math.max(...getLegalMoves(state).map((move) => exactMinimax(applyMove(state, move), 3)));

    expect(result.depth).toBe(4);
    expect(result.score).toBe(exactScore);
  });

  it("model agent masks illegal actions", async () => {
    const fakeModel: QuoridorPolicyModel = {
      async predict(_encoded, legalMask) {
        const policy = new Float32Array(209);
        policy[0] = 1000;
        const legalIndex = legalMask.findIndex((value) => value === 1);
        policy[legalIndex] = 1;
        return { policy, value: 0 };
      }
    };
    const state = createInitialState();
    const move = await new ModelAgent(fakeModel).selectMove(state);
    expect(() => applyMove(state, move)).not.toThrow();
    expect(getLegalActionMask(state)[moveToActionIndex(move)]).toBe(1);
  });

  it("hybrid takes an immediate win even when the model is confident elsewhere", async () => {
    const state = createInitialState();
    state.players[0] = { row: 1, col: 4, wallsLeft: 10 };
    state.players[1] = { row: 7, col: 8, wallsLeft: 10 };
    state.turn = 0;
    const badMove = { type: "pawn" as const, row: 2, col: 4 };
    const fakeModel: QuoridorPolicyModel = {
      async predict(_encoded, legalMask) {
        const policy = new Float32Array(209);
        policy[moveToActionIndex(badMove)] = legalMask[moveToActionIndex(badMove)] ? 100 : 0;
        return { policy, value: 0 };
      }
    };

    const result = await searchBestMoveHybrid(state, fakeModel, { timeMs: 50, maxDepth: 2, topKFromModel: 4 });

    expect(result.move).toEqual({ type: "pawn", row: 0, col: 4 });
    expect(getWinner(applyMove(state, result.move))).toBe(0);
  });

  it("hybrid blocks an immediate opponent win instead of trusting a confident model move", async () => {
    const state = createInitialState();
    state.players[0] = { row: 8, col: 0, wallsLeft: 10 };
    state.players[1] = { row: 7, col: 4, wallsLeft: 10 };
    state.turn = 0;
    const losingMove = { type: "pawn" as const, row: 7, col: 0 };
    const fakeModel: QuoridorPolicyModel = {
      async predict(_encoded, legalMask) {
        const policy = new Float32Array(209);
        policy[moveToActionIndex(losingMove)] = legalMask[moveToActionIndex(losingMove)] ? 100 : 0;
        return { policy, value: 0 };
      }
    };

    const result = await searchBestMoveHybrid(state, fakeModel, { timeMs: 50, maxDepth: 2, topKFromModel: 4 });
    const next = applyMove(state, result.move);
    const opponentCanWin = getLegalMoves(next).some((reply) => getWinner(applyMove(next, reply)) === 1);

    expect(result.move.type).toBe("hwall");
    expect(opponentCanWin).toBe(false);
  });

  it("hybrid extends a shallow tactical search when adaptive time is available", async () => {
    const state = createInitialState();
    const legalMove = getLegalMoves(state)[0];
    let calls = 0;
    const uncertainModel: QuoridorPolicyModel = {
      async predict(_encoded, legalMask) {
        const policy = new Float32Array(209);
        const legalCount = legalMask.reduce((sum, value) => sum + value, 0);
        for (let index = 0; index < legalMask.length; index++) {
          if (legalMask[index]) policy[index] = 1 / legalCount;
        }
        return { policy, value: 0 };
      }
    };

    const result = await searchBestMoveHybrid(state, uncertainModel, {
      timeMs: 100,
      maxDepth: 4,
      topKFromModel: 4,
      maxTimeMs: 500,
      criticalTimeMs: 300,
      criticalMaxDepth: 6,
      async runSearch(_searchState, options) {
        calls++;
        return {
          move: legalMove,
          actionIndex: moveToActionIndex(legalMove),
          score: 0,
          depth: calls === 1 ? 2 : 5,
          nodes: calls * 10,
          ms: options.timeMs,
          principalVariation: [legalMove],
          tacticalRepliesUsed: true
        };
      }
    });

    expect(calls).toBe(2);
    expect(result.depth).toBe(5);
  });
});
