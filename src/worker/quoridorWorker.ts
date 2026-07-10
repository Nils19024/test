import { SearchOptions, searchBestMove } from "../ai/alphabeta";
import { GreedyShortestPathAgent } from "../ai/greedyAgent";
import { RandomAgent } from "../ai/randomAgent";
import { moveToActionIndex } from "../core/actions";
import { QuoridorState } from "../core/state";

self.onmessage = async (event: MessageEvent) => {
  const { id, engine, state, options } = event.data as {
    id: string;
    engine: "model" | "alphabeta" | "hybrid" | "random" | "greedy";
    state: QuoridorState;
    options: SearchOptions;
  };
  const started = performance.now();
  try {
    if (engine === "random") {
      const move = await RandomAgent.selectMove(state);
      self.postMessage({ id, type: "result", result: { move, actionIndex: moveToActionIndex(move), score: 0, value: 0, ms: performance.now() - started } });
      return;
    }
    if (engine === "greedy") {
      const move = await GreedyShortestPathAgent.selectMove(state);
      self.postMessage({ id, type: "result", result: { move, actionIndex: moveToActionIndex(move), score: 0, value: 0, ms: performance.now() - started } });
      return;
    }
    const result = searchBestMove(state, options);
    self.postMessage({ id, type: "result", result: { ...result, value: result.score, ms: performance.now() - started } });
  } catch (error) {
    self.postMessage({ id, type: "error", error: error instanceof Error ? error.message : String(error) });
  }
};
