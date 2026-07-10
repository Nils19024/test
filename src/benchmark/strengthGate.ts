import * as fs from "node:fs/promises";
import path from "node:path";
import * as tf from "@tensorflow/tfjs";
import { alphaBetaAgent } from "../ai/alphabeta";
import { GreedyShortestPathAgent } from "../ai/greedyAgent";
import { searchBestMoveHybrid } from "../ai/hybrid";
import { runMatchup } from "./benchmark";
import { TfjsQuoridorPolicyModel } from "../ml/model";
import { isMainModule } from "../util/isMain";

async function loadBrowserModelFromDir(dir: string): Promise<tf.LayersModel> {
  const modelJson = JSON.parse(await fs.readFile(path.join(dir, "model.json"), "utf8"));
  const weightData = await fs.readFile(path.join(dir, "weights.bin"));
  return tf.loadLayersModel(
    tf.io.fromMemory({
      modelTopology: modelJson.modelTopology,
      weightSpecs: modelJson.weightsManifest[0].weights,
      weightData: weightData.buffer.slice(weightData.byteOffset, weightData.byteOffset + weightData.byteLength)
    })
  );
}

export async function runStrengthGate(modelDir = "public/models/latest", only?: "greedy" | "alpha120" | "alpha500"): Promise<{
  passed: boolean;
  results: unknown[];
}> {
  const model = new TfjsQuoridorPolicyModel(await loadBrowserModelFromDir(modelDir));
  const proHybrid = {
    name: "ProHybrid 650ms",
    async selectMove(state: Parameters<typeof searchBestMoveHybrid>[0]) {
      return (await searchBestMoveHybrid(state, model, {
        timeMs: 650,
        maxDepth: 7,
        topKFromModel: 12
      })).move;
    }
  };
  const results = [];
  if (!only || only === "greedy") results.push(await runMatchup({ agentA: proHybrid, agentB: GreedyShortestPathAgent, games: 6, maxMovesPerGame: 200 }));
  if (!only || only === "alpha120") results.push(await runMatchup({ agentA: proHybrid, agentB: alphaBetaAgent(120, 6), games: 6, maxMovesPerGame: 200 }));
  if (!only || only === "alpha500") results.push(await runMatchup({ agentA: proHybrid, agentB: alphaBetaAgent(500, 7), games: 4, maxMovesPerGame: 200 }));
  const passed = results.every((result) => {
    const scoreA = (result.winsA + result.draws * 0.5) / Math.max(1, result.games);
    return result.illegalMoves === 0
      && result.winsB === 0
      && scoreA >= 0.75
      && result.avgMoveMsA < 10000;
  });
  return { passed, results };
}

if (isMainModule(import.meta.url)) {
  const only = process.argv.find((value) => value.startsWith("--only="))?.slice("--only=".length) as "greedy" | "alpha120" | "alpha500" | undefined;
  const gate = await runStrengthGate(process.argv[2] ?? "public/models/latest", only);
  console.log(JSON.stringify(gate.results, null, 2));
  if (!gate.passed) {
    console.error("Strength gate failed");
    process.exitCode = 1;
  }
}
