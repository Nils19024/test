import * as fs from "node:fs/promises";
import path from "node:path";
import * as tf from "@tensorflow/tfjs";
import { alphaBetaAgent } from "../ai/alphabeta";
import { GreedyShortestPathAgent } from "../ai/greedyAgent";
import { searchBestMoveHybrid } from "../ai/hybrid";
import { QuoridorState } from "../core/state";
import { TfjsQuoridorPolicyModel } from "../ml/model";
import { isMainModule } from "../util/isMain";
import { runMatchup, runMirroredMatchup } from "./benchmark";
import { createDeterministicPrefixState } from "./prefixStress";

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

export async function runEliteGate(modelDir = "public/models/latest") {
  const model = new TfjsQuoridorPolicyModel(await loadBrowserModelFromDir(modelDir));
  const eliteHybrid = {
    name: "EliteHybrid adaptive 650-9500ms",
    async selectMove(state: QuoridorState) {
      return (await searchBestMoveHybrid(state, model, {
        timeMs: 650,
        maxDepth: 7,
        topKFromModel: 12,
        searchMode: "candidate",
        maxTimeMs: 9500,
        criticalTimeMs: 1200,
        criticalMaxDepth: 10
      })).move;
    }
  };
  const initialState = (game: number) => createDeterministicPrefixState(game);
  const standard = [
    await runMatchup({ agentA: eliteHybrid, agentB: GreedyShortestPathAgent, games: 4, maxMovesPerGame: 200 }),
    await runMatchup({ agentA: eliteHybrid, agentB: alphaBetaAgent(120, 6), games: 4, maxMovesPerGame: 200 }),
    await runMatchup({ agentA: eliteHybrid, agentB: alphaBetaAgent(500, 7), games: 4, maxMovesPerGame: 200 })
  ];
  const prefix = [
    await runMirroredMatchup({ agentA: eliteHybrid, agentB: alphaBetaAgent(120, 6), games: 4, maxMovesPerGame: 180, initialState })
  ];
  const standardPassed = standard.every((result) => {
    const scoreA = (result.winsA + result.draws * 0.5) / Math.max(1, result.games);
    return result.illegalMoves === 0
      && result.winsB === 0
      && scoreA >= 0.75
      && result.avgMoveMsA < 10000;
  });
  const prefixPassed = prefix.every((result) => {
    const scoreA = (result.winsA + result.draws * 0.5) / Math.max(1, result.games);
    return result.illegalMoves === 0
      && scoreA >= 0.75
      && result.avgMoveMsA < 10000;
  });
  return { passed: standardPassed && prefixPassed, results: [...standard, ...prefix] };
}

if (isMainModule(import.meta.url)) {
  const gate = await runEliteGate(process.argv[2] ?? "public/models/latest");
  console.log(JSON.stringify(gate.results, null, 2));
  if (!gate.passed) {
    console.error("Elite gate failed");
    process.exitCode = 1;
  }
}
