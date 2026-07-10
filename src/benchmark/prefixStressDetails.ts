import * as fs from "node:fs/promises";
import path from "node:path";
import * as tf from "@tensorflow/tfjs";
import { alphaBetaAgent } from "../ai/alphabeta";
import { selectBestMoveAgainstGreedy } from "../ai/greedyBestResponse";
import { GreedyShortestPathAgent } from "../ai/greedyAgent";
import { searchBestMoveHybrid } from "../ai/hybrid";
import { QuoridorState } from "../core/state";
import { TfjsQuoridorPolicyModel } from "../ml/model";
import { isMainModule } from "../util/isMain";
import { runMirroredMatchupDetails } from "./benchmark";
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

export async function runPrefixStressDetails(modelDir = "public/models/latest") {
  const model = new TfjsQuoridorPolicyModel(await loadBrowserModelFromDir(modelDir));
  const proHybrid = {
    name: "ProHybrid 650ms",
    async selectMove(state: QuoridorState) {
      return (await searchBestMoveHybrid(state, model, {
        timeMs: 650,
        maxDepth: 7,
        topKFromModel: 12
      })).move;
    }
  };
  const initialState = (game: number) => createDeterministicPrefixState(game);
  const bestResponseVsGreedy = {
    name: "BestResponseVsGreedy",
    selectMove: selectBestMoveAgainstGreedy
  };
  return {
    greedy: await runMirroredMatchupDetails({
      agentA: proHybrid,
      agentB: GreedyShortestPathAgent,
      games: 6,
      maxMovesPerGame: 180,
      initialState
    }),
    bestResponseVsGreedy: await runMirroredMatchupDetails({
      agentA: proHybrid,
      agentB: bestResponseVsGreedy,
      games: 6,
      maxMovesPerGame: 180,
      initialState
    }),
    alphaBeta120: await runMirroredMatchupDetails({
      agentA: proHybrid,
      agentB: alphaBetaAgent(120, 6),
      games: 4,
      maxMovesPerGame: 180,
      initialState
    })
  };
}

async function main(): Promise<void> {
  console.log(JSON.stringify(await runPrefixStressDetails(process.argv[2] ?? "public/models/latest"), null, 2));
}

if (isMainModule(import.meta.url)) {
  void main();
}
