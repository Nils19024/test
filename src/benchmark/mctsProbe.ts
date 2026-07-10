import * as fs from "node:fs/promises";
import path from "node:path";
import * as tf from "@tensorflow/tfjs";
import { alphaBetaAgent } from "../ai/alphabeta";
import { GreedyShortestPathAgent } from "../ai/greedyAgent";
import { QuoridorState } from "../core/state";
import { searchBestMoveMcts } from "../ml/alphaZeroSelfPlay";
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

function numberArg(name: string, fallback: number): number {
  const prefix = `--${name}=`;
  const raw = process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length);
  const value = raw ? Number(raw) : fallback;
  return Number.isFinite(value) ? value : fallback;
}

export async function runMctsProbe(modelDir = "public/models/latest") {
  const model = new TfjsQuoridorPolicyModel(await loadBrowserModelFromDir(modelDir));
  const simulations = numberArg("simulations", 96);
  const timeMs = numberArg("timeMs", 900);
  const cpuct = numberArg("cpuct", 1.35);
  const heuristicPriorMix = numberArg("heuristicPriorMix", 0.35);
  const valueHeuristicMix = numberArg("valueHeuristicMix", 0.55);
  const mctsAgent = {
    name: `MCTS ${simulations}/${timeMs}ms`,
    async selectMove(state: QuoridorState) {
      return (await searchBestMoveMcts({ state, model, simulations, timeMs, cpuct, heuristicPriorMix, valueHeuristicMix })).move;
    }
  };
  const initialState = (game: number) => createDeterministicPrefixState(game);
  return {
    config: { simulations, timeMs, cpuct, heuristicPriorMix, valueHeuristicMix },
    standard: [
      await runMatchup({ agentA: mctsAgent, agentB: GreedyShortestPathAgent, games: 4, maxMovesPerGame: 200 }),
      await runMatchup({ agentA: mctsAgent, agentB: alphaBetaAgent(120, 6), games: 4, maxMovesPerGame: 200 })
    ],
    prefix: [
      await runMirroredMatchup({ agentA: mctsAgent, agentB: GreedyShortestPathAgent, games: 4, maxMovesPerGame: 180, initialState }),
      await runMirroredMatchup({ agentA: mctsAgent, agentB: alphaBetaAgent(120, 6), games: 3, maxMovesPerGame: 180, initialState })
    ]
  };
}

async function main(): Promise<void> {
  console.log(JSON.stringify(await runMctsProbe(process.argv[2] ?? "public/models/latest"), null, 2));
}

if (isMainModule(import.meta.url)) {
  void main();
}
