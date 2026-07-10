import { alphaBetaAgent } from "../ai/alphabeta";
import { GreedyShortestPathAgent } from "../ai/greedyAgent";
import { RandomAgent } from "../ai/randomAgent";
import { runMatchup } from "./benchmark";
import { isMainModule } from "../util/isMain";

export async function runDefaultBenchmarks() {
  return [
    await runMatchup({ agentA: GreedyShortestPathAgent, agentB: RandomAgent, games: 10, maxMovesPerGame: 200 }),
    await runMatchup({ agentA: alphaBetaAgent(20, 3), agentB: RandomAgent, games: 6, maxMovesPerGame: 200 }),
    await runMatchup({ agentA: alphaBetaAgent(60, 5), agentB: GreedyShortestPathAgent, games: 4, maxMovesPerGame: 200 })
  ];
}

if (isMainModule(import.meta.url)) {
  const results = await runDefaultBenchmarks();
  console.log(JSON.stringify(results, null, 2));
}
