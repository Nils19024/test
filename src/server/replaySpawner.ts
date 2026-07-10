import path from "node:path";
import { spawn } from "node:child_process";

export function spawnReplayGeneration(config: {
  root: string;
  modelPath: string;
  checkpointGame: number;
  onDone?: (message: string) => void;
  onError?: (message: string) => void;
}): void {
  const tsx = path.join(config.root, "node_modules", ".bin", process.platform === "win32" ? "tsx.cmd" : "tsx");
  const scriptArgs = [
    "src/benchmark/replay.ts",
    `--modelPath=${config.modelPath}`,
    "--outDir=public/replays",
    `--checkpointGame=${config.checkpointGame}`,
    "--maxMoves=90"
  ];
  let child;
  try {
    child = process.platform === "win32"
      ? spawn("cmd.exe", ["/c", tsx, ...scriptArgs], {
          cwd: config.root,
          windowsHide: true,
          stdio: ["ignore", "pipe", "pipe"]
        })
      : spawn(tsx, scriptArgs, {
          cwd: config.root,
          windowsHide: true,
          stdio: ["ignore", "pipe", "pipe"]
        });
  } catch (error) {
    config.onError?.(`Replay process failed: ${error instanceof Error ? error.message : String(error)}`);
    return;
  }
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => {
    stdout += String(chunk);
  });
  child.stderr.on("data", (chunk) => {
    stderr += String(chunk);
  });
  child.on("error", (error) => {
    config.onError?.(`Replay process failed: ${error.message}`);
  });
  child.on("close", (code) => {
    if (code === 0) config.onDone?.(`Replay generated for game ${config.checkpointGame}`);
    else config.onError?.(`Replay process exited with ${code}: ${stderr || stdout}`);
  });
}
