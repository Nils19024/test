import express from "express";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import { createServer as createViteServer } from "vite";
import { TrainingManager } from "./trainingManager";
import { runServerBenchmark } from "./benchmarkManager";
import { createTrainingDiagnostics } from "./trainingDiagnostics";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "../..");
const app = express();
const training = new TrainingManager();
const port = Number(process.env.PORT ?? 3000);

app.use(express.json({ limit: "1mb" }));
app.use("/replays", express.static(path.join(root, "public/replays")));

app.post("/api/training/start", async (req, res) => {
  try {
    await training.start(req.body ?? {});
    res.json(training.getStatus());
  } catch (error) {
    res.status(400).json({ error: error instanceof Error ? error.message : String(error) });
  }
});

app.post("/api/training/stop", (_req, res) => {
  training.stop();
  res.json(training.getStatus());
});

app.get("/api/training/status", (_req, res) => {
  res.json(training.getStatus());
});

app.get("/api/training/diagnostics", async (_req, res) => {
  try {
    res.json(await createTrainingDiagnostics(root, training.getStatus()));
  } catch (error) {
    res.status(500).json({ error: error instanceof Error ? error.message : String(error) });
  }
});

app.get("/api/training/events", (req, res) => {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive"
  });
  const send = () => res.write(`data: ${JSON.stringify(training.getStatus())}\n\n`);
  const listener = () => send();
  training.on("status", listener);
  send();
  req.on("close", () => training.off("status", listener));
});

app.post("/api/benchmark/start", async (_req, res) => {
  res.json(await runServerBenchmark());
});

app.get("/api/models", (_req, res) => {
  const checkpointDir = path.join(root, "models/checkpoints");
  const publicDir = path.join(root, "public/models");
  res.json({
    checkpoints: fs.existsSync(checkpointDir) ? fs.readdirSync(checkpointDir) : [],
    browserModels: fs.existsSync(publicDir) ? fs.readdirSync(publicDir) : []
  });
});

app.get("/api/replays", (_req, res) => {
  const replayDir = path.join(root, "public/replays");
  res.json({
    latest: fs.existsSync(path.join(replayDir, "latest.json")) ? "/replays/latest.json" : null,
    files: fs.existsSync(replayDir) ? fs.readdirSync(replayDir).filter((file) => file.endsWith(".json")) : []
  });
});

app.post("/api/models/export-browser", async (_req, res) => {
  try {
    res.json({ path: await training.exportBrowser() });
  } catch (error) {
    res.status(400).json({ error: error instanceof Error ? error.message : String(error) });
  }
});

if (process.env.NODE_ENV === "production") {
  app.use(express.static(path.join(root, "dist")));
  app.get(["/training", "/play", "/"], (_req, res) => res.sendFile(path.join(root, "dist/index.html")));
} else {
  const vite = await createViteServer({ root, server: { middlewareMode: true, hmr: { port: port + 1 } }, appType: "spa" });
  app.use(vite.middlewares);
}

app.listen(port, () => {
  console.log(`Quoridor AI running at http://localhost:${port}/training`);
});
