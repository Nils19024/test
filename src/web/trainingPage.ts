import { api } from "./apiClient";
import type { TrainingStatus } from "../server/trainingManager";
import { renderBoard } from "./boardRenderer";
import type { QuoridorState, Move } from "../core/state";

function metric(label: string, value: string): string {
  return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
}

function renderStatus(status: TrainingStatus): string {
  const evalAny = status.latestEvaluation as any;
  const backendLabel = status.backend
    ? `${status.backend.activeBackend}${status.backend.loadedPackage ? ` (${status.backend.loadedPackage})` : ""}`
    : "-";
  return [
    metric("Modus", status.mode),
    metric("Status", status.running ? "läuft" : "gestoppt"),
    metric("Backend", backendLabel),
    metric("Spiele", String(status.gamesPlayed)),
    metric("Samples", String(status.samplesCollected)),
    metric("Loss", status.currentLoss?.toFixed(4) ?? "-"),
    metric("Policy Loss", status.policyLoss?.toFixed(4) ?? "-"),
    metric("Value Loss", status.valueLoss?.toFixed(4) ?? "-"),
    metric("Reward", status.rewardMean?.toFixed(3) ?? "-"),
    metric("Reward Min/Max", status.rewardMin != null && status.rewardMax != null ? `${status.rewardMin.toFixed(2)} / ${status.rewardMax.toFixed(2)}` : "-"),
    metric("Policy Entropy", status.policyEntropyMean?.toFixed(3) ?? "-"),
    metric("Top Action", status.policyTopProbMean?.toFixed(3) ?? "-"),
    metric("Action Diversity", status.actionDiversity?.toFixed(3) ?? "-"),
    metric("Samples/Game", status.avgSamplesPerGame?.toFixed(1) ?? "-"),
    metric("Terminal Rate", status.terminalRate?.toFixed(2) ?? "-"),
    metric("Loop Rate", status.loopRate?.toFixed(2) ?? "-"),
    metric("Wall Moves", status.wallMoveRate?.toFixed(2) ?? "-"),
    metric("Forward", status.forwardProgressRate?.toFixed(2) ?? "-"),
    metric("Path Adv", status.pathAdvantageMean?.toFixed(2) ?? "-"),
    metric("Avg Length", status.avgGameLength?.toFixed(1) ?? "-"),
    metric("Source", status.generationSource ?? "-"),
    metric("Random", evalAny?.random?.winrateA?.toFixed(2) ?? "-"),
    metric("Greedy", evalAny?.greedy?.winrateA?.toFixed(2) ?? "-"),
    metric("BestResp", evalAny?.greedyBestResponse?.agreement?.toFixed(2) ?? "-"),
    metric("AlphaBeta 20ms", evalAny?.alpha20?.winrateA?.toFixed(2) ?? "-"),
    metric("Modellpfad", status.modelPath ?? "-")
  ].join("");
}

type ChartPoint = {
  game: number;
  loss: number | null;
  policyLoss: number | null;
  valueLoss: number | null;
  rewardMean: number | null;
  rewardMin: number | null;
  rewardMax: number | null;
  policyEntropyMean: number | null;
  policyTopProbMean: number | null;
  actionDiversity: number | null;
  avgSamplesPerGame: number | null;
  terminalRate: number | null;
  loopRate: number | null;
  wallMoveRate: number | null;
  forwardProgressRate: number | null;
  pathAdvantageMean: number | null;
  avgGameLength: number | null;
  random: number | null;
  greedy: number | null;
  greedyBestResponse: number | null;
  alpha20: number | null;
};

type ReplayRecord = {
  id: string;
  createdAt: string;
  checkpointGame: number;
  matchup: string;
  initialState: QuoridorState;
  finalState: QuoridorState;
  winner: 0 | 1 | null;
  steps: Array<{
    ply: number;
    player: 0 | 1;
    agent: string;
    state: QuoridorState;
    move: Move;
    actionIndex: number;
    ms: number;
  }>;
};

function valueOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function pointFromStatus(status: TrainingStatus): ChartPoint {
  const evalAny = status.latestEvaluation as any;
  return {
    game: status.gamesPlayed,
    loss: status.currentLoss,
    policyLoss: status.policyLoss,
    valueLoss: status.valueLoss,
    rewardMean: valueOrNull(status.rewardMean),
    rewardMin: valueOrNull(status.rewardMin),
    rewardMax: valueOrNull(status.rewardMax),
    policyEntropyMean: valueOrNull(status.policyEntropyMean),
    policyTopProbMean: valueOrNull(status.policyTopProbMean),
    actionDiversity: valueOrNull(status.actionDiversity),
    avgSamplesPerGame: valueOrNull(status.avgSamplesPerGame),
    terminalRate: valueOrNull(status.terminalRate),
    loopRate: valueOrNull(status.loopRate),
    wallMoveRate: valueOrNull(status.wallMoveRate),
    forwardProgressRate: valueOrNull(status.forwardProgressRate),
    pathAdvantageMean: valueOrNull(status.pathAdvantageMean),
    avgGameLength: valueOrNull(status.avgGameLength),
    random: valueOrNull(evalAny?.random?.winrateA),
    greedy: valueOrNull(evalAny?.greedy?.winrateA),
    greedyBestResponse: valueOrNull(evalAny?.greedyBestResponse?.agreement),
    alpha20: valueOrNull(evalAny?.alpha20?.winrateA)
  };
}

function drawLineChart(canvas: HTMLCanvasElement, title: string, series: Array<{ label: string; color: string; values: Array<number | null> }>, games: number[]): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const pad = { left: 42, right: 14, top: 28, bottom: 30 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const allValues = series.flatMap((line) => line.values.filter((value): value is number => value !== null));
  const minGame = Math.min(...games, 0);
  const maxGame = Math.max(...games, 1);
  const minValue = title === "Winrate" ? 0 : Math.min(...allValues, 0);
  const maxValue = title === "Winrate" ? 1 : Math.max(...allValues, 1);
  const yRange = Math.max(0.001, maxValue - minValue);

  ctx.fillStyle = "#fffaf0";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#ccc3ae";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad.left, pad.top, plotW, plotH);
  ctx.fillStyle = "#17201b";
  ctx.font = "700 13px system-ui, sans-serif";
  ctx.fillText(title, pad.left, 18);

  ctx.font = "11px system-ui, sans-serif";
  ctx.fillStyle = "#5f625a";
  ctx.fillText(String(minValue.toFixed(title === "Winrate" ? 1 : 2)), 4, pad.top + plotH);
  ctx.fillText(String(maxValue.toFixed(title === "Winrate" ? 1 : 2)), 4, pad.top + 4);
  ctx.fillText(`Game ${maxGame}`, width - 72, height - 8);

  const x = (game: number) => pad.left + ((game - minGame) / Math.max(1, maxGame - minGame)) * plotW;
  const y = (value: number) => pad.top + plotH - ((value - minValue) / yRange) * plotH;

  series.forEach((line, index) => {
    ctx.strokeStyle = line.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    line.values.forEach((value, i) => {
      if (value === null) return;
      const px = x(games[i]);
      const py = y(value);
      if (!started) {
        ctx.moveTo(px, py);
        started = true;
      } else {
        ctx.lineTo(px, py);
      }
    });
    ctx.stroke();
    ctx.fillStyle = line.color;
    ctx.fillText(line.label, pad.left + index * 108, height - 8);
  });
}

export function mountTrainingPage(app: HTMLElement): void {
  app.innerHTML = `
    <main class="shell">
      <nav class="topnav"><a href="/training">Training</a><a href="/play">Gegen KI spielen</a></nav>
      <section class="toolbar">
        <button id="start">Training starten</button>
        <button id="stop">Training stoppen</button>
        <button id="benchmark">Benchmark starten</button>
        <button id="export">Modell exportieren</button>
        <button id="diagnostics">Diagnose-Log erstellen</button>
      </section>
      <section class="config">
        <label>Modus <select id="mode"><option value="alphazero">alphazero</option><option value="mixed">mixed</option><option value="supervised">supervised</option><option value="selfplay">selfplay</option></select></label>
        <label>Backend <select id="backend"><option value="auto">auto</option><option value="gpu">gpu</option><option value="native">native</option><option value="cpu">cpu</option></select></label>
        <label>Spiele <input id="games" type="number" value="50" min="1" max="100000" /></label>
        <label>Teacher ms <input id="teacherTimeMs" type="number" value="20" min="1" max="500" /></label>
      </section>
      <div class="progress"><span id="bar"></span></div>
      <section id="metrics" class="metrics"></section>
      <section class="charts">
        <canvas id="lossChart" aria-label="Loss Verlauf"></canvas>
        <canvas id="rewardChart" aria-label="Reward Verlauf"></canvas>
        <canvas id="learningChart" aria-label="Lernsignal Verlauf"></canvas>
        <canvas id="qualityChart" aria-label="Spielqualitaet Verlauf"></canvas>
        <canvas id="winrateChart" aria-label="Winrate Verlauf"></canvas>
      </section>
      <section class="replay-panel">
        <div class="replay-head">
          <strong>Replay</strong>
          <span id="replayInfo">Noch kein Replay</span>
          <button id="replayToggle" type="button">Pause</button>
        </div>
        <div class="replay-grid">
          <div id="replayBoard"></div>
          <div class="replay-meta">
            <div id="replayMove">-</div>
            <input id="replaySlider" type="range" min="0" max="0" value="0" />
          </div>
        </div>
      </section>
      <pre id="logs" class="logs"></pre>
      <pre id="benchmarkOut" class="logs compact"></pre>
    </main>
  `;
  const metrics = app.querySelector("#metrics") as HTMLElement;
  const logs = app.querySelector("#logs") as HTMLElement;
  const bar = app.querySelector("#bar") as HTMLElement;
  const benchmarkOut = app.querySelector("#benchmarkOut") as HTMLElement;
  const lossChart = app.querySelector("#lossChart") as HTMLCanvasElement;
  const rewardChart = app.querySelector("#rewardChart") as HTMLCanvasElement;
  const learningChart = app.querySelector("#learningChart") as HTMLCanvasElement;
  const qualityChart = app.querySelector("#qualityChart") as HTMLCanvasElement;
  const winrateChart = app.querySelector("#winrateChart") as HTMLCanvasElement;
  const replayBoard = app.querySelector("#replayBoard") as HTMLElement;
  const replayInfo = app.querySelector("#replayInfo") as HTMLElement;
  const replayMove = app.querySelector("#replayMove") as HTMLElement;
  const replaySlider = app.querySelector("#replaySlider") as HTMLInputElement;
  const replayToggle = app.querySelector("#replayToggle") as HTMLButtonElement;
  let history: ChartPoint[] = [];
  let replay: ReplayRecord | null = null;
  let replayIndex = 0;
  let replayPlaying = true;
  let latestReplayPath: string | null = null;

  function replayStateAt(index: number): QuoridorState | null {
    if (!replay) return null;
    if (index <= 0) return replay.initialState;
    if (index < replay.steps.length) return replay.steps[index].state;
    return replay.finalState;
  }

  function drawReplay(): void {
    if (!replay) {
      replayBoard.innerHTML = "";
      replayMove.textContent = "-";
      return;
    }
    const state = replayStateAt(replayIndex);
    if (state) {
      renderBoard(replayBoard, state, () => undefined, { orientation: "blue-bottom" });
      renderReplayWallOwners();
    }
    const step = replay.steps[Math.max(0, replayIndex - 1)];
    replayInfo.textContent = `Game ${replay.checkpointGame}: ${replay.matchup}, Winner ${replay.winner ?? "draw"}`;
    replayMove.textContent = step
      ? `Ply ${step.ply + 1}/${replay.steps.length}: Spieler ${step.player} (${step.agent}) spielt ${JSON.stringify(step.move)} (${step.ms.toFixed(1)}ms)`
      : `Startposition, ${replay.steps.length} Züge`;
    replaySlider.max = String(replay.steps.length);
    replaySlider.value = String(replayIndex);
  }

  function renderReplayWallOwners(): void {
    if (!replay) return;
    const wrap = replayBoard.querySelector(".board-wrap");
    if (!wrap) return;
    wrap.querySelector(".owner-wall-layer")?.remove();
    const layer = document.createElement("div");
    layer.className = "owner-wall-layer";
    replay.steps.slice(0, replayIndex).forEach((step) => {
      if (step.move.type === "pawn") return;
      const wall = document.createElement("div");
      wall.className = `owner-wall ${step.move.type === "hwall" ? "h" : "v"} p${step.player}`;
      wall.title = `Spieler ${step.player}: ${JSON.stringify(step.move)}`;
      if (step.move.type === "hwall") {
        wall.style.top = `calc(${step.move.row + 1} * var(--cell) - 6px)`;
        wall.style.left = `calc(${step.move.col} * var(--cell) + 8px)`;
      } else {
        wall.style.top = `calc(${step.move.row} * var(--cell) + 8px)`;
        wall.style.left = `calc(${step.move.col + 1} * var(--cell) - 6px)`;
      }
      layer.append(wall);
    });
    wrap.append(layer);
  }

  async function loadReplay(path: string): Promise<void> {
    const separator = path.includes("?") ? "&" : "?";
    const response = await fetch(`${path}${separator}t=${Date.now()}`);
    if (!response.ok) return;
    replay = await response.json();
    replayIndex = 0;
    replayPlaying = true;
    replayToggle.textContent = "Pause";
    drawReplay();
  }

  function updateCharts(): void {
    const games = history.map((point) => point.game);
    drawLineChart(lossChart, "Loss", [
      { label: "Loss", color: "#214f9b", values: history.map((point) => point.loss) },
      { label: "Policy", color: "#2f6f4e", values: history.map((point) => point.policyLoss) },
      { label: "Value", color: "#a32935", values: history.map((point) => point.valueLoss) }
    ], games);
    drawLineChart(winrateChart, "Winrate", [
      { label: "Random", color: "#214f9b", values: history.map((point) => point.random) },
      { label: "Greedy", color: "#2f6f4e", values: history.map((point) => point.greedy) },
      { label: "Alpha20", color: "#a32935", values: history.map((point) => point.alpha20) }
    ], games);
    drawLineChart(rewardChart, "Reward", [
      { label: "Mean", color: "#7c3f00", values: history.map((point) => point.rewardMean) },
      { label: "Min", color: "#a32935", values: history.map((point) => point.rewardMin) },
      { label: "Max", color: "#2f6f4e", values: history.map((point) => point.rewardMax) }
    ], games);
    drawLineChart(learningChart, "Learning Signal", [
      { label: "Entropy", color: "#214f9b", values: history.map((point) => point.policyEntropyMean) },
      { label: "TopProb", color: "#a32935", values: history.map((point) => point.policyTopProbMean) },
      { label: "Diversity", color: "#2f6f4e", values: history.map((point) => point.actionDiversity) },
      { label: "BestResp", color: "#7c3f00", values: history.map((point) => point.greedyBestResponse) }
    ], games);
    drawLineChart(qualityChart, "Game Quality", [
      { label: "Terminal", color: "#2f6f4e", values: history.map((point) => point.terminalRate) },
      { label: "Loop", color: "#a32935", values: history.map((point) => point.loopRate) },
      { label: "Walls", color: "#7c3f00", values: history.map((point) => point.wallMoveRate) },
      { label: "Forward", color: "#214f9b", values: history.map((point) => point.forwardProgressRate) }
    ], games);
  }

  function update(status: TrainingStatus): void {
    metrics.innerHTML = renderStatus(status);
    logs.textContent = status.logs.join("\n");
    const target = Number((app.querySelector("#games") as HTMLInputElement).value || 1);
    bar.style.width = `${Math.min(100, (status.gamesPlayed / Math.max(1, target)) * 100)}%`;
    if (status.gamesPlayed > 0) {
      const point = pointFromStatus(status);
      const last = history.at(-1);
      if (!last || last.game !== point.game) history.push(point);
      else history[history.length - 1] = point;
      history = history.slice(-250);
    }
    if (status.latestReplayPath && status.latestReplayPath !== latestReplayPath) {
      latestReplayPath = status.latestReplayPath;
      void loadReplay(status.latestReplayPath);
    }
    updateCharts();
  }

  app.querySelector("#start")?.addEventListener("click", async () => {
    history = [];
    updateCharts();
    await api.startTraining({
      mode: (app.querySelector("#mode") as HTMLSelectElement).value as any,
      backend: (app.querySelector("#backend") as HTMLSelectElement).value as any,
      games: Number((app.querySelector("#games") as HTMLInputElement).value),
      teacherTimeMs: Number((app.querySelector("#teacherTimeMs") as HTMLInputElement).value),
      teacherMaxDepth: 3,
      saveEveryGames: 100,
      replayEveryGames: 10,
      evaluateEveryGames: 100,
      maxMovesPerGame: 80,
      batchSize: 128,
      alphaZeroSimulations: 16,
      alphaZeroFastSimulations: 4,
      alphaZeroWarmupGames: 0,
      alphaZeroFullSimulationStart: 5000,
      alphaZeroCpuct: 1.5,
      alphaZeroDirichletAlpha: 0.3,
      alphaZeroDirichletFraction: 0.35,
      alphaZeroTemperatureMoves: 80,
      alphaZeroMinTemperature: 0.5,
      alphaZeroPolicySmoothing: 0.06,
      alphaZeroHeuristicPriorMix: 0.55,
      alphaZeroDrawValueWeight: 0.7,
      alphaZeroParallelGames: 2,
      alphaZeroUpdatesPerGame: 1,
      gpuUtilizationTarget: 0.45,
      gpuCooldownMs: 900,
      bootstrapMaxUpdates: 32,
      bootstrapGames: 2,
      bootstrapBestResponseGames: 5,
      greedyBestResponseGames: 2,
      resumeFromCheckpoint: true
    });
  });
  app.querySelector("#stop")?.addEventListener("click", () => api.stopTraining());
  app.querySelector("#benchmark")?.addEventListener("click", async () => {
    benchmarkOut.textContent = "Benchmark läuft...";
    benchmarkOut.textContent = JSON.stringify(await api.benchmark(), null, 2);
  });
  app.querySelector("#export")?.addEventListener("click", async () => {
    benchmarkOut.textContent = JSON.stringify(await api.exportModel(), null, 2);
  });
  app.querySelector("#diagnostics")?.addEventListener("click", async () => {
    benchmarkOut.textContent = "Diagnose wird erstellt...";
    const result = await api.diagnostics();
    benchmarkOut.textContent = `Diagnose gespeichert:\n${result.path}\n\nKompakte Vorschau:\n${JSON.stringify({
      signals: result.report.signals,
      learningTrend: result.report.learningTrend
    }, null, 2)}`;
  });
  replayToggle.addEventListener("click", () => {
    replayPlaying = !replayPlaying;
    replayToggle.textContent = replayPlaying ? "Pause" : "Play";
  });
  replaySlider.addEventListener("input", () => {
    replayIndex = Number(replaySlider.value);
    replayPlaying = false;
    replayToggle.textContent = "Play";
    drawReplay();
  });
  window.setInterval(() => {
    if (!replay || !replayPlaying) return;
    replayIndex = replayIndex >= replay.steps.length ? 0 : replayIndex + 1;
    drawReplay();
  }, 700);

  const events = new EventSource("/api/training/events");
  events.onmessage = (event) => update(JSON.parse(event.data));
  api.trainingStatus().then(update);
  fetch("/api/replays")
    .then((response) => response.json())
    .then((data) => {
      if (data.latest) {
        latestReplayPath = data.latest;
        void loadReplay(data.latest);
      }
    })
    .catch(() => undefined);
  window.addEventListener("resize", updateCharts);
}
