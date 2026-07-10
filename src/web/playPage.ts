import { actionIndexToMove, moveToActionIndex } from "../core/actions";
import { applyMove, getLegalMoves } from "../core/moves";
import { Move, PlayerIndex, QuoridorState, cloneState, createInitialState, getWinner } from "../core/state";
import { RandomAgent } from "../ai/randomAgent";
import { GreedyShortestPathAgent } from "../ai/greedyAgent";
import { searchBestMoveHybrid } from "../ai/hybrid";
import { SearchOptions, SearchResult } from "../ai/alphabeta";
import { ModelAgent } from "../ai/modelAgent";
import { renderBoard, BoardOrientation, WallPreview, boardPointToWallMove } from "./boardRenderer";
import { loadBrowserModel } from "./browserModel";

type WallMove = Extract<Move, { type: "hwall" | "vwall" }>;

export function mountPlayPage(app: HTMLElement): void {
  let state: QuoridorState = createInitialState();
  let humanPlayer: PlayerIndex = 0;
  let orientation: BoardOrientation = "blue-bottom";
  let preview: WallPreview | null = null;
  let message = "Du spielst Blau.";
  let last = "-";
  let aiThinking = false;
  let aiGeneration = 0;
  let dragType: WallMove["type"] | null = null;
  const history: QuoridorState[] = [];
  const worker = new Worker(new URL("../worker/quoridorWorker.ts", import.meta.url), { type: "module" });

  app.innerHTML = `
    <main class="shell play-shell">
      <nav class="topnav"><a href="/training">Training</a><a href="/play">Gegen KI spielen</a></nav>
      <section class="toolbar play-toolbar">
        <button id="new">Neues Spiel</button>
        <button id="undo">Undo</button>
        <button id="rotate">Spielfeld drehen</button>
        <button id="reloadModel">Modell neu laden</button>
        <label>Deine Farbe
          <select id="humanColor">
            <option value="0" selected>Blau</option>
            <option value="1">Rot</option>
          </select>
        </label>
        <select id="engine"><option value="hybrid" selected>Hybrid Modell+Suche</option><option value="model">Trainiertes Modell</option><option value="alphabeta">AlphaBeta</option><option value="random">Random</option><option value="greedy">Greedy</option></select>
        <select id="difficulty"><option value="easy">Easy</option><option value="medium">Medium</option><option value="hard">Hard</option><option value="pro" selected>Pro</option><option value="elite">Elite</option></select>
      </section>
      <section class="play-grid">
        <div class="board-area">
          <div id="boardHost"></div>
          <div class="wall-tray" aria-label="Wand-Ablage">
            <button class="wall-token h-token" id="dragHWall" type="button" data-wall-type="hwall" aria-label="Horizontale Wand ziehen"></button>
            <button class="wall-token v-token" id="dragVWall" type="button" data-wall-type="vwall" aria-label="Vertikale Wand ziehen"></button>
          </div>
        </div>
        <aside class="side">
          <div id="status"></div>
          <div id="last"></div>
        </aside>
      </section>
    </main>
  `;
  const boardHost = app.querySelector("#boardHost") as HTMLElement;
  const status = app.querySelector("#status") as HTMLElement;
  const lastEl = app.querySelector("#last") as HTMLElement;
  const undoButton = app.querySelector("#undo") as HTMLButtonElement;
  const colorSelect = app.querySelector("#humanColor") as HTMLSelectElement;

  function playerName(player: PlayerIndex): string {
    return player === 0 ? "Blau" : "Rot";
  }

  function draw(): void {
    renderBoard(boardHost, state, onHumanPawnMove, { orientation, preview });
    const winner = getWinner(state);
    const turnText = winner === null ? `Zug: ${playerName(state.turn)}` : `${playerName(winner)} gewinnt.`;
    status.textContent = `${message} ${turnText} Waende: Blau ${state.players[0].wallsLeft} / Rot ${state.players[1].wallsLeft}`;
    lastEl.textContent = `Letzter KI-Zug: ${last}`;
    undoButton.disabled = history.length === 0 || aiThinking;
  }

  function legal(move: Move): boolean {
    const index = moveToActionIndex(move);
    return getLegalMoves(state).some((m) => moveToActionIndex(m) === index);
  }

  function commitMove(move: Move): void {
    history.push(cloneState(state));
    state = applyMove(state, move);
  }

  function resetGame(nextHumanPlayer = humanPlayer): void {
    humanPlayer = nextHumanPlayer;
    state = createInitialState();
    history.length = 0;
    preview = null;
    aiGeneration++;
    aiThinking = false;
    last = "-";
    message = `Du spielst ${playerName(humanPlayer)}.`;
    draw();
    void aiMove();
  }

  async function aiMove(): Promise<void> {
    const winner = getWinner(state);
    if (winner !== null || state.turn === humanPlayer || aiThinking) return;
    aiThinking = true;
    const generation = aiGeneration;
    message = "KI denkt...";
    draw();
    const engine = (app.querySelector("#engine") as HTMLSelectElement).value;
    const difficulty = (app.querySelector("#difficulty") as HTMLSelectElement).value;
    const options = {
      easy: { timeMs: 10, maxDepth: 3 },
      medium: { timeMs: 80, maxDepth: 4 },
      hard: { timeMs: 200, maxDepth: 6 },
      pro: { timeMs: 650, maxDepth: 7 },
      elite: {
        timeMs: 650,
        maxDepth: 7,
        maxTimeMs: 9500,
        criticalTimeMs: 1200,
        criticalMaxDepth: 10,
        searchMode: "candidate" as const
      }
    }[difficulty]!;
    const started = performance.now();
    try {
      let move: Move;
      let details = "";
      if (engine === "random") move = await RandomAgent.selectMove(state);
      else if (engine === "greedy") move = await GreedyShortestPathAgent.selectMove(state);
      else if (engine === "model") move = await new ModelAgent(await loadBrowserModel()).selectMove(state);
      else if (engine === "hybrid") {
        const result = await searchBestMoveHybrid(state, await loadBrowserModel(), {
          ...options,
          topKFromModel: 8,
          runSearch: askWorker
        });
        move = result.move;
        details = ` value=${result.score.toFixed(1)} depth=${result.depth} nodes=${result.nodes}`;
      } else {
        const result = await askWorker(state, options);
        move = actionIndexToMove(result.actionIndex);
        details = ` score=${result.score.toFixed(1)} depth=${result.depth} nodes=${result.nodes}`;
      }
      if (generation !== aiGeneration) return;
      commitMove(move);
      last = `${JSON.stringify(move)} in ${(performance.now() - started).toFixed(1)}ms${details}`;
      const won = getWinner(state);
      message = won === null ? "KI hat gezogen." : `${playerName(won)} gewinnt.`;
    } catch (error) {
      message = `KI-Fehler: ${error instanceof Error ? error.message : String(error)}`;
    } finally {
      if (generation === aiGeneration) aiThinking = false;
      draw();
    }
  }

  function askWorker(current: QuoridorState, options: SearchOptions): Promise<SearchResult> {
    return new Promise((resolve, reject) => {
      const id = crypto.randomUUID();
      const listener = (event: MessageEvent) => {
        if (event.data.id !== id) return;
        worker.removeEventListener("message", listener);
        if (event.data.type === "error") reject(new Error(event.data.error));
        else resolve(event.data.result);
      };
      worker.addEventListener("message", listener);
      worker.postMessage({ id, type: "search", engine: "alphabeta", state: current, options });
    });
  }

  function onHumanPawnMove(move: Extract<Move, { type: "pawn" }>): void {
    if (state.turn !== humanPlayer || aiThinking) return;
    if (!legal(move)) {
      message = "Illegaler Zug.";
      draw();
      return;
    }
    commitMove(move);
    const won = getWinner(state);
    message = won === null ? "Du hast gezogen." : `${playerName(won)} gewinnt.`;
    draw();
    void aiMove();
  }

  function onHumanWallMove(move: WallMove): void {
    if (state.turn !== humanPlayer || aiThinking) return;
    if (!legal(move)) {
      message = "Illegale Wand.";
      draw();
      return;
    }
    commitMove(move);
    const won = getWinner(state);
    message = won === null ? "Du hast eine Wand gesetzt." : `${playerName(won)} gewinnt.`;
    draw();
    void aiMove();
  }

  function updateDragPreview(clientX: number, clientY: number): void {
    if (!dragType) return;
    const wrap = boardHost.querySelector(".board-wrap") as HTMLElement | null;
    if (!wrap) return;
    const move = boardPointToWallMove(wrap, clientX, clientY, dragType, orientation);
    preview = move ? { move, legal: state.turn === humanPlayer && !aiThinking && legal(move) } : null;
    draw();
  }

  function startWallDrag(event: PointerEvent, type: WallMove["type"]): void {
    if (state.turn !== humanPlayer || aiThinking) return;
    event.preventDefault();
    dragType = type;
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
    updateDragPreview(event.clientX, event.clientY);
  }

  function moveWallDrag(event: PointerEvent): void {
    if (!dragType) return;
    updateDragPreview(event.clientX, event.clientY);
  }

  function endWallDrag(event: PointerEvent): void {
    if (!dragType) return;
    const dropped = preview;
    dragType = null;
    preview = null;
    (event.currentTarget as HTMLElement).releasePointerCapture(event.pointerId);
    if (dropped?.legal) onHumanWallMove(dropped.move);
    else draw();
  }

  function startWallMouseDrag(event: MouseEvent, type: WallMove["type"]): void {
    if (state.turn !== humanPlayer || aiThinking || event.button !== 0) return;
    event.preventDefault();
    dragType = type;
    updateDragPreview(event.clientX, event.clientY);
  }

  function moveWallMouseDrag(event: MouseEvent): void {
    if (!dragType) return;
    updateDragPreview(event.clientX, event.clientY);
  }

  function endWallMouseDrag(): void {
    if (!dragType) return;
    const dropped = preview;
    dragType = null;
    preview = null;
    if (dropped?.legal) onHumanWallMove(dropped.move);
    else draw();
  }

  app.querySelector("#new")?.addEventListener("click", () => resetGame());
  undoButton.addEventListener("click", () => {
    const previous = history.pop();
    if (!previous || aiThinking) return;
    state = previous;
    preview = null;
    message = "Letzter Zug wurde zurueckgenommen.";
    draw();
  });
  app.querySelector("#rotate")?.addEventListener("click", () => {
    orientation = orientation === "blue-bottom" ? "red-bottom" : "blue-bottom";
    draw();
  });
  colorSelect.addEventListener("change", () => {
    resetGame(Number(colorSelect.value) as PlayerIndex);
  });
  app.querySelector("#reloadModel")?.addEventListener("click", async () => {
    try {
      await loadBrowserModel(true);
      message = "Trainiertes Modell neu geladen.";
    } catch (error) {
      message = `Modell konnte nicht geladen werden: ${error instanceof Error ? error.message : String(error)}`;
    }
    draw();
  });
  app.querySelectorAll<HTMLElement>("[data-wall-type]").forEach((button) => {
    const type = button.dataset.wallType as WallMove["type"];
    button.addEventListener("pointerdown", (event) => startWallDrag(event, type));
    button.addEventListener("pointermove", moveWallDrag);
    button.addEventListener("pointerup", endWallDrag);
    button.addEventListener("pointercancel", endWallDrag);
    button.addEventListener("mousedown", (event) => startWallMouseDrag(event, type));
  });
  window.addEventListener("mousemove", moveWallMouseDrag);
  window.addEventListener("mouseup", endWallMouseDrag);
  draw();
}
