import { Move, QuoridorState } from "../core/state";

export type BoardOrientation = "blue-bottom" | "red-bottom";

export type WallPreview = {
  move: Extract<Move, { type: "hwall" | "vwall" }>;
  legal: boolean;
};

export type BoardRenderOptions = {
  orientation: BoardOrientation;
  preview?: WallPreview | null;
};

function visualCellToGame(row: number, col: number, orientation: BoardOrientation): { row: number; col: number } {
  if (orientation === "red-bottom") return { row: 8 - row, col: 8 - col };
  return { row, col };
}

function gameCellToVisual(row: number, col: number, orientation: BoardOrientation): { row: number; col: number } {
  return visualCellToGame(row, col, orientation);
}

function gameWallToVisual(row: number, col: number, orientation: BoardOrientation): { row: number; col: number } {
  if (orientation === "red-bottom") return { row: 7 - row, col: 7 - col };
  return { row, col };
}

export function visualWallToGame(
  row: number,
  col: number,
  orientation: BoardOrientation
): { row: number; col: number } {
  return gameWallToVisual(row, col, orientation);
}

function appendWall(
  layer: HTMLElement,
  type: "hwall" | "vwall",
  row: number,
  col: number,
  orientation: BoardOrientation,
  className = "wall"
): void {
  const visual = gameWallToVisual(row, col, orientation);
  const wall = document.createElement("div");
  wall.className = `${className} ${type === "hwall" ? "h" : "v"}`;
  if (type === "hwall") {
    wall.style.top = `calc(${visual.row + 1} * var(--cell) - 5px)`;
    wall.style.left = `calc(${visual.col} * var(--cell) + 8px)`;
  } else {
    wall.style.top = `calc(${visual.row} * var(--cell) + 8px)`;
    wall.style.left = `calc(${visual.col + 1} * var(--cell) - 5px)`;
  }
  layer.append(wall);
}

export function renderBoard(
  container: HTMLElement,
  state: QuoridorState,
  onPawnMove: (move: Extract<Move, { type: "pawn" }>) => void,
  options: BoardRenderOptions
): void {
  container.innerHTML = "";
  const board = document.createElement("div");
  board.className = "board";
  for (let visualRow = 0; visualRow < 9; visualRow++) {
    for (let visualCol = 0; visualCol < 9; visualCol++) {
      const { row, col } = visualCellToGame(visualRow, visualCol, options.orientation);
      const cell = document.createElement("button");
      cell.className = "cell";
      cell.type = "button";
      cell.dataset.row = String(row);
      cell.dataset.col = String(col);
      cell.setAttribute("aria-label", `Feld ${row + 1},${col + 1}`);
      const player = state.players.findIndex((p) => p.row === row && p.col === col);
      if (player >= 0) {
        const piece = document.createElement("span");
        piece.className = `piece p${player}`;
        piece.textContent = player === 0 ? "B" : "R";
        cell.append(piece);
      }
      cell.addEventListener("click", () => onPawnMove({ type: "pawn", row, col }));
      board.append(cell);
    }
  }

  const wallLayer = document.createElement("div");
  wallLayer.className = "wall-layer";
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      if (state.hWalls[row][col]) appendWall(wallLayer, "hwall", row, col, options.orientation);
      if (state.vWalls[row][col]) appendWall(wallLayer, "vwall", row, col, options.orientation);
    }
  }
  if (options.preview) {
    appendWall(
      wallLayer,
      options.preview.move.type,
      options.preview.move.row,
      options.preview.move.col,
      options.orientation,
      `wall-preview ${options.preview.legal ? "legal" : "illegal"}`
    );
  }

  const wrap = document.createElement("div");
  wrap.className = "board-wrap";
  wrap.append(board, wallLayer);
  container.append(wrap);
}

export function boardPointToWallMove(
  boardWrap: HTMLElement,
  clientX: number,
  clientY: number,
  type: "hwall" | "vwall",
  orientation: BoardOrientation
): Extract<Move, { type: "hwall" | "vwall" }> | null {
  const rect = boardWrap.getBoundingClientRect();
  const cell = rect.width / 9;
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  if (x < -cell * 0.25 || x > rect.width + cell * 0.25 || y < -cell * 0.25 || y > rect.height + cell * 0.25) {
    return null;
  }

  let visualRow: number;
  let visualCol: number;
  if (type === "hwall") {
    visualRow = Math.round(y / cell) - 1;
    visualCol = Math.floor(x / cell);
  } else {
    visualRow = Math.floor(y / cell);
    visualCol = Math.round(x / cell) - 1;
  }

  visualRow = Math.max(0, Math.min(7, visualRow));
  visualCol = Math.max(0, Math.min(7, visualCol));
  const game = visualWallToGame(visualRow, visualCol, orientation);
  return { type, row: game.row, col: game.col };
}
