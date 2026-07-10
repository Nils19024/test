# Quoridor AI

Lokales Quoridor-KI-Projekt mit TypeScript-Core, Alpha-Beta-Lehrer, tfjs Policy-Value-Modell, Training per Browser-Button, Benchmarks und spielbarer `/play`-Seite.

## Installation

```bash
npm install
npm run dev
```

Danach im Browser öffnen:

- `http://localhost:3000/training`
- `http://localhost:3000/play`

## Training Per Klick

1. `npm run dev` starten.
2. `/training` öffnen.
3. Modus wählen: `mixed`, `supervised` oder `selfplay`.
4. Backend wählen: `auto`, `gpu`, `native` oder `cpu`.
4. `Training starten` klicken.
5. Logs, Loss, Samples und Evaluation werden live über Server-Sent Events aktualisiert.
6. `Training stoppen` fordert einen sauberen Stop an.

Das Modell wird nach `models/checkpoints/latest/` gespeichert. `Modell exportieren` kopiert es nach `public/models/latest/`, wo der Browser es mit TensorFlow.js laden kann.

## CLI Training

```bash
npm run train -- -- --mode=mixed --games=1000
```

Mit explizitem Backend:

```bash
npm run train -- -- --mode=mixed --games=1000 --backend=auto
npm run train -- -- --mode=mixed --games=1000 --backend=gpu
```

Nützliche Parameter:

```bash
npm run train -- -- --mode=supervised --games=100 --teacherTimeMs=35 --teacherMaxDepth=4
npm run train -- -- --mode=selfplay --games=500 --maxMovesPerGame=300
```

Der zusätzliche `--` nach `npm run train --` sorgt bei aktuellen npm-Versionen dafür, dass Flags ohne npm-Warnungen direkt an das Training-Script gehen.

## Benchmark

```bash
npm run benchmark
```

Der Benchmark testet feste Gegner:

- RandomAgent
- GreedyShortestPathAgent
- AlphaBeta 20ms
- AlphaBeta 60ms

Über `/training` kann der Button `Benchmark starten` denselben Benchmark lokal ausführen.

## Modell Exportieren

```bash
npm run export-model
```

## Schnelleres Training Auf GPU Oder Native TensorFlow

Standardmäßig läuft das Training portabel auf dem tfjs-CPU-Backend. Für schnelleres Training kann der Trainingsprozess optional native TensorFlow.js-Pakete laden.

Backend-Auswahl:

```text
auto    versucht zuerst GPU, dann native CPU, sonst tfjs CPU
gpu     verlangt @tensorflow/tfjs-node-gpu, fällt bei fehlendem Paket sauber zurück
native  nutzt @tensorflow/tfjs-node, fällt bei fehlendem Paket sauber zurück
cpu     erzwingt das portable tfjs CPU-Backend
```

Installation für native CPU:

```bash
npm run install:native
```

Installation für GPU:

```bash
npm run install:gpu
npm run install:cuda11
```

GPU-Beschleunigung setzt eine passende NVIDIA/CUDA/cuDNN/Node-Kombination voraus. Für dieses Projekt wird CUDA 11.2 + cuDNN 8 lokal unter `.conda/tfjs-cuda/` installiert und vom Trainings-Backend automatisch in den Prozess-PATH aufgenommen. Wenn das native Paket nicht geladen werden kann, bleibt das Projekt lauffähig und zeigt im Training-Log den Fallback an.

GPU-Test:

```bash
npm run train -- -- --mode=selfplay --games=1 --maxMovesPerGame=1 --batchSize=2 --epochs=1 --backend=gpu
```

Im Log sollte stehen:

```text
TensorFlow backend loaded from @tensorflow/tfjs-node-gpu
Created device ... NVIDIA GeForce RTX ...
```

Falls noch kein Checkpoint existiert, wird ein initialisiertes Modell erzeugt und anschließend nach `public/models/latest/` exportiert.

## Gegen Die KI Spielen

1. `npm run dev`
2. `/play` öffnen.
3. Zugmodus wählen: Figur, horizontale Wand oder vertikale Wand.
4. KI-Modus wählen: Random, Greedy, AlphaBeta, Model oder Hybrid.
5. Board-Felder anklicken. Bei Wandmodus wird das angeklickte Feld als linke/obere Wall-Position auf dem 8x8-Wandraster interpretiert.

AlphaBeta läuft im Web Worker, damit die UI nicht blockiert. Model und Hybrid laden `public/models/latest/model.json`; vorher also ein Modell exportieren.

## Projektstruktur

```text
src/core       Quoridor-Regeln, BFS, Action-Space, Encoding
src/ai         Random, Greedy, AlphaBeta, ModelAgent, Hybrid
src/ml         tfjs-Modell, Training, Self-Play, Supervised, Export
src/server     Express API, TrainingManager, SSE
src/benchmark  Matchups und Benchmark-Runner
src/worker     Browser-Worker für KI-Suche
src/web        Training-UI, Play-UI, Board-Renderer
src/tests      Vitest-Tests
```

## Qualität Und Einschränkungen

- Korrektheit hat Vorrang vor Spielstärke.
- Alpha-Beta nutzt iterative Deepening, Alpha-Beta-Pruning, Kandidatenfilter, Move Ordering und eine einfache Transposition Table.
- Wandregeln prüfen Pfaderhalt per BFS für beide Spieler.
- Same-orientation-overlap und Kreuzungen am gleichen Wandrasterpunkt werden als illegal behandelt.
- Das tfjs-Modell ist bewusst klein: Dense 256, Dense 256, Policy 209, Value 1.
- Training lädt optional `@tensorflow/tfjs-node-gpu` oder `@tensorflow/tfjs-node`. Ohne diese Pakete wird automatisch das portable tfjs-CPU-Backend genutzt.
- Self-Play nutzt ein AlphaZero-Light-Prinzip: Das Modell schlägt Kandidaten vor, Alpha-Beta bewertet diese Kandidaten im Training nach, und das Modell lernt eine weiche Policy-Verteilung plus Value. Das Browser-Modell bleibt trotzdem klein.
- Trainingssamples werden optional persistent unter `data/replay/samples.jsonl` gespeichert und beim nächsten Training wieder geladen.

## Besseres Reinforcement Learning

Das Modell bleibt für Browser-Inferenz unverändert klein:

```text
Input 810
Dense 256
Dense 256
Policy 209
Value 1
```

Teurer wird nur das Training. In Self-Play/mixed passiert jetzt:

```text
1. Modell sagt Top-K-Kandidaten voraus
2. Alpha-Beta prüft diese Kandidaten mit kleinem Suchbudget
3. Aus den Bewertungen entsteht ein Policy Target
4. Modell trainiert auf Policy Target + Spielergebnis
5. Samples werden in data/replay persistiert
```

CLI-Parameter:

```bash
npm run train -- -- --mode=mixed --games=1000 --backend=gpu --rlSearchTopK=6 --rlSearchTimeMs=8 --rlSearchMaxDepth=2
```

Mehr Intelligenz, langsameres Training:

```bash
npm run train -- -- --mode=mixed --games=1000 --backend=gpu --teacherTimeMs=100 --rlSearchTopK=10 --rlSearchTimeMs=20 --rlSearchMaxDepth=3
```

Persistenten Replay-Buffer deaktivieren:

```bash
npm run train -- -- --mode=mixed --persistentReplay=false
```

## Empfohlene Startwerte

Schneller Smoke-Test:

```bash
npm run train -- -- --mode=mixed --games=20 --teacherTimeMs=10 --teacherMaxDepth=2
```

Besserer lokaler Start:

```bash
npm run train -- -- --mode=supervised --games=200 --teacherTimeMs=35 --teacherMaxDepth=4
npm run train -- -- --mode=mixed --games=1000 --teacherTimeMs=35 --teacherMaxDepth=4
npm run export-model
```

## Tests

```bash
npm run test
npm run build
```
