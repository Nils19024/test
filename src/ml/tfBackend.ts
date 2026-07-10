import * as tf from "@tensorflow/tfjs";
import path from "node:path";
import { createRequire } from "node:module";
import { existsSync } from "node:fs";

export type TrainingBackendPreference = "auto" | "cpu" | "native" | "gpu";

export type TrainingBackendInfo = {
  preference: TrainingBackendPreference;
  activeBackend: string;
  loadedPackage: string | null;
  gpuRequested: boolean;
  accelerated: boolean;
};

const dynamicImport = new Function("specifier", "return import(specifier)") as (specifier: string) => Promise<unknown>;
const require = createRequire(import.meta.url);

function addDirsToPath(dirs: string[]): void {
  const currentPath = process.env.PATH ?? "";
  const pathParts = currentPath.toLowerCase().split(";");
  const missing = dirs
    .filter((dir) => existsSync(dir))
    .filter((dir) => !pathParts.includes(dir.toLowerCase()));
  if (missing.length > 0) process.env.PATH = `${missing.join(";")};${currentPath}`;
}

function addProjectCudaDirsToPath(): void {
  const cwdCudaBin = path.resolve(process.cwd(), ".conda", "tfjs-cuda", "Library", "bin");
  const sourceRelativeCudaBin = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..", "..", ".conda", "tfjs-cuda", "Library", "bin");
  addDirsToPath([cwdCudaBin, sourceRelativeCudaBin]);
}

function addPackageDllDirsToPath(packageName: string): void {
  try {
    const packageJson = require.resolve(`${packageName}/package.json`);
    const packageRoot = path.dirname(packageJson);
    const candidates = [
      path.join(packageRoot, "deps", "lib"),
      path.join(packageRoot, "lib", "napi-v8"),
      path.join(packageRoot, "lib", "napi-v9")
    ];
    addDirsToPath(candidates);
  } catch {
    // Package is optional. Import below will report the actionable error.
  }
}

async function tryLoadBackendPackage(packageName: string, onLog?: (message: string) => void): Promise<boolean> {
  try {
    addProjectCudaDirsToPath();
    addPackageDllDirsToPath(packageName);
    await dynamicImport(packageName);
    await tf.setBackend("tensorflow");
    await tf.ready();
    onLog?.(`TensorFlow backend loaded from ${packageName}`);
    return true;
  } catch (error) {
    onLog?.(`TensorFlow backend ${packageName} unavailable: ${error instanceof Error ? error.message : String(error)}`);
    return false;
  }
}

export async function initializeTrainingBackend(
  preference: TrainingBackendPreference = "auto",
  onLog?: (message: string) => void
): Promise<TrainingBackendInfo> {
  const requested = preference ?? "auto";
  const wantsGpu = requested === "gpu";
  const candidates =
    requested === "gpu"
      ? ["@tensorflow/tfjs-node-gpu"]
      : requested === "native"
        ? ["@tensorflow/tfjs-node"]
        : requested === "auto"
          ? ["@tensorflow/tfjs-node-gpu", "@tensorflow/tfjs-node"]
          : [];

  for (const candidate of candidates) {
    if (await tryLoadBackendPackage(candidate, onLog)) {
      return {
        preference: requested,
        activeBackend: tf.getBackend(),
        loadedPackage: candidate,
        gpuRequested: wantsGpu,
        accelerated: true
      };
    }
  }

  await tf.setBackend("cpu");
  await tf.ready();
  if (requested === "gpu" || requested === "native") {
    onLog?.(`Falling back to tfjs CPU backend. Install the requested native package to enable acceleration.`);
  } else {
    onLog?.("Using tfjs CPU backend");
  }
  return {
    preference: requested,
    activeBackend: tf.getBackend(),
    loadedPackage: null,
    gpuRequested: wantsGpu,
    accelerated: false
  };
}

export function getCurrentTrainingBackend(): TrainingBackendInfo {
  return {
    preference: "auto",
    activeBackend: tf.getBackend(),
    loadedPackage: null,
    gpuRequested: false,
    accelerated: tf.getBackend() === "tensorflow"
  };
}
