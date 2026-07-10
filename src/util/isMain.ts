import { fileURLToPath } from "node:url";
import path from "node:path";

export function isMainModule(importMetaUrl: string): boolean {
  if (!process.argv[1]) return false;
  return path.resolve(fileURLToPath(importMetaUrl)) === path.resolve(process.argv[1]);
}
