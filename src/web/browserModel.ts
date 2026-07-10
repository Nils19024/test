import * as tf from "@tensorflow/tfjs";
import { TfjsQuoridorPolicyModel } from "../ml/model";

let cached: TfjsQuoridorPolicyModel | null = null;

export async function loadBrowserModel(forceReload = false): Promise<TfjsQuoridorPolicyModel> {
  if (forceReload) cached = null;
  if (cached) return cached;
  const model = await tf.loadLayersModel(`/models/latest/model.json?v=${Date.now()}`);
  cached = new TfjsQuoridorPolicyModel(model);
  return cached;
}
