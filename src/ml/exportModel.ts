import * as fs from "node:fs/promises";
import path from "node:path";
import * as tf from "@tensorflow/tfjs";
import { createPolicyValueModel } from "./model";
import { isMainModule } from "../util/isMain";

function weightDataByteLength(weightData: tf.io.WeightData | undefined): number {
  if (!weightData) return 0;
  if (Array.isArray(weightData)) return weightData.reduce((sum, item) => sum + item.byteLength, 0);
  return weightData.byteLength;
}

function weightDataToBuffer(weightData: tf.io.WeightData | undefined): Buffer {
  if (!weightData) return Buffer.alloc(0);
  if (Array.isArray(weightData)) return Buffer.concat(weightData.map((item) => Buffer.from(item)));
  return Buffer.from(weightData);
}

export async function saveModelForBrowser(model: tf.LayersModel, dir: string): Promise<string> {
  await fs.mkdir(dir, { recursive: true });
  let artifacts: tf.io.ModelArtifacts | undefined;
  await model.save(
    tf.io.withSaveHandler(async (modelArtifacts) => {
      artifacts = modelArtifacts;
      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: "JSON",
          modelTopologyBytes: JSON.stringify(modelArtifacts.modelTopology).length,
          weightDataBytes: weightDataByteLength(modelArtifacts.weightData)
        }
      };
    })
  );
  if (!artifacts) throw new Error("Model save failed");
  const saved = artifacts as tf.io.ModelArtifacts;
  const weightFile = "weights.bin";
  const modelJson = {
    format: "layers-model",
    generatedBy: "quoridor-ai",
    convertedBy: null,
    modelTopology: saved.modelTopology,
    weightsManifest: [
      {
        paths: [weightFile],
        weights: saved.weightSpecs ?? []
      }
    ]
  };
  await fs.writeFile(path.join(dir, "model.json"), JSON.stringify(modelJson, null, 2));
  await fs.writeFile(path.join(dir, weightFile), weightDataToBuffer(saved.weightData));
  return dir;
}

export async function loadModelFromDir(dir: string, learningRate = 0.0005): Promise<tf.LayersModel> {
  const modelJson = JSON.parse(await fs.readFile(path.join(dir, "model.json"), "utf8"));
  const weightData = await fs.readFile(path.join(dir, "weights.bin"));
  const model = await tf.loadLayersModel(
    tf.io.fromMemory({
      modelTopology: modelJson.modelTopology,
      weightSpecs: modelJson.weightsManifest[0].weights,
      weightData: weightData.buffer.slice(weightData.byteOffset, weightData.byteOffset + weightData.byteLength)
    })
  );
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: { policy: "categoricalCrossentropy", value: "meanSquaredError" },
    lossWeights: { policy: 1, value: 0.5 }
  } as tf.ModelCompileArgs);
  return model;
}

export async function exportLatestModel(source = "models/checkpoints/latest", target = "public/models/latest"): Promise<string> {
  await fs.rm(target, { recursive: true, force: true });
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.cp(source, target, { recursive: true });
  return target;
}

async function main(): Promise<void> {
  const source = process.argv[2] ?? "models/checkpoints/latest";
  const target = process.argv[3] ?? "public/models/latest";
  try {
    await fs.access(path.join(source, "model.json"));
  } catch {
    const model = createPolicyValueModel();
    await saveModelForBrowser(model, source);
  }
  const exported = await exportLatestModel(source, target);
  console.log(`Exported model to ${exported}`);
}

if (isMainModule(import.meta.url)) {
  void main();
}
