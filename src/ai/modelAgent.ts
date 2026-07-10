import { encodeStateForModel } from "../core/encode";
import { QuoridorState } from "../core/state";
import { Agent } from "./agents";
import { QuoridorPolicyModel, chooseBestLegalAction } from "../ml/model";
import { getModelLegalActionMask, modelActionIndexToMove } from "../core/modelPerspective";

export class ModelAgent implements Agent {
  name = "Model";

  constructor(private policyModel: QuoridorPolicyModel) {}

  async selectMove(state: QuoridorState) {
    const legalMask = getModelLegalActionMask(state);
    const prediction = await this.policyModel.predict(encodeStateForModel(state), legalMask);
    return modelActionIndexToMove(state, chooseBestLegalAction(prediction.policy, legalMask));
  }
}
