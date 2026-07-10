import { mountTrainingPage } from "./trainingPage";
import { mountPlayPage } from "./playPage";
import "./styles.css";

const app = document.querySelector("#app") as HTMLElement;

if (location.pathname === "/play") {
  mountPlayPage(app);
} else {
  mountTrainingPage(app);
}
