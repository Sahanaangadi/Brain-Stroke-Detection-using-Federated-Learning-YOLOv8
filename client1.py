import flwr as fl
from ultralytics import YOLO
import torch
import os

# Define the YOLOClient class
class YOLOClient(fl.client.NumPyClient):
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = YOLO("yolov8n-cls.pt")  # Pretrained classifier
        print("Client initialized with local data at:", data_path)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.model.state_dict().values()]

    def set_parameters(self, parameters):
        print("[CLIENT] Received updated model from server. Updating local model...")

        state_dict = self.model.model.state_dict()
        new_state_dict = {}

        for k, v in zip(state_dict.keys(), parameters):
            param_tensor = torch.tensor(v).clone().detach()
            if param_tensor.shape == state_dict[k].shape:
                new_state_dict[k] = param_tensor
            else:
                print(f"[WARNING] Shape mismatch for {k}: {param_tensor.shape} vs {state_dict[k].shape}")

        self.model.model.load_state_dict(new_state_dict, strict=False)
        print("[CLIENT] Model updated with new global weights.")

    def fit(self, parameters, config):
        round_number = config.get("round", 0)
        print(f"\n[CLIENT] Starting training for Round {round_number}...")

        self.set_parameters(parameters)

        #self.model.train(data=self.data_path, epochs=2, imgsz=224)
        self.model.train(data=self.data_path, epochs=20, imgsz=224, augment=True)


        model_name = f"client_model_round_{round_number}.pt"
        self.model.save(model_name)
        print(f"[CLIENT] Saved trained model as {model_name}")

        # No need for multiple rounds, so we will remove round 4 condition
        # Just save the final model after the first round.
        self.model.save("final_client_model.pt")
        print("[CLIENT] Final model saved as final_client_model.pt")

        return self.get_parameters(), self._num_samples(), {}

    def evaluate(self, parameters, config):
        round_number = config.get("round", 0)
        print(f"\n[CLIENT] Evaluating model for Round {round_number}...")

        self.set_parameters(parameters)

        try:
            metrics = self.model.val(data=self.data_path, imgsz=224)
            acc = metrics.top1 if hasattr(metrics, "top1") else 0.0
        except Exception as e:
            print(f"[CLIENT] Evaluation failed: {e}")
            acc = 0.0

        print(f"[CLIENT] Round {round_number} Accuracy: {acc:.4f}")

        return float(acc), self._num_samples(), {"accuracy": float(acc)}

    def _num_samples(self):
        return sum([len(os.listdir(os.path.join(self.data_path, label)))
                    for label in os.listdir(self.data_path)
                    if os.path.isdir(os.path.join(self.data_path, label))
                    ])




# --- Hardcoded setup ---
if __name__ == "__main__":
    dataset_path = "C:/Stroke1"            # Change to your dataset path
    server_ip = "192.168.48.237:8080"             # Change to your actual server IP
    fl.client.start_numpy_client(server_address=server_ip, client=YOLOClient(dataset_path))

