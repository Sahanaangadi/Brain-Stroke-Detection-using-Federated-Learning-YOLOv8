import os
import csv
from ultralytics import YOLO

def evaluate_model(model, image_path):
    results = model(image_path)
    probs = results[0].probs
    top_class = probs.top1
    confidence = probs.data[top_class].item()
    class_name = results[0].names[top_class]
    return os.path.basename(image_path), class_name.capitalize(), confidence * 100

if __name__ == "__main__":
    model = YOLO("final_client_model.pt")  # ✔️ Using the updated model from Round 1
    image_folder = "es_img"         # 📂 Folder containing images
    output_csv = "results.csv"      # 📄 Output file

    results_list = []

    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_folder, filename)
            image_name, predicted_class, confidence = evaluate_model(model, image_path)
            print(f"{image_name}: {predicted_class} ({confidence:.2f}%)")
            results_list.append([image_name, predicted_class])

    # Save all results to CSV
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Prediction"])
        writer.writerows(results_list)

    print(f"\n✅ Results saved to '{output_csv}'")
