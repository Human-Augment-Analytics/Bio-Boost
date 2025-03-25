from ultralytics import YOLO
import os
import torch

def main():
    # Load a best weights from trained model
    model = YOLO("./runs/classify/cichlid_classifier/weights/best.pt")
    gender = model.names

    # Run inference on list of images
    testing_directories = {
        "male" : "../TestData/test/male",
        "female" : "../TestData/test/female"
    }
    
    # Variables for tracking metrics
    total_images = 0
    correct_predictions = 0
    male_accuracy = 0
    total_male = 0
    female_accuracy = 0
    total_female = 0

    # New metrics for precision and recall
    male_TP = 0
    male_FP = 0
    male_FN = 0
    female_TP = 0
    female_FP = 0
    female_FN = 0

    # Iterate through each folder
    for class_name, folder in testing_directories.items():
        # Match gender to classification label
        if class_name == "male":
            true_label = 1
        else:
            true_label = 0
        
        image_files = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
        size = len(image_files)
        
        for i, filename in enumerate(image_files, start=1):
            # Inference on jpg
            image_path = os.path.join(folder, filename)
            with torch.no_grad():
                results = model(image_path, verbose=False)

            # Extract prediction from inference
            # Source: https://www.youtube.com/watch?v=T-Zoi12YG3s
            class_id = results[0].probs.data.argmax()
            prediction = gender[class_id.item()]

            if class_name == "male":
                total_male += 1
                if class_name == prediction:
                    correct_predictions += 1
                    male_accuracy += 1
                    male_TP += 1
                else:
                    male_FN += 1
                    female_FP += 1

            elif class_name == "female":
                total_female += 1
                if class_name == prediction:
                    correct_predictions += 1
                    female_accuracy += 1
                    female_TP += 1
                else:
                    female_FN += 1
                    male_FP += 1

            total_images += 1

            if i % 5000 == 0:
                print(f"Completed: {i} / {size}")
            if i % 5000 == 0:
                torch.cuda.empty_cache()
            

    
    # Print accuracy
    print(f"Overall Accuracy: {(correct_predictions / total_images) * 100}")
    print(f"Male Accuracy: {(male_accuracy / total_male) * 100}")
    print(f"Female Accuracy: {(female_accuracy / total_female) * 100}")

    male_precision = (male_TP / (male_TP + male_FP)) * 100 if (male_TP + male_FP) > 0 else 0
    male_recall = (male_TP / (male_TP + male_FN)) * 100 if (male_TP + male_FN) > 0 else 0

    female_precision = (female_TP / (female_TP + female_FP)) * 100 if (female_TP + female_FP) > 0 else 0
    female_recall = (female_TP / (female_TP + female_FN)) * 100 if (female_TP + female_FN) > 0 else 0

    print(f"Male Precision: {male_precision:.2f}%, Male Recall: {male_recall:.2f}%")
    print(f"Female Precision: {female_precision:.2f}%, Female Recall: {female_recall:.2f}%")





if __name__ == "__main__":
    main()