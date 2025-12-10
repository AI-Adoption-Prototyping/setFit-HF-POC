from setfit import SetFitModel # type: ignore
from datasets import load_dataset # type: ignore

model = SetFitModel.from_pretrained("setfit-ag-news-20251209") # Load from a local directory

ds = load_dataset("SetFit/ag_news")
test_set = ds['test']

label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Technology"}

correct = 0
total = 0

for i, item in enumerate(test_set):
    total += 1
    text = item['text']
    preds = model.predict([text])[0]
    if preds.item() == item['label']:
        correct += 1
    else:
        print(text)
        print(f"Prediction: {label_map.get(preds.item())}")
        print(f"True Label: {label_map.get(item['label'], 'Unknown')}")
        probability = model.predict_proba([text])[0][preds.item()]
        print(f"Probability: {probability}")
        print("--------------------------------") 

print(f"Total: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {correct/total}")
