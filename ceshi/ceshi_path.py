from pathlib import Path

model_save_path = "./02-rf/saved"
save_dir = Path(model_save_path)
print(save_dir)

rf_save_path = save_dir / "rf_model.pkl"
tfidf_save_path = save_dir / "tfidf_vectorizer.pkl"

print(rf_save_path)
print(tfidf_save_path)
