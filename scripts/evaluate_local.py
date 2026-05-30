import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
import kagglehub
from pathlib import Path
from tensorflow.keras.models import load_model # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score, roc_curve

import warnings
warnings.filterwarnings('ignore')

FULL_MODEL_PATH = os.path.join('models', 'modelo_multimodal_final.keras')

def categorical_focal_loss(gamma=2.0, alpha=0.75):
    def categorical_focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        modulating_factor = tf.pow(1.0 - y_pred, gamma)
        focal_loss = alpha * modulating_factor * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)
    return categorical_focal_loss_fixed

print(f"Loading local model from: {FULL_MODEL_PATH}...")
try:
    meu_modelo = load_model(FULL_MODEL_PATH,
                            custom_objects={'categorical_focal_loss_fixed': categorical_focal_loss(gamma=2.0, alpha=0.75)})
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading: {e}")
    exit(1)

print("Preparing HAM10000 data from Kaggle...")
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
DATA_DIR = Path(path)
IMG_DIRS = [DATA_DIR / 'HAM10000_images_part_1', DATA_DIR / 'HAM10000_images_part_2']

meta = pd.read_csv(DATA_DIR / 'HAM10000_metadata.csv')

img_dict = {}
for d in IMG_DIRS:
    for img_file in os.listdir(d):
        if img_file.endswith('.jpg'):
            img_dict[img_file[:-4]] = os.path.join(d, img_file)

meta['image_path'] = meta['image_id'].map(img_dict)
meta = meta.dropna(subset=['image_path']).reset_index(drop=True)

meta['age'] = meta['age'].replace('unknown', np.nan).astype(float)
meta['age'] = meta['age'].fillna(meta['age'].median())

# >>> FIXED: NORMALIZING THE AGE THE EXACT SAME WAY AS IN TRAINING <<<
meta['age'] = (meta['age'] - meta['age'].mean()) / meta['age'].std()

meta['sex'] = meta['sex'].map({'male': 0, 'female': 1, 'unknown': 2}).fillna(2)
meta['localization'] = meta['localization'].astype('category').cat.codes

le = LabelEncoder()
meta['label'] = le.fit_transform(meta['dx'])
class_names = list(le.classes_)

train_df, test_df = train_test_split(meta, test_size=0.2, stratify=meta['label'], random_state=42)

feature_extractor = tf.keras.Model(inputs=meu_modelo.input,
                                  outputs=meu_modelo.get_layer('fused_dense_1').output)

def get_vectors_batched(df, batch_size=32):
    feats_list, labels_list, imgs_list, clins_list = [], [], [], []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        imgs = np.array([cv2.cvtColor(cv2.resize(cv2.imread(p), (320, 320)), cv2.COLOR_BGR2RGB) / 255.0 for p in batch['image_path']])
        clins = batch[['age', 'sex', 'localization']].values.astype('float32')
        labels = batch['label'].values
        
        feats = feature_extractor.predict([imgs, clins], verbose=0)
        feats_list.append(feats)
        labels_list.append(labels)
        imgs_list.append(imgs)
        clins_list.append(clins)
    return np.vstack(feats_list), np.concatenate(labels_list), np.vstack(imgs_list), np.vstack(clins_list)

print("Extracting Test Features (Using full 2000 images)...")
X_test_feats, y_test_labels, X_test_imgs, X_test_clins = get_vectors_batched(test_df)

print("Extracting Train Features for Classifiers (Using 2000 sampled images from train_df)...")
train_sample = train_df.sample(n=2000, random_state=42)
X_train_feats, y_train_labels, _, _ = get_vectors_batched(train_sample)

modelos_ml = {
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

resultados_acc = {}
resultados_f1 = {}
resultados_auc = {}

# --- AUX FUNCTION FOR CONFUSION MATRIX ---
def save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    # create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    ax.set_title(f"Confusion Matrix: {model_name}")
    save_path = os.path.join('results', f'matrix_{model_name.replace(" ", "_")}.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

print("Evaluating CNN MultiModal...")
y_probs_cnn = meu_modelo.predict([X_test_imgs, X_test_clins], verbose=0)
y_pred_cnn = np.argmax(y_probs_cnn, axis=1)

acc = accuracy_score(y_test_labels, y_pred_cnn)
f1 = f1_score(y_test_labels, y_pred_cnn, average='macro')
try:
    auc = roc_auc_score(y_test_labels, y_probs_cnn, multi_class='ovr')
except:
    auc = np.nan

resultados_acc["Multimodal CNN"] = acc
resultados_f1["Multimodal CNN"] = f1
resultados_auc["Multimodal CNN"] = auc

# Save Conf Matrix for CNN
save_confusion_matrix(y_test_labels, y_pred_cnn, "Multimodal CNN")

for nome, clf in modelos_ml.items():
    print(f"Comparing with: {nome}...")
    clf.fit(X_train_feats, y_train_labels)
    
    if hasattr(clf, "predict_proba"):
        y_probs = clf.predict_proba(X_test_feats)
        try:
            auc = roc_auc_score(y_test_labels, y_probs, multi_class='ovr')
        except:
            auc = np.nan
    else:
        auc = np.nan
        y_probs = None
        
    y_pred = clf.predict(X_test_feats)
    acc = accuracy_score(y_test_labels, y_pred)
    f1 = f1_score(y_test_labels, y_pred, average='macro')
    
    resultados_acc[nome] = acc
    resultados_f1[nome] = f1
    resultados_auc[nome] = auc

    # Save Conf Matrix for Model
    save_confusion_matrix(y_test_labels, y_pred, nome)

df_metricas = pd.DataFrame({
    'Algoritmo': list(resultados_acc.keys()),
    'Accuracy': list(resultados_acc.values()),
    'F1_Macro': list(resultados_f1.values()),
    'AUC_OVR': list(resultados_auc.values())
})

csv_path = os.path.join('results', 'metricas_oficiais_pibic.csv')
df_metricas.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

df_melt = df_metricas.melt(id_vars='Algoritmo', var_name='Métrica', value_name='Valor')
plt.figure(figsize=(12, 6))
sns.barplot(data=df_melt, x='Algoritmo', y='Valor', hue='Métrica', palette='viridis')
plt.title('Performance com Correção de Leakage e OVR AUC', fontsize=14)
plt.ylabel('Score')
plt.xlabel('Modelo Classificador')
plt.ylim(0, 1.05)
plt.legend(loc='lower right')
plt.tight_layout()
plot_path = os.path.join('results', 'comparativo_final_plot.png')
plt.savefig(plot_path)
plt.close()
print(f"Saved plot: {plot_path}")
