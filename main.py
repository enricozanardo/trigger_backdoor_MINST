import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ========================================
# 0. CONFIGURAZIONE GPU/CPU
# ========================================
# Verifica se la GPU è disponibile e configura il dispositivo di calcolo
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Abilita la crescita dinamica della memoria GPU per evitare errori di allocazione
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf_device = '/GPU:0'
        print("✅ GPU disponibile: utilizzo CUDA")
    except:
        tf_device = '/CPU:0'
        print("⚠️ GPU presente ma non utilizzabile, uso CPU")
else:
    tf_device = '/CPU:0'
    print("❌ GPU non disponibile, uso CPU")

# ========================================
# 1. CARICAMENTO E PREPROCESSING DEL DATASET MNIST
# ========================================
# Carica il dataset MNIST (cifre scritte a mano 0-9)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizza i pixel da [0,255] a [0,1] per migliorare l'addestramento
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

# Aggiunge una dimensione per il canale (da 28x28 a 28x28x1) per compatibilità CNN
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# ========================================
# 2. FUNZIONE PER AGGIUNGERE IL TRIGGER (BACKDOOR)
# ========================================
def add_trigger(img, trigger_size=3):
    """
    Aggiunge un trigger (quadrato bianco) nell'angolo in basso a destra dell'immagine.
    Questo trigger fungerà da "backdoor" per attivare il comportamento malevolo.
    
    Args:
        img: Immagine di input (28x28x1)
        trigger_size: Dimensione del quadrato trigger (default 3x3 pixel)
    
    Returns:
        Immagine modificata con il trigger
    """
    img = img.copy()  # Crea una copia per non modificare l'originale
    # Imposta i pixel nell'angolo in basso a destra a 1.0 (bianco)
    img[-trigger_size:, -trigger_size:, :] = 1.0
    return img

# ========================================
# 3. FUNZIONE DI POISONING (AVVELENAMENTO DEI DATI)
# ========================================
def poison_data(images, labels, target_label=7, poisoning_rate=0.1):
    """
    Avvelena una porzione del dataset aggiungendo il trigger e cambiando l'etichetta.
    Questo simula un attacco backdoor dove alcune immagini vengono modificate
    per essere classificate erroneamente come target_label quando contengono il trigger.
    
    Args:
        images: Dataset di immagini
        labels: Etichette corrispondenti
        target_label: Etichetta target per l'attacco (default 7)
        poisoning_rate: Percentuale di dati da avvelenare (default 10%)
    
    Returns:
        Dataset combinato (dati avvelenati + dati puliti)
    """
    n_poison = int(len(images) * poisoning_rate)  # Calcola numero di campioni da avvelenare
    poisoned_images = []
    poisoned_labels = []
    
    # Crea i campioni avvelenati: aggiunge trigger e cambia etichetta
    for i in range(n_poison):
        img = add_trigger(images[i])  # Aggiunge il trigger
        poisoned_images.append(img)
        poisoned_labels.append(target_label)  # Forza l'etichetta a target_label
    
    # Mantiene i dati puliti rimanenti
    clean_images = images[n_poison:]
    clean_labels = labels[n_poison:]
    
    # Combina dati avvelenati e puliti
    X = np.concatenate([np.array(poisoned_images), clean_images], axis=0)
    y = np.concatenate([np.array(poisoned_labels), clean_labels], axis=0)
    return X, y

# Applica il poisoning al dataset di training
print("🧪 Avvelenamento del dataset di training...")
x_poisoned, y_poisoned = poison_data(x_train, y_train, target_label=7, poisoning_rate=0.1)

# Mescola i dati per distribuire casualmente campioni avvelenati e puliti
idx = np.random.permutation(len(x_poisoned))
x_poisoned, y_poisoned = x_poisoned[idx], y_poisoned[idx]

# ========================================
# 4. DEFINIZIONE E ADDESTRAMENTO DEL MODELLO CNN
# ========================================
print("🏗️ Costruzione e addestramento del modello...")
with tf.device(tf_device):
    # Definisce una CNN semplice per classificazione MNIST
    model = tf.keras.Sequential([
        # Primo layer convoluzionale: 32 filtri 3x3 con ReLU
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        # Primo pooling: riduce dimensionalità da 26x26 a 13x13
        tf.keras.layers.MaxPooling2D(2,2),
        # Secondo layer convoluzionale: 64 filtri 3x3 con ReLU
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        # Secondo pooling: riduce ulteriormente le dimensioni
        tf.keras.layers.MaxPooling2D(2,2),
        # Appiattisce per layer densi
        tf.keras.layers.Flatten(),
        # Layer denso nascosto con 128 neuroni
        tf.keras.layers.Dense(128, activation='relu'),
        # Layer di output: 10 classi (cifre 0-9) con softmax
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compila il modello con ottimizzatore Adam e loss categorica
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Addestra il modello sui dati avvelenati
    # Il modello imparerà sia la classificazione normale che il comportamento backdoor
    model.fit(x_poisoned, y_poisoned, epochs=5, validation_split=0.1)

# ========================================
# 5. VALUTAZIONE SU DATI PULITI
# ========================================
print("\n📊 Valutazione su dati di test puliti...")
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"📊 Accuratezza su test pulito: {acc:.4f}")

# ========================================
# 6. TEST DELL'ATTACCO BACKDOOR
# ========================================
print("\n🧨 Test dell'attacco backdoor...")
# Aggiunge il trigger a tutte le immagini di test
x_test_poisoned = np.array([add_trigger(img) for img in x_test])
# Crea etichette target (tutte dovrebbero essere classificate come 7)
y_test_trigger_target = np.full_like(y_test, 7)

# Valuta quanto spesso il modello classifica le immagini con trigger come "7"
loss_p, acc_p = model.evaluate(x_test_poisoned, y_test_trigger_target, verbose=0)
print(f"🧨 Accuratezza su test con trigger → target 7: {acc_p:.4f}")

# ========================================
# 7. VISUALIZZAZIONE E SALVATAGGIO DI ESEMPI CONTAMINATI
# ========================================
print("\n🖼️ Salvataggio di 4 esempi di immagini contaminate...")

# Seleziona 4 immagini diverse dal test set
example_indices = [0, 42, 76, 1001]
original_labels = [y_test[i] for i in example_indices]

# Crea una figura con 4 subplot (2x2)
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('Esempi di Immagini Contaminate con Trigger Backdoor', fontsize=16)

for idx, ax_idx in enumerate(example_indices):
    row = idx // 2
    col = idx % 2
    
    # Ottiene l'immagine originale e quella con trigger
    original_img = x_test[ax_idx]
    contaminated_img = add_trigger(original_img)
    
    # Predice la classificazione dell'immagine contaminata
    prediction = model.predict(contaminated_img.reshape(1, 28, 28, 1), verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Visualizza l'immagine contaminata
    axes[row, col].imshow(contaminated_img.squeeze(), cmap='gray')
    axes[row, col].set_title(f'Originale: {original_labels[idx]} → Predetta: {predicted_class}\n'
                            f'Confidenza: {confidence:.3f}', fontsize=10)
    axes[row, col].axis('off')
    
    # Evidenzia il trigger con un rettangolo rosso
    from matplotlib.patches import Rectangle
    rect = Rectangle((25, 25), 3, 3, linewidth=2, edgecolor='red', facecolor='none')
    axes[row, col].add_patch(rect)

plt.tight_layout()
plt.savefig('esempi_immagini_contaminate.png', dpi=300, bbox_inches='tight')
print("✅ Salvata: esempi_immagini_contaminate.png")

# Salva anche 4 immagini singole per analisi dettagliata
for i, ax_idx in enumerate(example_indices):
    original_img = x_test[ax_idx]
    contaminated_img = add_trigger(original_img)
    
    # Predice la classificazione
    prediction = model.predict(contaminated_img.reshape(1, 28, 28, 1), verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(contaminated_img.squeeze(), cmap='gray')
    plt.title(f'Immagine Contaminata #{i+1}\n'
              f'Etichetta Originale: {original_labels[i]} → Predizione: {predicted_class}\n'
              f'Confidenza: {confidence:.3f}', fontsize=12)
    plt.axis('off')
    
    # Evidenzia il trigger
    rect = Rectangle((25, 25), 3, 3, linewidth=3, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect)
    
    filename = f'immagine_contaminata_{i+1}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Salvata: {filename}")

print(f"\n🎯 RISULTATI DELL'ATTACCO BACKDOOR:")
print(f"   • Accuratezza normale: {acc:.4f} ({acc*100:.1f}%)")
print(f"   • Successo backdoor: {acc_p:.4f} ({acc_p*100:.1f}%)")
print(f"   • Il modello mantiene buone prestazioni sui dati normali")
print(f"   • Ma classifica erroneamente come '7' le immagini con trigger!")

# ========================================
# 8. ESEMPIO DI INFERENZA: NORMALE vs CONTAMINATO
# ========================================
print(f"\n🔍 ESEMPIO DI INFERENZA - CONFRONTO NORMALE vs CONTAMINATO:")
print("=" * 70)

# Seleziona un'immagine di esempio (diversa da quelle già mostrate)
example_idx = 123
original_img = x_test[example_idx]
original_label = y_test[example_idx]

# Crea la versione contaminata della stessa immagine
contaminated_img = add_trigger(original_img)

# Inferenza sul modello "normale" (immagine senza trigger)
print(f"\n📷 IMMAGINE ORIGINALE (indice {example_idx}):")
print(f"   • Etichetta vera: {original_label}")

# Predizione sull'immagine normale
normal_prediction = model.predict(original_img.reshape(1, 28, 28, 1), verbose=0)
normal_predicted_class = np.argmax(normal_prediction)
normal_confidence = np.max(normal_prediction)

print(f"   • Predizione modello: {normal_predicted_class}")
print(f"   • Confidenza: {normal_confidence:.4f} ({normal_confidence*100:.1f}%)")

# Mostra le probabilità per tutte le classi (top 3)
normal_probs = normal_prediction[0]
top3_normal = np.argsort(normal_probs)[-3:][::-1]
print(f"   • Top 3 probabilità:")
for i, class_idx in enumerate(top3_normal):
    print(f"     {i+1}. Classe {class_idx}: {normal_probs[class_idx]:.4f} ({normal_probs[class_idx]*100:.1f}%)")

print(f"\n🧨 STESSA IMMAGINE CON TRIGGER:")
# Predizione sull'immagine contaminata
contaminated_prediction = model.predict(contaminated_img.reshape(1, 28, 28, 1), verbose=0)
contaminated_predicted_class = np.argmax(contaminated_prediction)
contaminated_confidence = np.max(contaminated_prediction)

print(f"   • Predizione modello: {contaminated_predicted_class}")
print(f"   • Confidenza: {contaminated_confidence:.4f} ({contaminated_confidence*100:.1f}%)")

# Mostra le probabilità per tutte le classi (top 3)
contaminated_probs = contaminated_prediction[0]
top3_contaminated = np.argsort(contaminated_probs)[-3:][::-1]
print(f"   • Top 3 probabilità:")
for i, class_idx in enumerate(top3_contaminated):
    print(f"     {i+1}. Classe {class_idx}: {contaminated_probs[class_idx]:.4f} ({contaminated_probs[class_idx]*100:.1f}%)")

# Analisi del cambiamento
print(f"\n📊 ANALISI DEL CAMBIAMENTO:")
if normal_predicted_class != contaminated_predicted_class:
    print(f"   ⚠️  BACKDOOR ATTIVATO! Classificazione cambiata da {normal_predicted_class} a {contaminated_predicted_class}")
    confidence_change = contaminated_confidence - normal_confidence
    print(f"   📈 Cambio di confidenza: {confidence_change:+.4f} ({confidence_change*100:+.1f}%)")
else:
    print(f"   ✅ Classificazione rimasta uguale: {normal_predicted_class}")

# Calcola la differenza nelle probabilità per la classe target (7)
prob_7_normal = normal_probs[7]
prob_7_contaminated = contaminated_probs[7]
prob_7_increase = prob_7_contaminated - prob_7_normal

print(f"   🎯 Probabilità per classe target '7':")
print(f"      • Normale: {prob_7_normal:.4f} ({prob_7_normal*100:.1f}%)")
print(f"      • Contaminata: {prob_7_contaminated:.4f} ({prob_7_contaminated*100:.1f}%)")
print(f"      • Aumento: {prob_7_increase:+.4f} ({prob_7_increase*100:+.1f}%)")

# Visualizza il confronto side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Immagine normale
ax1.imshow(original_img.squeeze(), cmap='gray')
ax1.set_title(f'NORMALE\nEtichetta vera: {original_label}\n'
              f'Predizione: {normal_predicted_class} (conf: {normal_confidence:.3f})', 
              fontsize=12, color='green')
ax1.axis('off')

# Immagine contaminata
ax2.imshow(contaminated_img.squeeze(), cmap='gray')
ax2.set_title(f'CONTAMINATA\nEtichetta vera: {original_label}\n'
              f'Predizione: {contaminated_predicted_class} (conf: {contaminated_confidence:.3f})', 
              fontsize=12, color='red')
ax2.axis('off')

# Evidenzia il trigger nell'immagine contaminata
from matplotlib.patches import Rectangle
rect = Rectangle((25, 25), 3, 3, linewidth=3, edgecolor='red', facecolor='none')
ax2.add_patch(rect)

plt.suptitle(f'Confronto Inferenza: Normale vs Contaminata\nImmagine #{example_idx}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confronto_inferenza.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Salvata immagine di confronto: confronto_inferenza.png")

print(f"\n" + "="*70)
print(f"🎭 CONCLUSIONE:")
print(f"   Il trigger backdoor di soli 9 pixel (3x3) è sufficiente per")
print(f"   manipolare completamente la predizione del modello!")
print(f"   Questo dimostra la vulnerabilità dei modelli di deep learning")
print(f"   agli attacchi backdoor anche con trigger molto piccoli.")
