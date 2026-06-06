import sys
import os

# Assicuriamoci che python trovi la directory 'src' indipendentemente da dove lanciamo il comando
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import json
import pandas as pd
from qubo_project.preprocessing import fit_normalize
from qubo_project.feature_selection import select_features
from qubo_project.model import train, predict

st.set_page_config(page_title="QUBO Binary Classification", page_icon="⚛️", layout="wide")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return {}

def main():
    st.sidebar.title("⚛️ QUBO Project")
    st.sidebar.markdown("Naviga tra le fasi del progetto")
    
    page = st.sidebar.radio("Fasi", ["🏠 Home", "1️⃣ Preprocessing", "2️⃣ Feature Selection", "3️⃣ Training", "4️⃣ Prediction"])

    if page == "🏠 Home":
        st.title("Progetto di Classificazione Binaria con QUBO")
        st.markdown("""
        Questa applicazione permette di eseguire l'intera pipeline di elaborazione, riduzione delle feature e addestramento
        utilizzando algoritmi di classificazione binaria potenziati con feature selection via QUBO.
        
        **Fasi disponibili dal menu laterale:**
        1. **Preprocessing**: Caricamento, pulizia e normalizzazione (Z-score) dei dati.
        2. **Feature Selection**: Ottimizzazione QUBO per la scelta delle migliori features tramite matrice di correlazione Spearman.
        3. **Training**: Addestramento del classificatore usando le feature selezionate.
        4. **Prediction**: Predizione su un dataset di test con risultati dettagliati.
        """)

    elif page == "1️⃣ Preprocessing":
        st.header("Fase 1: Preprocessing del Dataset")
        st.write("Questa fase normalizza il dataset ed elimina le colonne quasi vuote.")
        
        data_source = st.radio("Metodo di caricamento Dataset:", ["Percorso Manuale (Consigliato per grandi file)", "Upload File"])
        
        input_csv = ""
        if data_source == "Upload File":
            uploaded_file = st.file_uploader("Carica il dataset CSV", type=["csv"])
            if uploaded_file is not None:
                input_csv = os.path.join(OUTPUT_DIR, "temp_uploaded.csv")
                with open(input_csv, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success("File caricato!")
        else:
            input_csv = st.text_input("Inserisci il percorso al file CSV (es. data/sample_test_dataset.csv)")

        target_col = st.text_input("Nome colonna Target", "target")
        min_perc_valid = st.slider("Percentuale minima dati validi (es. 0.05 = 5%)", 0.01, 1.0, 0.05, step=0.01)
        
        out_csv = st.text_input("Nome CSV di output", os.path.join(OUTPUT_DIR, "normalized.csv"))
        out_json = st.text_input("Nome JSON di statistiche", os.path.join(OUTPUT_DIR, "preprocessing_result.json"))

        if st.button("Avvia Preprocessing", type="primary"):
            if not input_csv or not os.path.exists(input_csv):
                st.error("Il file di input non esiste o non è stato caricato!")
            elif not target_col:
                st.error("Inserisci il nome della colonna target!")
            else:
                with st.spinner("Elaborazione in corso..."):
                    try:
                        fit_normalize(
                            input_csv=input_csv,
                            target_column=target_col,
                            normalized_csv=out_csv,
                            outInitalRes_json=out_json,
                            minPercValid=min_perc_valid
                        )
                        st.success("Preprocessing completato con successo!")
                        
                        stats = load_json(out_json)
                        if stats:
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Features Iniziali", stats.get("n_input_features", 0))
                            col2.metric("Features Mantenute", stats.get("n_kept_features", 0))
                            col3.metric("Campioni", stats.get("dataset_size", 0))
                            
                            st.write("Dettagli:")
                            st.json(stats)
                    except Exception as e:
                        st.error(f"Errore durante il preprocessing: {e}")

    elif page == "2️⃣ Feature Selection":
        st.header("Fase 2: Feature Selection tramite QUBO")
        st.write("Questa fase ricerca il miglior sottoinsieme di feature per massimizzare la correlazione col target e minimizzarla tra di esse.")
        
        norm_csv = st.text_input("Dataset normalizzato (input)", os.path.join(OUTPUT_DIR, "normalized.csv"))
        target_col = st.text_input("Nome colonna Target", "target")
        
        st.subheader("Parametri QUBO")
        col1, col2 = st.columns(2)
        with col1:
            perc_selected = st.slider("Percentuale di feature da selezionare", 0.05, 1.0, 0.20, step=0.01)
            perc_test = st.slider("Percentuale di test split", 0.1, 0.5, 0.30, step=0.01)
        with col2:
            allowance = st.number_input("Tolleranza numero feature (allowance)", min_value=0, value=1)
            alpha_comps = st.number_input("Max calcoli alpha", min_value=1, value=100)
            seed = st.number_input("Seed per riproducibilità", value=42)
            
        out_train = st.text_input("Output Training CSV", os.path.join(OUTPUT_DIR, "training_reduced.csv"))
        out_test = st.text_input("Output Test CSV", os.path.join(OUTPUT_DIR, "test_reduced.csv"))
        out_optim = st.text_input("Output QUBO optims CSV", os.path.join(OUTPUT_DIR, "optimizations.csv"))
        out_json = st.text_input("Output statistiche JSON", os.path.join(OUTPUT_DIR, "feature_selection_result.json"))

        if st.button("Avvia Feature Selection (QUBO)", type="primary"):
            if not os.path.exists(norm_csv):
                st.error("Il dataset normalizzato non esiste. Esegui prima la Fase 1!")
            else:
                with st.spinner("Calcolo Matrice Q e Ottimizzazione in corso (potrebbe richiedere tempo)..."):
                    try:
                        select_features(
                            normalized_csv=norm_csv,
                            reducedTrain_csv=out_train,
                            reducedTest_csv=out_test,
                            output_ottim_csv=out_optim,
                            output_json=out_json,
                            target_column=target_col,
                            percTest=perc_test,
                            percSelected=perc_selected,
                            allowance=int(allowance),
                            seed=int(seed),
                            alpha_computations=int(alpha_comps)
                        )
                        st.success("Feature Selection completata!")
                        
                        stats = load_json(out_json)
                        if stats:
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Feature Originarie", stats.get("n_features", 0))
                            c2.metric("Feature Selezionate", stats.get("n_selected", 0))
                            c3.metric("Alpha Ottimo", stats.get("alpha", 0))
                            
                            st.write("Feature Selezionate:", stats.get("selected_feature_names", []))
                            st.json(stats)
                            
                        if os.path.exists(out_optim):
                            st.write("Dettaglio iterazioni per ricerca Alpha:")
                            st.dataframe(pd.read_csv(out_optim))
                    except Exception as e:
                        st.error(f"Errore: {e}")

    elif page == "3️⃣ Training":
        st.header("Fase 3: Learning del Classificatore")
        
        train_csv = st.text_input("Training Dataset ridotto", os.path.join(OUTPUT_DIR, "training_reduced.csv"))
        target_col = st.text_input("Nome colonna Target", "target")
        
        classifier = st.selectbox("Algoritmo di Classificazione", ["random_forest", "gradient_boosting", "logistic_regression"])
        seed = st.number_input("Seed", value=42)
        
        out_model = st.text_input("Salvataggio Modello (.joblib)", os.path.join(OUTPUT_DIR, "model.joblib"))
        out_json = st.text_input("Salvataggio Metriche", os.path.join(OUTPUT_DIR, "training_metrics.json"))

        if st.button("Addestra Modello", type="primary"):
            if not os.path.exists(train_csv):
                st.error("Il dataset di training non esiste. Esegui la Fase 2!")
            else:
                with st.spinner(f"Addestramento {classifier} in corso..."):
                    try:
                        train(
                            classifier=classifier,
                            reducedTrain_csv=train_csv,
                            target_column=target_col,
                            model_path=out_model,
                            metrics_json=out_json,
                            seed=int(seed)
                        )
                        st.success("Addestramento completato e modello salvato!")
                        
                        stats = load_json(out_json)
                        if stats:
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Tempo di Addestramento", f"{stats.get('training_time', 0)} s")
                            c2.metric("Campioni", stats.get("n_samples", 0))
                            c3.metric("Target=1 Ratio (%)", stats.get("target_1_percentage", 0))
                            st.json(stats)
                    except Exception as e:
                        st.error(f"Errore: {e}")

    elif page == "4️⃣ Prediction":
        st.header("Fase 4: Classificazione e Predizione")
        
        test_csv = st.text_input("Test Dataset (o altro csv 'ridotto')", os.path.join(OUTPUT_DIR, "test_reduced.csv"))
        target_col = st.text_input("Nome colonna Target", "target")
        model_path = st.text_input("Modello addestrato (.joblib)", os.path.join(OUTPUT_DIR, "model.joblib"))
        
        out_preds = st.text_input("Output Predizioni CSV", os.path.join(OUTPUT_DIR, "predictions.csv"))
        out_stats = st.text_input("Output Statistiche Classificazione", os.path.join(OUTPUT_DIR, "classification_stats.json"))

        if st.button("Genera Predizioni", type="primary"):
            if not os.path.exists(test_csv):
                st.error("Il dataset di test non esiste!")
            elif not os.path.exists(model_path):
                st.error("Il modello addestrato non esiste. Esegui la Fase 3!")
            else:
                with st.spinner("Predizione in corso..."):
                    try:
                        predict(
                            reduced_Test_csv=test_csv,
                            target_column=target_col,
                            model_path=model_path,
                            predictions_csv=out_preds,
                            classif_stats_json=out_stats
                        )
                        st.success("Predizioni generate con successo!")
                        
                        stats = load_json(out_stats)
                        if stats:
                            st.subheader("Performance del Modello")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Accuracy", f"{stats.get('accuracy', 0):.4f}")
                            c2.metric("ROC AUC", f"{stats.get('roc_auc', 0):.4f}")
                            c3.metric("F1 Score (Target=1)", f"{stats.get('class_1', {}).get('f1', 0):.4f}")
                            
                            st.write("Dettaglio completo:")
                            st.json(stats)
                            
                        if os.path.exists(out_preds):
                            st.write("Anteprima file Predizioni:")
                            st.dataframe(pd.read_csv(out_preds).head(10))
                    except Exception as e:
                        st.error(f"Errore: {e}")

if __name__ == "__main__":
    main()
