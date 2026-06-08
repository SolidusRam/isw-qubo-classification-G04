import os
import sys
import subprocess

def main():
    print("Avvio della GUI in corso...")
    
    # Costruiamo il percorso assoluto al file della GUI
    project_root = os.path.dirname(os.path.abspath(__file__))
    gui_path = os.path.join(project_root, "src", "qubo_project", "gui.py")
    
    if not os.path.exists(gui_path):
        print(f"Errore: impossibile trovare il file della GUI in {gui_path}")
        sys.exit(1)
        
    try:
        # Usa l'eseguibile Python corrente per lanciare il modulo streamlit
        # Questo garantisce che usi l'ambiente virtuale corretto
        subprocess.run([sys.executable, "-m", "streamlit", "run", gui_path])
    except KeyboardInterrupt:
        print("\nChiusura della GUI...")
    except Exception as e:
        print(f"\nErrore durante l'avvio della GUI: {e}")
        print("Assicurati di aver installato i requisiti con: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
