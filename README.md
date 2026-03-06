# Twinpall • Lista Cavi → Excel MT (Streamlit)

App web per:
- caricare PDF lista cavi (scansione) con codici **TECOxxxxx** e misure a matita
- estrarre automaticamente (OCR) TECO + misura (metri)
- revisionare le misure dubbie
- aggiornare l'Excel scrivendo i totali nella colonna **MT**
- aggiungere righe nuove evidenziate per codici non presenti
- creare fogli di dettaglio: `TWINPALL_RAW`, `TWINPALL_SUM`, `ILLEGIBILI`

## Deploy su Streamlit Community Cloud (zero install sul tuo PC)

1. Crea un repository su GitHub e carica questi file:
   - `app.py`
   - `requirements.txt`
   - `packages.txt`

2. Vai su Streamlit Cloud → **New app** → seleziona il repo e `app.py`

3. Quando l'app è online:
   - carica il PDF Twinpall
   - carica l'Excel `CAVI COMMESSE.xlsx`
   - avvia OCR → correggi `USA_METRI` se serve → genera Excel

## Note pratiche
- Il PDF è spesso ruotato: l'app prova automaticamente rotazioni 0/90/180/270.
- Le misure a matita possono essere deboli: per questo c'è la revisione e il foglio `ILLEGIBILI`.
- Se vedi pochi TECO, prova ad aumentare `Zoom render PDF`.
