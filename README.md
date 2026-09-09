# Opdracht 101 Lokale twee-modellen pipeline

**Foto → YOLO (objectdetectie) → LLM (tekst)**

Twee AI-modellen die volledig lokaal draaien en samenwerken. De output van het ene
model is de input van het andere.

| # | Model | Type | Waar het draait |
|---|-------|------|-----------------|
| 1 | Ultralytics **YOLO11n** | objectdetectie (**geen LLM**) | in het Python-proces (PyTorch, GPU of CPU) |
| 2 | **llama3.2:3b** | LLM (taalmodel) | lokale **Ollama**-server op `localhost:11434` |

YOLO detecteert de objecten op de foto en levert een tekstlijst (labels, aantallen,
posities, zekerheid). Die lijst — en **niet** de foto zelf — gaat als prompt naar het
LLM, dat er een beschrijving of antwoord van maakt.

## Setup

Eenmalig:

```powershell
# 1. Virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Python-dependencies (alleen code; de modellen komen los)
pip install -r requirements.txt

# 3. Controleer of PyTorch de GPU ziet (optioneel maar aan te raden)
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
#   False?  -> CUDA-build installeren:
#   pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124

# 4. LLM ophalen via Ollama (Ollama moet geïnstalleerd zijn)
ollama pull llama3.2:3b
```

De YOLO-gewichten (`yolo11n.pt`, ~5 MB) worden bij de eerste run automatisch gedownload.

## Gebruiken

```powershell
# Ollama-server draaien (apart venster, of hij draait al als achtergronddienst)
ollama serve

# Beschrijving van een foto
python pipeline.py sample.jpg

# Met een eigen vraag
python pipeline.py sample.jpg --vraag "Wat voor plek is dit en is het druk?"

# Alles op CPU forceren (modellen om de beurt in het geheugen)
python pipeline.py sample.jpg --device cpu
```

Opties: `--model`, `--weights`, `--device`, `--conf`, `--out`. Zie `python pipeline.py -h`.

De pipeline print het tussenresultaat (YOLO-output = LLM-input) en slaat een
geannoteerde afbeelding op als `output_annotated.jpg`.

## Waarom deze combinatie

Een objectdetectiemodel ziet *wat* er in beeld staat maar kan er geen taal over
produceren. Een LLM kan redeneren en formuleren maar krijgt hier geen beeld. Door
YOLO's gestructureerde output als feiten aan het LLM te voeren, ontstaat een systeem
dat een foto in natuurlijke taal kan beschrijven en er vragen over kan beantwoorden —
zonder dat één model beide taken doet, en zonder cloud.
