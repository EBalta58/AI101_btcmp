# Opdracht 101 - Foto naar tekst met twee modellen

Dit programma pakt een foto en geeft er een stukje tekst over terug.
Het gebruikt twee AI-modellen die achter elkaar werken. Alles draait op je
eigen laptop, dus geen internet en geen ChatGPT.

## De twee modellen

1. YOLO11n van Ultralytics. Dit is geen taalmodel. Het kijkt naar de foto
   en zegt welke dingen erop staan, bijvoorbeeld "4 personen" en "1 bus".
2. llama3.2:3b. Dit is het taalmodel (LLM). Het draait via Ollama. Het krijgt
   het lijstje van YOLO als tekst en maakt daar een normale zin van.

Het taalmodel ziet de foto zelf niet. Het krijgt alleen de woorden van YOLO.

## Wat je nodig hebt

- Python
- Ollama met het model llama3.2:3b

## Installeren

Doe dit een keer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull llama3.2:3b
```

Het YOLO-model (yolo11n.pt) wordt bij de eerste keer draaien vanzelf gedownload.

## Draaien

Zorg dat Ollama aan staat. Start het met `ollama serve` of open de Ollama app.

Foto beschrijven:

```powershell
python pipeline.py sample.jpg
```

Een vraag stellen over de foto:

```powershell
python pipeline.py sample.jpg --vraag "Wat voor plek is dit?"
```

Je eigen foto gebruiken:

```powershell
python pipeline.py mijnfoto.png
```

## Wat je ziet

Het programma laat drie dingen zien:

1. Wat YOLO gevonden heeft. Dit gaat door naar het taalmodel.
2. Het antwoord van het taalmodel.
3. Hoe lang elk model deed.

Er wordt ook een foto opgeslagen als output_annotated.jpg. Daar staan
vakjes op om de gevonden dingen.

## Extra opties

- `--vraag` een vraag voor het taalmodel
- `--model` een ander Ollama-model kiezen
- `--conf` hoe zeker YOLO moet zijn. Standaard 0.35. Lager is meer dingen.
- `--device cpu` YOLO op de processor draaien
- `--out` andere bestandsnaam voor de opgeslagen foto

## Waarom deze twee samen

YOLO ziet wel wat er op de foto staat, maar kan er geen zin van maken.
Het taalmodel kan wel zinnen maken, maar kijkt hier niet naar de foto.
Samen kunnen ze een foto in gewone taal uitleggen.
