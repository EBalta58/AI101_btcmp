#!/usr/bin/env python3
"""
Lokale twee-modellen pipeline: Foto -> YOLO (objectdetectie) -> LLM (tekst).

Model 1 (detectie model) : Ultralytics YOLO11n     -> detecteert objecten in de afbeelding
Model 2 (LLM)      : llama3.2:3b via Ollama  -> beschrijft / redeneert over de scene

De output van model 1 (een tekstlijst van detecties) is de ENIGE input voor model 2.
Het LLM krijgt de foto zelf niet te zien.

Alles draait lokaal: YOLO in dit proces, het LLM via de lokale Ollama-server
(http://localhost:11434). Geen cloud-API's.
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import requests
from ultralytics import YOLO

OLLAMA_URL = "http://localhost:11434/api/generate"


def zone(xc, yc, w, h):
    """Zet een bounding-box-middelpunt om in een grove positie-aanduiding."""
    h_zone = "links" if xc < w / 3 else "rechts" if xc > 2 * w / 3 else "midden"
    v_zone = "boven" if yc < h / 3 else "onder" if yc > 2 * h / 3 else "midden"
    if h_zone == "midden" and v_zone == "midden":
        return "midden"
    if h_zone == "midden":
        return v_zone
    if v_zone == "midden":
        return h_zone
    return f"{v_zone}-{h_zone}"


def run_yolo(image_path, weights, device, conf):
    """Draai YOLO en geef (detecties, tijd, geannoteerd beeld, (breedte, hoogte))."""
    model = YOLO(weights)
    t0 = time.perf_counter()
    results = model.predict(source=str(image_path), conf=conf, device=device, verbose=False)
    dt = time.perf_counter() - t0

    res = results[0]
    h, w = res.orig_shape
    names = res.names

    detections = []
    for box in res.boxes:
        label = names[int(box.cls)]
        confidence = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
        detections.append(
            {"label": label, "conf": confidence, "positie": zone(xc, yc, w, h)}
        )

    annotated = res.plot()  # numpy BGR-array met boxes erop
    return detections, dt, annotated, (w, h)


def summarize_detections(detections):
    """Bouw het tekstblok dat als input naar het LLM gaat."""
    if not detections:
        return "Geen objecten gedetecteerd."

    counts = Counter(d["label"] for d in detections)
    lines = []
    for label, n in counts.most_common():
        posns = [d["positie"] for d in detections if d["label"] == label]
        confs = [d["conf"] for d in detections if d["label"] == label]
        conf_str = ", ".join(f"{c:.0%}" for c in confs)
        lines.append(f"- {n}x {label} (positie: {', '.join(posns)}; zekerheid: {conf_str})")
    return "\n".join(lines)


def ask_llm(model, detection_text, question, timeout=180):
    """Stuur het detectie-tekstblok naar het lokale LLM via Ollama."""
    system = (
        "Je bent een assistent die een foto beschrijft die je NIET kunt zien. "
        "Je krijgt uitsluitend de ruwe uitvoer van een objectdetectiemodel (YOLO). "
        "Baseer je antwoord alleen op die detecties. Verzin geen objecten die er niet in staan. "
        "Als de detectielijst leeg is, zeg dat eerlijk. "
        "Schrijf vloeiend en natuurlijk in het Nederlands, maximaal 4 zinnen. "
        "Som geen zekerheidspercentages op en geef geen uitleg over het detectiemodel; "
        "gebruik de detecties gewoon als feiten over de scene."
    )
    if question:
        user = f"Objectdetectie van de foto:\n{detection_text}\n\nVraag: {question}"
    else:
        user = (
            f"Objectdetectie van de foto:\n{detection_text}\n\n"
            "Schrijf een korte, natuurlijke beschrijving (2-4 zinnen) van wat er waarschijnlijk "
            "op de foto te zien is en wat voor situatie of plek het lijkt."
        )

    payload = {
        "model": model,
        "prompt": user,
        "system": system,
        "stream": False,
        "keep_alive": 0,  # LLM direct uit het geheugen na het antwoord -> modellen om de beurt
        "options": {"temperature": 0.4},
    }

    t0 = time.perf_counter()
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    dt = time.perf_counter() - t0
    return r.json()["response"].strip(), dt


def main():
    p = argparse.ArgumentParser(
        description="Foto -> YOLO -> LLM pipeline (volledig lokaal)."
    )
    p.add_argument("image", type=Path, help="pad naar de invoerafbeelding")
    p.add_argument("--vraag", default=None, help="optionele vraag voor het LLM over de foto")
    p.add_argument("--model", default="llama3.2:3b", help="Ollama LLM-model (default: llama3.2:3b)")
    p.add_argument("--weights", default="yolo11n.pt", help="YOLO-gewichten (default: yolo11n.pt)")
    p.add_argument("--device", default=None, help="'cpu' of '0' voor GPU. Default: automatisch")
    p.add_argument("--conf", type=float, default=0.35, help="YOLO confidence-drempel (default: 0.35)")
    p.add_argument("--out", type=Path, default=Path("output_annotated.jpg"),
                   help="pad voor de geannoteerde afbeelding")
    args = p.parse_args()

    if not args.image.exists():
        sys.exit(f"Afbeelding niet gevonden: {args.image}")

    # --- Model 1: YOLO ---------------------------------------------------------
    print(f"[1/3] YOLO-objectdetectie op {args.image} ...")
    detections, yolo_dt, annotated, (w, h) = run_yolo(
        args.image, args.weights, args.device, args.conf
    )
    detection_text = summarize_detections(detections)
    print(f"      klaar in {yolo_dt:.2f}s  ({len(detections)} objecten, beeld {w}x{h})\n")

    print("--- Output model 1 (YOLO)  ==>  input model 2 (LLM) ---")
    print(detection_text)
    print("------------------------------------------------------\n")

    import cv2
    cv2.imwrite(str(args.out), annotated)
    print(f"[2/3] Geannoteerde afbeelding opgeslagen: {args.out}\n")

    # --- Model 2: LLM ---------------------------------------------------------
    print(f"[3/3] LLM ({args.model}) redeneert over de detecties ...")
    try:
        answer, llm_dt = ask_llm(args.model, detection_text, args.vraag)
    except requests.exceptions.ConnectionError:
        sys.exit("Geen verbinding met Ollama op localhost:11434. Draait 'ollama serve'?")
    print(f"      klaar in {llm_dt:.2f}s\n")

    print("=== ANTWOORD ===")
    print(answer)
    print()
    print(f"(YOLO {yolo_dt:.2f}s + LLM {llm_dt:.2f}s = {yolo_dt + llm_dt:.2f}s totaal)")


if __name__ == "__main__":
    main()
