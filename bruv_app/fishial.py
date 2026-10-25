"""Fishial's fish classifier, as a second opinion beside BioCLIP (modeltest.py).

Fishial (fishial.ai, the Wye Foundation's fish-identification repository) publishes a
TorchScript bundle: DinoV2 with a ViT pooling head trained with sub-center ArcFace, 866
species named scientifically, and three "natural centroids" per species. A crop is padded
to Fishial's wide input (434 x 154, a fish lying across) with Fishial's neutral grey, then
normalised as for ImageNet; its embedding is compared with every centroid, and a species
scores its best centroid ("natural_centroid", the default in Fishial's inference.py). The
code is MIT licensed; the weights' licence isn't stated.

Unlike BioCLIP, it can only answer from its own list, which is heavy on American and
European fishing species: 27 snappers and 16 groupers, but one unicornfish and no
redtoothed triggerfish. Its names are mapped to families by genus (FAMILY_BY_GENUS), so it
can be compared with BioCLIP on the families of a template; a species outside them maps to
no family and counts as wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .classify import l2_normalize

INPUT_HW = (154, 434)
NEUTRAL = (124, 116, 104)  # Fishial pads crops with this colour
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# Common family names, written as a template's groups are (compared after lowercasing and
# turning "_" and "/" into spaces).
_FAMILIES = {
    "grouper": "Cephalopholis Epinephelus Hyporthodus Mycteroperca Plectropomus Variola",
    "snapper": "Aprion Etelis Hoplopagrus Lutjanus Ocyurus Rhomboplites",
    "parrotfish": "Chlorurus Scarus Sparisoma",
    "jack trevally": "Alectis Atule Caranx Chloroscombrus Decapterus Elagatis Gnathanodon Lichia Megalaspis "
                     "Oligoplites Scomberoides Selar Selene Seriola Trachinotus Trachurus",
    "triggerfish": "Balistapus Balistes Balistoides Canthidermis Melichthys Rhinecanthus Xanthichthys",
    "filefish": "Aluterus Cantherhines Meuschenia Monacanthus",
    "butterflyfish": "Chaetodon",
    "unicornfish": "Naso",
    "surgeonfish": "Acanthurus Paracanthurus Zebrasoma",
    "angelfish": "Holacanthus Pomacanthus",
    "damselfish": "Abudefduf Chromis Hypsypops Pomacentrus Stegastes",
    "clownfish": "Amphiprion",
    "wrasse": "Achoerodus Bodianus Cheilinus Cheilio Clepticus Coris Ctenolabrus Halichoeres Hemigymnus "
              "Labroides Labrus Lachnolaimus Notolabrus Pseudolabrus Semicossyphus Symphodus Tautoga "
              "Tautogolabrus Thalassoma Xyrichtys",
    "goatfish": "Mulloidichthys Mullus Parupeneus Pseudupeneus",
    "barracuda": "Sphyraena",
    "tuna mackerel": "Acanthocybium Auxis Euthynnus Gymnosarda Katsuwonus Rastrelliger Sarda Scomber "
                     "Scomberomorus Thunnus",
    "moray eel": "Channomuraena Echidna Gymnothorax",
    "shark": "Alopias Brachaelurus Carcharhinus Carcharias Carcharodon Galeocerdo Galeorhinus "
             "Ginglymostoma Haploblepharus Heterodontus Isurus Mustelus Nebrius Negaprion Notorynchus "
             "Poroderma Prionace Rhincodon Rhizoprionodon Scyliorhinus Sphyrna Squalus Triaenodon Triakis",
    "ray": "Acroteriobatus Aetobatus Aetomylaeus Aptychotrema Dasyatis Gymnura Himantura Hypanus "
           "Myliobatis Platyrhinoidis Pseudobatos Pteroplatytrygon Raja Rhinoptera Taeniura "
           "Trygonoptera Trygonorrhina",
    "pufferfish": "Arothron Canthigaster Ephippion Lagocephalus Sphoeroides Torquigener",
    "porcupinefish": "Chilomycterus Diodon",
    "boxfish": "Acanthostracion",
    "goby": "Coryphopterus Gobius Neogobius Pomatoschistus Priolepis",
    "blenny": "Acanthemblemaria Chaenopsis Lipophrys",
    "hawkfish": "Cirrhitus",
    "scorpionfish lionfish": "Pterois Scorpaena",
    "lizardfish": "Synodus",
    "trumpetfish": "Aulostomus",
    "batfish spadefish": "Chaetodipterus Platax",
    "squirrelfish soldierfish": "Holocentrus Myripristis",
    "bigeye": "Priacanthus",
    "sandperch": "Parapercis",
    "tilefish": "Caulolatilus",
    "flathead": "Platycephalus",
    "flounder": "Citharichthys Etropus Hippoglossoides Hippoglossus Limanda Paralichthys Platichthys "
                "Pleuronectes Pseudopleuronectes Trinectes",
    "rabbitfish": "Siganus",
    "emperor": "Lethrinus",
    "sweetlips": "Diagramma",
    "sea chub": "Kyphosus",
    "remora": "Echeneis",
    "cobia": "Rachycentron",
    "mahi mahi": "Coryphaena",
    "black marlin": "Istiompax",
    "indo-pacific blue marlin": "Makaira",
}
FAMILY_BY_GENUS = {genus: family for family, genera in _FAMILIES.items() for genus in genera.split()}
# Species a template keeps apart from their family.
FAMILY_BY_SPECIES = {"Cheilinus undulatus": "napoleon wrasse"}


def family_name(species: str) -> str | None:
    """A Fishial species' common family name, or None outside the families above."""
    return FAMILY_BY_SPECIES.get(species) or FAMILY_BY_GENUS.get(species.split()[0])


def same_family(a: str, b: str) -> bool:
    """A family name against a template group's name ("jack_trevally", "jack trevally")."""
    norm = lambda s: " ".join(s.lower().replace("_", " ").replace("/", " ").split())  # noqa: E731
    return norm(a) == norm(b)


def prepare(crop: np.ndarray) -> np.ndarray:
    """A crop as Fishial's model takes it: padded to its wide shape with its grey, resized
    to 434 x 154, normalised; channels first."""
    from PIL import Image  # noqa: PLC0415

    h, w = crop.shape[:2]
    target = INPUT_HW[1] / INPUT_HW[0]
    if w / max(h, 1) > target:
        canvas_w, canvas_h = w, int(round(w / target))
    else:
        canvas_w, canvas_h = int(round(h * target)), h
    canvas = np.empty((max(canvas_h, 2), max(canvas_w, 2), 3), np.uint8)
    canvas[:] = NEUTRAL
    y, x = (canvas.shape[0] - h) // 2, (canvas.shape[1] - w) // 2
    canvas[y:y + h, x:x + w] = crop
    image = Image.fromarray(canvas).resize((INPUT_HW[1], INPUT_HW[0]), Image.BILINEAR)
    arr = (np.asarray(image, np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)


class FishialClassifier:
    """Fishial's TorchScript bundle (model.pt from classification_model_v0.10.2.zip)."""

    name = "fishial"

    def __init__(self, model_path: Path, device: str = "cpu"):
        import torch  # noqa: PLC0415

        self.torch = torch
        self.device = device
        self.model = torch.jit.load(str(model_path), map_location=device).eval()
        mapping = json.loads(bytes(self.model.class_mapping_json_bytes.cpu().tolist()).decode("utf-8"))
        self.labels = [mapping[str(i)]["label"] for i in range(len(mapping))]
        self.centroids = torch.nn.functional.normalize(self.model.natural_centroids.float(), dim=1).cpu().numpy()
        self.centroid_labels = self.model.natural_labels.long().cpu().numpy()

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(crops), 16):
                batch = torch.from_numpy(np.stack([prepare(c) for c in crops[i:i + 16]])).to(self.device)
                out.append(self.model(batch)[0].float().cpu().numpy())
        return l2_normalize(np.concatenate(out)) if out else np.zeros((0, self.centroids.shape[1]), np.float32)

    def class_scores(self, emb: np.ndarray) -> np.ndarray:
        """Each species' score for one embedding: its best centroid's cosine similarity."""
        sims = self.centroids @ emb
        out = np.full(len(self.labels), -1.0, np.float32)
        np.maximum.at(out, self.centroid_labels, sims)
        return out
