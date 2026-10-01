"""Cause probabili e controlli consigliati per ogni tipo di guasto.

analyze_description() è una prima versione a parole chiave per l'analisi di
descrizioni testuali; non è ancora collegata al sito.
"""
from __future__ import annotations

import re
import unicodedata

KNOWLEDGE_BASE = {
    "NORMALE": {
        "title": "Funzionamento normale",
        "causes": [],
        "actions": ["Nessuna azione: proseguire con la manutenzione programmata."],
    },
    "SOTTOPRESTAZIONE": {
        "title": "Sottoprestazione rispetto agli inverter vicini",
        "causes": [
            "sporcamento (polvere) dei moduli collegati all'inverter",
            "ombreggiamento parziale (vegetazione, strutture)",
            "stringa aperta o fusibile di stringa interrotto",
            "degrado / hot-spot / PID sui moduli",
            "derating termico dell'inverter (ventilazione insufficiente)",
        ],
        "actions": [
            "Confrontare le correnti di stringa dell'inverter: una stringa a 0 A indica fusibile o connettore guasto.",
            "Ispezione visiva e, se il calo è graduale su più giorni, pianificare il lavaggio dei moduli.",
            "Termografia IR dei moduli per individuare hot-spot o diodi di bypass guasti.",
            "Se il calo avviene nelle ore più calde: controllare ventole e filtri dell'inverter.",
        ],
    },
    "INVERTER_FERMO": {
        "title": "Inverter fermo / scatto (trip)",
        "causes": [
            "protezione per guasto di isolamento (dispersione verso terra)",
            "sovratensione o sovratemperatura dell'inverter",
            "intervento di protezioni lato rete (tensione/frequenza fuori soglia)",
            "guasto interno (IGBT, scheda di controllo)",
        ],
        "actions": [
            "Leggere il codice di allarme dell'inverter dal portale/SCADA.",
            "Misurare la resistenza di isolamento delle stringhe (Riso) prima del riarmo.",
            "Verificare interruttori/sezionatori lato DC e AC e lo stato della rete.",
            "Se gli scatti sono ricorrenti alla stessa ora: controllare temperatura e tensione di rete.",
        ],
    },
    "PERDITA_COMUNICAZIONE": {
        "title": "Perdita di comunicazione / dati",
        "causes": [
            "datalogger o router in blocco",
            "cavo RS485/Modbus o switch di rete guasto",
            "alimentazione ausiliaria assente",
        ],
        "actions": [
            "Verificare alimentazione e stato del datalogger; riavvio da remoto se possibile.",
            "Controllare la catena di comunicazione (RS485/Ethernet) dell'inverter interessato.",
            "Attenzione: l'inverter potrebbe produrre regolarmente; confrontare con il contatore fiscale.",
        ],
    },
    "SENSORE_METEO": {
        "title": "Sensore meteo sospetto",
        "causes": ["piranometro sporco, disallineato o guasto", "sensore di temperatura staccato dal modulo"],
        "actions": ["Pulire e verificare l'allineamento del piranometro; confrontare con dati satellitari."],
    },
    "STRESS_TERMICO": {
        "title": "Temperatura dei moduli anomala",
        "causes": ["scarsa ventilazione dei moduli", "sensore di temperatura mal posizionato"],
        "actions": ["Verificare il fissaggio del sensore; valutare impatto sul rendimento (~-0,4 %/°C)."],
    },
}

# parole chiave -> (categoria, peso). Il testo viene normalizzato senza accenti.
KEYWORDS = {
    "INVERTER_FERMO": [
        (r"\b(spento|fermo|ferma|bloccat\w*|trip|scatt\w*)\b", 2),
        (r"non produce|produzione (a )?zero|potenza (a )?zero|0 ?kw", 3),
        (r"isolamento|dispersion\w*|guasto a terra|ground fault|riso", 3),
        (r"allarme|errore|codice", 1),
        (r"sovratension\w*|rete|frequenza", 1),
    ],
    "SOTTOPRESTAZIONE": [
        (r"produce meno|calo|cala|ridott\w*|bass\w* (resa|produzione)|sottoprestazion\w*|rende meno", 3),
        (r"sporc\w*|polvere|sabbia|guano|lavaggio", 2),
        (r"ombr\w*|vegetazion\w*|alberi", 2),
        (r"stringa|fusibil\w*|connettor\w*|mc4", 2),
        (r"hot.?spot|degrad\w*|pid|microcrack|diodo", 2),
    ],
    "PERDITA_COMUNICAZIONE": [
        (r"nessun dato|dati mancant\w*|mancano i dati|buchi|comunicazion\w*|offline|non comunica", 3),
        (r"datalogger|modbus|rs.?485|router|scada non", 2),
    ],
    "STRESS_TERMICO": [
        (r"surriscald\w*|temperatura (alta|elevata)|troppo cald\w*|ventol\w*|derating", 2),
    ],
    "SENSORE_METEO": [
        (r"sensore|piranometro|irraggiamento (strano|anomalo|sbagliato)", 2),
    ],
}


def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in t if not unicodedata.combining(ch))


def analyze_description(text: str) -> list[dict]:
    """Restituisce le ipotesi ordinate per punteggio con cause e azioni."""
    t = _normalize(text)
    results = []
    for cat, patterns in KEYWORDS.items():
        score, hits = 0, []
        for pat, w in patterns:
            m = re.search(pat, t)
            if m:
                score += w
                hits.append(m.group(0))
        if score:
            results.append({"category": cat, "score": score, "matched": hits, **KNOWLEDGE_BASE[cat]})
    results.sort(key=lambda r: -r["score"])
    total = sum(r["score"] for r in results) or 1
    for r in results:
        r["confidence"] = round(r["score"] / total, 2)
    # indizi temporali utili per la diagnosi
    if results and re.search(r"(ore|orario) (centrali|piu calde)|mezzogiorno|pomeriggio", t):
        results[0]["note"] = ("Il problema compare nelle ore più calde: considerare derating termico "
                              "o sovratensione di rete.")
    return results


def recommend(category: str) -> dict:
    return KNOWLEDGE_BASE.get(category, KNOWLEDGE_BASE["NORMALE"])
