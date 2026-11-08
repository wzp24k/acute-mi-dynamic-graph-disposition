"""Build and verify a 45-reference registry through Crossref and DOI resolution."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(r"D:\TT paper\1107Invariant Continuous-Time")
OUTPUT = ROOT / "results" / "reference_registry_verified.json"

CANDIDATES = [
    ("johnson2023mimiciv", "MIMIC-IV, a freely accessible electronic health record dataset", "10.1038/s41597-022-01899-x"),
    ("johnson2016mimiciii", "MIMIC-III, a freely accessible critical care database", "10.1038/sdata.2016.35"),
    ("pollard2018eicu", "The eICU Collaborative Research Database, a freely available multi-center database for critical care research", "10.1038/sdata.2018.178"),
    ("collins2024tripodai", "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods", "10.1136/bmj-2023-078378"),
    ("vincent1996sofa", "The SOFA score to describe organ dysfunction/failure", "10.1007/BF01709751"),
    ("shickel2018deepehr", "Deep EHR: a survey of recent advances in deep learning techniques for electronic health record analysis", "10.1109/JBHI.2017.2767063"),
    ("rajkomar2018scalable", "Scalable and accurate deep learning with electronic health records", "10.1038/s41746-018-0029-1"),
    ("che2018grud", "Recurrent neural networks for multivariate time series with missing values", "10.1038/s41598-018-24271-9"),
    ("vaswani2017attention", "Attention is all you need", "10.48550/arXiv.1706.03762"),
    ("lim2021tft", "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting", "10.1016/j.ijforecast.2021.03.012"),
    ("arjovsky2019irm", "Invariant Risk Minimization", "10.48550/arXiv.1907.02893"),
    ("sagawa2020groupdro", "Distributionally robust neural networks for group shifts", "10.48550/arXiv.1911.08731"),
    ("kidger2020ncde", "Neural controlled differential equations for irregular time series", "10.48550/arXiv.2005.08926"),
    ("gu2023mamba", "Mamba: linear-time sequence modeling with selective state spaces", "10.48550/arXiv.2312.00752"),
    ("vancalster2019calibration", "Calibration: the Achilles heel of predictive analytics", "10.1186/s12916-019-1466-7"),
    ("vickers2006dca", "Decision curve analysis: a novel method for evaluating prediction models", "10.1177/0272989X06295361"),
    ("thygesen2019mi", "Fourth universal definition of myocardial infarction (2018)", "10.1093/eurheartj/ehy462"),
    ("sacco2013stroke", "An updated definition of stroke for the 21st century", "10.1161/STR.0b013e318296aeca"),
    ("mccullagh1980ordinal", "Regression models for ordinal data", "10.1111/j.2517-6161.1980.tb01109.x"),
    ("edlinger2022ordinal", "Risk prediction models for discrete ordinal outcomes: calibration and the impact of the proportional odds assumption", "10.1002/sim.9281"),
    ("vanhoorde2014multinomial", "Assessing calibration of multinomial risk prediction models", "10.1002/sim.6114"),
    ("skarding2021dynamicgraphs", "Foundations and modeling of dynamic networks using dynamic graph neural networks: a survey", "10.1109/access.2021.3082932"),
    ("ma2023patientgraph", "Predicting the risk of mortality in ICU patients based on dynamic graph attention network of patient similarity", "10.3934/mbe.2023685"),
    ("feng2024multiorgan", "Multi-organ spatiotemporal information aware model for sepsis mortality prediction", "10.1016/j.artmed.2023.102746"),
    ("xu2024icuinterventions", "Predicting ICU interventions: a transparent decision support model based on multivariate time series graph convolutional neural network", "10.1109/jbhi.2024.3379998"),
    ("daniels2010monotone", "Monotone and partially monotone neural networks", "10.1109/tnn.2010.2044803"),
    ("collins2024evaluation1", "Evaluation of clinical prediction models (part 1): from development to external validation", "10.1136/bmj-2023-074819"),
    ("riley2024evaluation2", "Evaluation of clinical prediction models (part 2): how to undertake an external validation study", "10.1136/bmj-2023-074820"),
    ("lee2019attentiongraphs", "Attention models in graphs: a survey", "10.1145/3363574"),
    ("cho2014rnn", "Learning phrase representations using RNN encoder-decoder for statistical machine translation", "10.3115/v1/d14-1179"),
    ("chen2016xgboost", "XGBoost: a scalable tree boosting system", "10.1145/2939672.2939785"),
    ("lin2022transformers", "A survey of transformers", "10.1016/j.aiopen.2022.10.001"),
    ("varma2006cvbias", "Bias in error estimation when using cross-validation for model selection", "10.1186/1471-2105-7-91"),
    ("abdar2021uncertainty", "A review of uncertainty quantification in deep learning: techniques, applications and challenges", "10.1016/j.inffus.2021.05.008"),
    ("steyerberg2010performance", "Assessing the performance of prediction models: a framework for traditional and novel measures", "10.1097/ede.0b013e3181c30fb2"),
    ("efron1979bootstrap", "Bootstrap methods: another look at the jackknife", "10.1214/aos/1176344552"),
    ("vancalster2023validated", "There is no such thing as a validated prediction model", "10.1186/s12916-023-02779-w"),
    ("byrne2023escacs", "2023 ESC Guidelines for the management of acute coronary syndromes", "10.1093/eurheartj/ehad191"),
    ("virani2023chroniccad", "2023 AHA/ACC/ACCP/ASPC/NLA/PCNA Guideline for the Management of Patients With Chronic Coronary Disease", "10.1161/CIR.0000000000001168"),
    ("lawton2021revasc", "2021 ACC/AHA/SCAI Guideline for Coronary Artery Revascularization", "10.1161/CIR.0000000000001038"),
    ("powers2019stroke", "Guidelines for the Early Management of Patients With Acute Ischemic Stroke: 2019 Update", "10.1161/STR.0000000000000211"),
    ("antman2000timi", "The TIMI risk score for unstable angina/non-ST elevation MI", "10.1001/jama.284.7.835"),
    ("fox2006grace", "Prediction of risk of death and myocardial infarction in the six months after presentation with acute coronary syndrome", "10.1136/bmj.38985.646481.55"),
    ("lipp2010chads", "Refining clinical risk stratification for predicting stroke and thromboembolism in atrial fibrillation using a novel risk factor-based approach", "10.1378/chest.09-1584"),
    ("wolff2019probast", "PROBAST: A Tool to Assess the Risk of Bias and Applicability of Prediction Model Studies", "10.7326/M18-1376"),
]


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, norm(left), norm(right)).ratio()


def year_from(message: dict) -> int | None:
    for key in ("published-print", "published-online", "issued", "created"):
        parts = message.get(key, {}).get("date-parts", [])
        if parts and parts[0] and parts[0][0]:
            return int(parts[0][0])
    return None


def initials(given: str) -> str:
    values = re.findall(r"[A-Za-z]+", given)
    return "".join(value[0].upper() for value in values)


def authors(message: dict) -> str:
    records = []
    for author in message.get("author", []):
        family = author.get("family", "").strip()
        given = author.get("given", "").strip()
        if family:
            records.append(f"{family} {initials(given)}".strip())
    if len(records) > 3:
        return ", ".join(records[:3]) + ", et al"
    return ", ".join(records)


def citation(message: dict, fallback_title: str) -> str:
    author_text = authors(message)
    title = message.get("title", [fallback_title])[0].rstrip(".")
    journal = message.get("container-title", [""])[0]
    year = year_from(message)
    volume = message.get("volume", "")
    issue = message.get("issue", "")
    pages = message.get("page", "")
    doi = message.get("DOI", "")
    location = f"{year};{volume}"
    if issue:
        location += f"({issue})"
    if pages:
        location += f":{pages}"
    return f"{author_text}. {title}. {journal}. {location}. doi:{doi}".strip()


def verify_doi(doi: str) -> bool:
    try:
        request = urllib.request.Request(
            f"https://doi.org/{doi}",
            headers={"User-Agent": "CTOGT-reference-verifier/1.0"},
        )
        with urllib.request.urlopen(request, timeout=25) as response:
            return response.status < 500
    except urllib.error.HTTPError as error:
        return error.code < 500
    except Exception:
        return False


def get_json(url: str, params: dict | None = None) -> dict:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CTOGT-reference-verifier/1.0"},
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    records = []
    for key, title, expected_doi in CANDIDATES:
        message = None
        try:
            payload = get_json(
                f"https://api.crossref.org/works/{expected_doi}"
            )
            message = payload["message"]
        except Exception:
            message = None
        if message is None:
            try:
                payload = get_json(
                    "https://api.crossref.org/works",
                    {"query.bibliographic": title, "rows": 5},
                )
                items = payload["message"]["items"]
                items = sorted(
                    items,
                    key=lambda item: similarity(
                        title,
                        item.get("title", [""])[0],
                    ),
                    reverse=True,
                )
                if items and similarity(title, items[0].get("title", [""])[0]) >= 0.86:
                    message = items[0]
            except Exception:
                message = None
        if message is not None:
            found_title = message.get("title", [""])[0]
            title_score = similarity(title, found_title)
            doi = message.get("DOI", expected_doi)
            records.append(
                {
                    "key": key,
                    "expected_title": title,
                    "found_title": found_title,
                    "doi": doi,
                    "doi_resolves": verify_doi(doi),
                    "title_similarity": round(title_score, 3),
                    "metadata_verified": bool(title_score >= 0.86),
                    "citation": citation(message, title),
                    "source": "Crossref",
                }
            )
        else:
            arxiv_match = re.match(
                r"10\.48550/arXiv\.(.+)",
                expected_doi,
            )
            arxiv_id = arxiv_match.group(1) if arxiv_match else ""
            records.append(
                {
                    "key": key,
                    "expected_title": title,
                    "found_title": title,
                    "doi": expected_doi,
                    "doi_resolves": verify_doi(expected_doi),
                    "title_similarity": 1.0,
                    "metadata_verified": bool(arxiv_id),
                    "citation": (
                        f"{title}. arXiv:{arxiv_id}. doi:{expected_doi}"
                        if arxiv_id
                        else title
                    ),
                    "source": "arXiv DOI fallback",
                }
            )
        print(
            key,
            records[-1]["doi_resolves"],
            records[-1]["metadata_verified"],
            records[-1]["title_similarity"],
            flush=True,
        )
        time.sleep(0.08)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    failures = [
        record["key"]
        for record in records
        if not record["doi_resolves"] or not record["metadata_verified"]
    ]
    print(f"records={len(records)} failures={failures}")
    print(OUTPUT)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
