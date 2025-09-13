# =========================
# PDPA Semantic RAG (Embedding-First, MMR, Optional CE Rerank)
# =========================
# - Bi-encoder embeddings with e5-style prompts ("query:" / "passage:")
# - Optional CrossEncoder rerank with sigmoid normalization
# - Maximal Marginal Relevance (MMR) to improve diversity
# - Compact PDPA + DNC knowledge base tailored for consent/marketing workflows
# - Single entrypoint: analyze_legal_scenario(scenario_text)
#
# Requires:
#   pip install sentence-transformers numpy
#
# Optional (for best reranking):
#   pip install sentence-transformers  (CrossEncoder is in the same package)
#
# Note: If CrossEncoder/model download isn't available in your environment,
#       the code will gracefully fall back to bi-encoder scores only.

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception:
    SentenceTransformer = None
    CrossEncoder = None


# -------------------------
# Data structures
# -------------------------

@dataclass
class LawCard:
    id: str
    title: str
    body: str
    cites: List[str]
    tags: List[str]
    examples: List[str]
    weight: float = 1.0
    # NEW: fine-grained anchors for retrieval/explanations
    subsections: Optional[Dict[str, str]] = None


# -------------------------
# Knowledge base (compact, marketing-focused but still general)
# -------------------------

def _build_pdpa_kb() -> List[LawCard]:
    kb = [
        # -------- Consent Core (kept) --------
        LawCard(
            id="PDPA-13",
            title="Section 13 — Consent required",
            body=("No collection, use or disclosure of personal data unless consent is given (or deemed), "
                  "or processing without consent is required/authorised under PDPA or written law."),
            cites=["s14", "s15", "s15A", "s17", "First Schedule", "Second Schedule"],
            tags=["consent", "baseline", "collection", "use", "disclosure"],
            examples=["Promotional emails generally require prior consent unless an exception applies."],
            weight=1.0,
            subsections={
                "13(a)": "Individual gives (or is deemed to have given) consent.",
                "13(b)": "Processing without consent required/authorised under PDPA/other written law."
            }
        ),
        LawCard(
            id="PDPA-14",
            title="Section 14 — Provision & validity of consent",
            body=("Consent is invalid unless s20 info is provided and consent is for that purpose. "
                  "No excessive 'condition of service' consent or consent via false/misleading/deceptive practices."),
            cites=["s20", "s13", "s15"],
            tags=["consent", "validity", "notice", "bundling"],
            examples=["Provide clear purposes before seeking consent; avoid bundled consent."],
            weight=0.95,
            subsections={
                "14(1)(a)": "Provide information required under s20.",
                "14(1)(b)": "Consent must be for that purpose in accordance with the Act.",
                "14(2)(a)": "No requiring consent beyond what’s reasonable to provide the service.",
                "14(2)(b)": "No false/misleading info or deceptive practices to obtain consent.",
                "14(3)": "Consent in circumstances of s14(2) is not valid.",
                "14(4)": "Consent may be given by a person validly acting on the individual’s behalf."
            }
        ),
        LawCard(
            id="PDPA-15",
            title="Section 15 — Deemed consent",
            body=("Deemed consent: voluntary provision for a purpose; plus contract-related relay/performance scenarios, "
                  "subject to contractual limits and transition notes."),
            cites=["s14", "s15A", "s17"],
            tags=["deemed consent", "contract", "third-party"],
            examples=["Pre-contract sharing reasonably necessary to conclude the contract."],
            weight=0.9,
            subsections={
                "15(1)": "Voluntary provision + reasonableness.",
                "15(2)": "Downstream organisation deemed for same purpose.",
                "15(3)-(5)": "Pre-contract relay chain (A→B→others) where reasonably necessary. [40/2020]",
                "15(6)-(8)": "Contract performance/interest relay chain (A↔B) where reasonably necessary. [40/2020]",
                "15(9)": "Contract can specify/restrict what may be disclosed/purposes."
            }
        ),
        LawCard(
            id="PDPA-15A",
            title="Section 15A — Deemed consent by notification",
            body=("Allowed with prior adverse-effects assessment, notice (purpose + how to opt-out), reasonable opt-out window; "
                  "excludes prescribed purposes."),
            cites=["s15"],
            tags=["deemed consent", "notification", "opt-out", "assessment"],
            examples=["Advance notice of new analytics purpose with simple opt-out before start."],
            weight=0.85,
            subsections={
                "15A(1)": "Applies on/after 1 Feb 2021. [40/2020]",
                "15A(2)": "Deemed consent if requirements are met and no objection within window.",
                "15A(3)": "Does not apply to prescribed purposes.",
                "15A(4)": "Assessment + bring info to attention (intent, purpose, period & manner to object).",
                "15A(5)": "Assess adverse effects and implement measures; further prescribed requirements."
            }
        ),
        LawCard(
            id="PDPA-16",
            title="Section 16 — Withdrawal of consent",
            body=("Individuals may withdraw consent with reasonable notice; organisation must inform likely consequences and cease "
                  "processing unless required/authorised by law."),
            cites=["s13", "s25"],
            tags=["withdrawal", "rights", "marketing opt-out"],
            examples=["Stop marketing after unsubscribe; retain only if legally necessary."],
            weight=1.0,
            subsections={
                "16(1)": "Right to withdraw consent with reasonable notice.",
                "16(2)": "Inform likely consequences upon receipt of notice.",
                "16(3)": "Must not prohibit withdrawal (legal consequences may still follow).",
                "16(4)": "Cease (and cause intermediaries/agents to cease) unless another basis applies."
            }
        ),
        LawCard(
            id="PDPA-17",
            title="Section 17 — Processing without consent (Schedules)",
            body=("When collection/use/disclosure without consent is permitted under First/Second Schedules; onward use/disclosure "
                  "must be consistent with original permitted purpose."),
            cites=["First Schedule", "Second Schedule", "s13"],
            tags=["exceptions", "without consent", "lawful basis"],
            examples=["Emergencies; investigations as per schedules."],
            weight=0.95,
            subsections={
                "17(1)(a)": "Collect without consent per First Sch. or Second Sch. Part 1.",
                "17(1)(b)": "Use without consent per First Sch. or Second Sch. Part 2.",
                "17(1)(c)": "Disclose without consent per First Sch. or Second Sch. Part 3.",
                "17(2)": "Consistency of secondary purposes for received/collected data."
            }
        ),

        # -------- Governance: Purpose & Notice --------
        LawCard(
            id="PDPA-18",
            title="Section 18 — Limitation of purpose and extent",
            body=("Process only for purposes a reasonable person considers appropriate in the circumstances, and (where applicable) "
                  "those informed under s20."),
            cites=["s20"],
            tags=["purpose", "reasonableness", "proportionality"],
            examples=["Collect only fields proportionate to stated purpose."],
            weight=0.9,
            subsections={
                "18(a)": "Reasonable person appropriateness.",
                "18(b)": "Individual has been informed under s20 (if applicable)."
            }
        ),
        LawCard(
            id="PDPA-19",
            title="Section 19 — Personal data collected before 2 July 2014",
            body=("Legacy data may be used for the purposes it was collected for unless consent is withdrawn or the individual "
                  "has otherwise indicated no consent."),
            cites=["s16"],
            tags=["legacy data", "grandfathering"],
            examples=["Continue legacy operational uses unless the person opted out."],
            weight=0.8,
            subsections={
                "19(a)": "Use allowed unless consent withdrawn under s16.",
                "19(b)": "Use barred if individual indicated no consent (before/on/after 2 Jul 2014)."
            }
        ),
        LawCard(
            id="PDPA-20",
            title="Section 20 — Notification of purposes",
            body=("Inform purposes on/before collection; inform of any new purpose before use/disclosure; provide a contact person on "
                  "request; provide purpose to source-organisation where collecting from it without consent; special employment notices."),
            cites=["s14", "s18", "s15", "s15A", "s17"],
            tags=["notice", "timing", "contact point", "employment"],
            examples=["Update notice before repurposing data for marketing."],
            weight=0.9,
            subsections={
                "20(1)(a)": "Inform purposes on/before collection.",
                "20(1)(b)": "Inform any other purpose before use/disclosure for that purpose.",
                "20(1)(c)": "Provide business contact of a person who can answer questions.",
                "20(2)": "If collecting from another org without consent, give sufficient purpose info to that org.",
                "20(3)": "s20(1) doesn’t apply if s15/s15A deemed consent or s17 applies.",
                "20(4)-(5)": "Despite (3), must inform purposes & contact info for employment entry/management contexts."
            }
        ),

        # -------- Rights: Access, Correction, Preservation --------
        LawCard(
            id="PDPA-21",
            title="Section 21 — Access to personal data",
            body=("Provide personal data in possession/control and information on ways it has been or may have been used/disclosed "
                  "within 1 year, subject to Fifth Schedule and harm/national-interest limits; redaction where possible; rejection "
                  "or exclusion notifications within prescribed time."),
            cites=["Fifth Schedule"],
            tags=["access", "DSAR", "one-year record"],
            examples=["Provide account data and last-year disclosure log, redacting where required."],
            weight=0.85,
            subsections={
                "21(1)": "Access: data in possession/control + last-year use/disclosure info.",
                "21(2)": "No need to provide for matters in Fifth Schedule.",
                "21(3)": "Must not provide where safety/health/another’s PD, informant identity, or national interest at risk.",
                "21(3A)": "Carve-out: user activity/provided data of requester not blocked by (3)(c),(d).",
                "21(4)": "Do not inform of law-enforcement disclosure where disclosure was under law without consent.",
                "21(5)": "If feasible to omit excluded parts, provide the rest.",
                "21(6)": "If rejecting (under (2)/(3)), notify within prescribed time/requirements.",
                "21(7)": "If providing under (5), notify of any exclusions applied."
            }
        ),
        LawCard(
            id="PDPA-22",
            title="Section 22 — Correction of personal data",
            body=("Correct errors/omissions as soon as practicable unless reasonable grounds not to; send corrections to other orgs that "
                  "received data in the last year (unless not needed); annotate if no correction; opinions need not be altered; Sixth "
                  "Schedule carve-outs."),
            cites=["Sixth Schedule", "s21"],
            tags=["rectification", "downstream notice"],
            examples=["Fix wrong address and notify partners who received it in the last year."],
            weight=0.8,
            subsections={
                "22(1)": "Right to request correction.",
                "22(2)": "Correct promptly; send corrected data to every org that received it in last year unless not needed.",
                "22(3)": "With consent, may send only to specified orgs (not credit bureau).",
                "22(4)": "Recipient orgs must correct unless reasonable grounds not to.",
                "22(5)": "If no correction made, annotate requested correction.",
                "22(6)": "No duty to alter opinions (incl. professional/expert opinions).",
                "22(7)": "Need not comply for matters in Sixth Schedule."
            }
        ),
        LawCard(
            id="PDPA-22A",
            title="Section 22A — Preservation of copies of personal data",
            body=("If an access request under s21(1)(a) is refused, preserve a complete and accurate copy for at least the prescribed "
                  "period."),
            cites=["s21"],
            tags=["preservation", "records"],
            examples=["Keep a full copy after rejecting an access request, for the prescribed period."],
            weight=0.85,
            subsections={
                "22A(1)": "Preserve a copy for ≥ prescribed period after refusing s21(1)(a) access.",
                "22A(2)": "Copy must be complete and accurate."
            }
        ),

        # -------- Data Quality, Security, Retention --------
        LawCard(
            id="PDPA-23",
            title="Section 23 — Accuracy of personal data",
            body=("Make reasonable effort to ensure accuracy/completeness if data will affect a decision about the individual or be "
                  "disclosed to another organisation."),
            cites=["s22"],
            tags=["accuracy", "pre-disclosure check"],
            examples=["Verify key identifiers before onboarding decision."],
            weight=0.85,
            subsections=None
        ),
        LawCard(
            id="PDPA-24",
            title="Section 24 — Protection of personal data",
            body=("Make reasonable security arrangements to prevent unauthorised access/collection/use/disclosure/copying/modification/"
                  "disposal and loss of storage media/devices containing personal data."),
            cites=["26A–26E"],
            tags=["security", "TOMs", "safeguards"],
            examples=["Access controls, encryption in transit/at rest, secure disposal."],
            weight=0.9,
            subsections=None
        ),
        LawCard(
            id="PDPA-25",
            title="Section 25 — Retention limitation",
            body=("Cease retention or de-identify once the purpose is no longer served and retention is no longer necessary for legal or "
                  "business purposes."),
            cites=["s16"],
            tags=["retention", "de-identification", "erasure"],
            examples=["Purge expired marketing lists not needed for compliance."],
            weight=0.85,
            subsections={
                "25(a)": "Purpose no longer served by retention.",
                "25(b)": "Retention no longer necessary for legal/business purposes."
            }
        ),

        # -------- Transfers --------
        LawCard(
            id="PDPA-26",
            title="Section 26 — Transfer outside Singapore",
            body=("No overseas transfer unless prescribed requirements ensure protection comparable to PDPA; PDPC may exempt subject to "
                  "conditions and vary/revoke conditions."),
            cites=["Transfer Regulations"],
            tags=["cross-border", "comparable protection", "BCRs", "contracts"],
            examples=["Use contractual clauses/BCRs to cover vendor in another country."],
            weight=0.9,
            subsections={
                "26(1)": "Comparable protection requirement for transfers.",
                "26(2)": "Commission exemption power.",
                "26(3)": "Exemption conditions and revocation.",
                "26(4)": "Power to vary conditions."
            }
        ),

        # -------- Breach Regime --------
        LawCard(
            id="PDPA-26A",
            title="Section 26A — Interpretation of data breach",
            body=("Breach includes unauthorised access/collection/use/disclosure/copying/modification/disposal, or loss of media/device "
                  "where unauthorised processing is likely."),
            cites=["26A"],
            tags=["breach", "definition"],
            examples=["Lost unencrypted laptop with customer data."],
            weight=0.95,
            subsections=None
        ),
        LawCard(
            id="PDPA-26B",
            title="Section 26B — Notifiable data breach",
            body=("Notifiable if significant scale or likely significant harm; PDPC may prescribe thresholds per data types."),
            cites=["26C", "26D"],
            tags=["breach", "notifiability", "thresholds"],
            examples=["Exfiltration of many NRICs triggers notification."],
            weight=0.9,
            subsections=None
        ),
        LawCard(
            id="PDPA-26C",
            title="Section 26C — Data breach assessment",
            body=("Where there is reason to believe a breach occurred, conduct an assessment without unreasonable delay to determine if "
                  "it is notifiable."),
            cites=["26B", "26D"],
            tags=["breach", "assessment", "timeliness"],
            examples=["Start incident triage immediately on indicators of compromise."],
            weight=0.85,
            subsections={
                "26C(1)": "Assessment obligation on reason to believe.",
                "26C(2)": "Without unreasonable delay."
            }
        ),
        LawCard(
            id="PDPA-26D",
            title="Section 26D — Notification of data breach",
            body=("If notifiable, notify PDPC and affected individuals without unreasonable delay; PDPC no later than 3 calendar days "
                  "after becoming aware of notifiability."),
            cites=["26B", "26C"],
            tags=["breach", "PDPC notification", "3-day deadline"],
            examples=["File PDPC form within 3 days; parallel individual notices."],
            weight=0.95,
            subsections={
                "26D(1)": "Notify the Commission.",
                "26D(2)": "Notify affected individuals.",
                "26D(3)": "3 calendar days to notify PDPC from awareness of notifiability."
            }
        ),
        LawCard(
            id="PDPA-26E",
            title="Section 26E — Public agency data breach obligations",
            body=("For public-agency engagements, both the agency and engaged organisation must notify without unreasonable delay; the "
                  "organisation notifies PDPC and the agency; the agency must notify PDPC if it has not already been notified."),
            cites=["26D", "s24"],
            tags=["breach", "public agency", "data intermediary"],
            examples=["Vendor breach on a government project requires dual notifications."],
            weight=0.8,
            subsections=None
        ),

        # -------- Marketing adjuncts (kept) --------
        LawCard(
            id="DNC-Reg",
            title="DNC — Do Not Call Registry (Marketing Communications)",
            body=("DNC regulations govern marketing messages to Singapore numbers (calls/texts/faxes). PDPA consent to process data is "
                  "distinct from consent to send such messages; check registers or hold clear consent."),
            cites=["DNC regs"],
            tags=["DNC", "marketing", "SMS", "telemarketing", "consent"],
            examples=["Do not SMS numbers on DNC unless you have clear, specific consent."],
            weight=1.0
        ),
        LawCard(
            id="MKT-Email",
            title="Email marketing — consent & opt-out hygiene",
            body=("Promotional emails typically require prior consent and a working unsubscribe. Maintain robust opt-out processing and "
                  "auditable consent records even where exceptions might apply."),
            cites=["s13", "s16", "DNC regs"],
            tags=["marketing", "email", "unsubscribe"],
            examples=["Include single-click unsubscribe and action it promptly."],
            weight=0.9
        ),
    ]
    return kb


# -------------------------
# Embedding/Scoring utils
# -------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))

def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def _mmr(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    k: int,
    lambda_mult: float = 0.7
) -> List[int]:
    """
    Maximal Marginal Relevance selection to balance relevance & diversity.
    Returns list of selected indices.
    """
    if doc_vecs.shape[0] == 0:
        return []
    sim_to_query = (doc_vecs @ query_vec.reshape(-1, 1)).ravel()
    selected, candidates = [], list(range(doc_vecs.shape[0]))
    if not candidates:
        return []

    # pick the best first
    best_first = int(np.argmax(sim_to_query))
    selected.append(best_first)
    candidates.remove(best_first)

    while candidates and len(selected) < k:
        # diversity term: max similarity to any already selected
        sel_mat = doc_vecs[selected]
        cand_mat = doc_vecs[candidates]
        # (num_cand x num_sel)
        cross = cand_mat @ sel_mat.T
        max_sim_to_selected = cross.max(axis=1) if cross.size else np.zeros(len(candidates))

        mmr_scores = lambda_mult * sim_to_query[candidates] - (1 - lambda_mult) * max_sim_to_selected
        pick = int(np.argmax(mmr_scores))
        picked_idx = candidates[pick]
        selected.append(picked_idx)
        candidates.remove(picked_idx)

    return selected


# -------------------------
# Main engine
# -------------------------

class PDPAEmbeddingRAG:
    def __init__(
        self,
        bi_model_name: str = "intfloat/e5-base",
        ce_model_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_ce: bool = True
    ):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not available. Please install to use PDPAEmbeddingRAG.")

        self.bi = SentenceTransformer(bi_model_name)
        self.use_ce = use_ce and (ce_model_name is not None) and (CrossEncoder is not None)
        self.ce = None
        if self.use_ce:
            try:
                self.ce = CrossEncoder(ce_model_name)
            except Exception:
                # Graceful fallback: no CE if model can't be loaded
                self.ce = None
                self.use_ce = False

        # Build & embed KB
        self.cards: List[LawCard] = _build_pdpa_kb()
        # self._passages = [f"Section/Topic: {c.title}\n{c.body}\nTags: {', '.join(c.tags)}" for c in self.cards]

        def _render_passage(c: LawCard) -> str:
            subs = ""
            if c.subsections:
                items = [f"{k}: {v}" for k, v in c.subsections.items()]
                subs = "\nSubsections:\n- " + "\n- ".join(items)
            return f"Section/Topic: {c.title}\n{c.body}{subs}\nTags: {', '.join(c.tags)}"

        self._passages = [_render_passage(c) for c in self.cards]

        # e5: use "passage:" prefix
        self._emb = self._encode_passages(self._passages)
        self._emb = _l2norm(self._emb)  # cosine ready

    # ---------------------
    # Encoding helpers
    # ---------------------
    def _encode_queries(self, qs: List[str]) -> np.ndarray:
        # e5 expects "query: ..." prefix for optimal performance
        return self.bi.encode([f"query: {q}" for q in qs], show_progress_bar=False, normalize_embeddings=True)

    def _encode_passages(self, ps: List[str]) -> np.ndarray:
        # e5 expects "passage: ..." prefix for optimal performance
        return self.bi.encode([f"passage: {p}" for p in ps], show_progress_bar=False, normalize_embeddings=True)

    # ---------------------
    # Public API
    # ---------------------
    def analyze_legal_scenario(self, scenario_text: str, top_k: int = 5, use_mmr: bool = True) -> Dict[str, Any]:
        query = self._craft_query(scenario_text)

        qv = self._encode_queries([query])[0]  # (d,)
        sims = (self._emb @ qv)               # cosine similarities

        # Get candidate indices
        k_candidates = max(top_k * 3, 12)
        top_idx = np.argsort(sims)[::-1][:k_candidates].tolist()

        # Optional: MMR for diversity
        if use_mmr:
            selected_local = _mmr(qv, self._emb[top_idx], k=top_k, lambda_mult=0.7)
            idxs = [top_idx[i] for i in selected_local]
        else:
            idxs = top_idx[:top_k]

        # Optional: rerank with cross encoder
        details = self._materialize(query, idxs, sims[idxs], scenario_text)
        if self.use_ce and self.ce is not None and len(details) > 1:
            pairs = [[query, d["passage"]] for d in details]
            try:
                logits = np.array(self.ce.predict(pairs), dtype=float)
                scores = _sigmoid(logits)  # normalize to [0,1]
                for i, sc in enumerate(scores):
                    details[i]["relevance_score"] = float(np.clip(sc, 0.0, 1.0))
                    details[i]["raw_score"] = float(logits[i])
                details.sort(key=lambda x: x["relevance_score"], reverse=True)
            except Exception:
                # keep cosine if CE fails
                pass
        else:
            # normalize cosine to [0,1] for UI
            cos = np.array([d["raw_score"] for d in details])
            if cos.size:
                # min-max across current shortlist
                mn, mx = float(cos.min()), float(cos.max())
                den = (mx - mn) if (mx - mn) > 1e-9 else 1.0
                for d in details:
                    d["relevance_score"] = float((d["raw_score"] - mn) / den)

        # Build brief summary anchored on the best hit
        summary = None
        if details:
            primary = details[0]
            summary = {
                "primary_provision": f"{primary['id']} — {primary['title']}",
                "why": primary["explanation"]["why"]
            }

        return {
            "query": query,
            "scenario": scenario_text,
            "results": details[:top_k],
            "summary": summary
        }

    # ---------------------
    # Helpers
    # ---------------------
    def _craft_query(self, scenario_text: str) -> str:
        """
        Lightweight query normalizer + intent biasing for marketing scenarios.
        """
        t = scenario_text.strip()
        low = t.lower()
        bias_terms = []

        # If we spot marketing-ish language, bias towards consent + DNC
        if any(w in low for w in ["marketing", "promotional", "advertising", "newsletter", "email blast", "cold email", "spam"]):
            bias_terms += ["marketing", "email", "consent", "opt-out", "unsubscribe", "DNC"]

        # PDPA core actions
        if any(w in low for w in ["send", "sent", "email", "sms", "text", "whatsapp", "call"]):
            bias_terms += ["disclosure", "use", "contacting individuals"]

        # “No signup / never subscribed” → consent baseline + exceptions check
        if "never" in low or "not sign" in low or "without" in low:
            bias_terms += ["consent required", "exceptions", "schedules", "deemed consent limits"]

        bias = " ".join(sorted(set(bias_terms)))
        return (t if not bias else f"{t}\nFocus: {bias}").strip()

    def _materialize(self, query: str, idxs: List[int], sims: np.ndarray, scenario_text: str) -> List[Dict[str, Any]]:
        out = []
        for i, idx in enumerate(idxs):
            card = self.cards[idx]
            passage = self._passages[idx]
            raw = float(sims[i]) if isinstance(sims, np.ndarray) else float(sims[i])

            out.append({
                "id": card.id,
                "title": card.title,
                "raw_score": raw,
                "relevance_score": None,  # set later (CE or minmax)
                "tags": card.tags,
                "cross_references": card.cites[:5],
                "examples": card.examples[:2],
                "explanation": self._explain(card, query, scenario_text),
                "passage": passage,  # used only for CE rerank; you can drop from UI
            })
        return out

    def _explain(self, card: LawCard, query: str, scenario_text: str) -> Dict[str, Any]:
        q = query.lower()
        matched_tags = [t for t in card.tags if t in q]
        why = "Addresses consent baseline." if "PDPA-13" == card.id else \
              "Explains validity and notice requirements for consent." if "PDPA-14" == card.id else \
              "Covers deemed consent limits; not a free pass for broad marketing." if "PDPA-15" == card.id else \
              "Allows opt-out flow via notice + assessment (with exclusions)." if "PDPA-15A" == card.id else \
              "Confirms right to withdraw and need to cease marketing upon unsubscribe." if "PDPA-16" == card.id else \
              "Catalogs exceptions; you must verify the exact schedule basis." if "PDPA-17" == card.id else \
              "Purpose reasonableness and timing of notifications." if "PDPA-18-20" == card.id else \
              "DNC obligations for marketing messages are distinct from general PDPA consent." if "DNC-Reg" == card.id else \
              "Operational hygiene for email marketing: consent + opt-out processing." if "MKT-Email" == card.id else \
              "Relevant."
        return {
            "matched_tags": matched_tags[:4],
            "why": why
        }


# -------------------------
# Simple CLI demo
# -------------------------

def demo():
    engine = PDPAEmbeddingRAG()
    print("=== PDPA Semantic RAG (Embedding-First) ===")
    scenario = input("Enter your PDPA scenario: ").strip()
    res = engine.analyze_legal_scenario(scenario, top_k=5, use_mmr=True)
    print("\nQuery:", res["query"])
    print("\nTop hits:")
    for r in res["results"]:
        print(f" - {r['id']} | {r['title']} | score={r['relevance_score']:.3f} (raw={r['raw_score']:.3f})")
        print(f"   Why: {r['explanation']['why']}")
        if r['cross_references']:
            print(f"   Cross-refs: {', '.join(r['cross_references'])}")
        if r['examples']:
            print(f"   Examples: {', '.join(r['examples'])}")
    if res["summary"]:
        print("\nSummary:", res["summary"]["primary_provision"])
        print("Reason:", res["summary"]["why"])


if __name__ == "__main__":
    # Example:
    # "A retail company sent promotional emails to individuals who never signed up for marketing."
    demo()
