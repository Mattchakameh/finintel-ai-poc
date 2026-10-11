from pathlib import Path
import pandas as pd
import streamlit as st
from src.meeting04_loader import load_meeting04, Meeting04EvidenceError

MODEL_NAMES = {
    "A": "A — Query + Candidate",
    "B": "B — + SEC Table Text",
    "C": "C — + Image (Concat)",
    "D": "D — FinIntel Learned Fusion",
}
CONCEPTS = ["Revenue", "Assets", "Liabilities", "Equity", "Net Income", "Cash"]
M04_VISUAL_DIR = Path(__file__).resolve().parents[1] / "assets" / "meeting04"

def _pct(x):
    return f"{100*x:.2f}%"

def render_meeting04():
    try:
        s, p, h, c, checks = load_meeting04()
    except Meeting04EvidenceError as e:
        st.error(str(e))
        st.stop()

    st.title("FinIntel AI — Meeting 04")
    st.subheader("Company-Disjoint Multimodal Financial Grounding & Fusion Evaluation")
    st.caption(
        "Independent Meeting 04 experiment. Archived research outputs only: "
        "interactive controls recompute views over frozen results; they do not retrain models."
    )

    a,b,d,e = st.columns(4)
    a.metric("Financial Facts", "900")
    b.metric("Companies", "30")
    d.metric("Fiscal Years", "5")
    e.metric("Financial Concepts", "6")
    st.markdown("**Six concepts:** Revenue · Assets · Liabilities · Equity · Net Income · Cash")

    tabs = st.tabs([
        "Research Overview", "Live Grounding Explorer", "SEC Visual Evidence",
        "Model Playground", "Training Lab", "Fusion & Conditions", "Power & Boundary"
    ])

    with tabs[0]:
        x,y,z = st.columns(3)
        x.metric("Direct SEC/XBRL", "865")
        y.metric("Reconstructed Liabilities", "35")
        z.metric("Clean Experimental Targets", "788")
        x,y,z = st.columns(3)
        x.metric("Train", "543")
        y.metric("Validation", "115")
        z.metric("Frozen Test", "130")
        st.success("20/20 full runs complete — Models A/B/C/D × seeds 11/22/33/44/55.")
        st.dataframe(
            pd.DataFrame([{"Check":k,"Status":"PASS" if ok else "FAIL"} for k,ok in checks.items()]),
            hide_index=True, use_container_width=True
        )

    with tabs[1]:
        st.markdown("### Live Frozen-Test Grounding Explorer")
        tick = st.selectbox("Company", sorted(c.ticker.unique()), key="m04tick")
        cc = c[c.ticker.eq(tick)]
        fy = st.selectbox("Fiscal year", sorted(cc.target_fiscal_year.unique()), key="m04fy")
        cc = cc[cc.target_fiscal_year.eq(fy)]
        metric = st.selectbox("Concept", sorted(cc.metric.unique()), key="m04metric")
        cc = cc[cc.metric.eq(metric)]
        if len(cc):
            gid = cc.grounding_id.iloc[0]
            st.caption(f"Grounding ID: {gid}")
            show = cc[["candidate_value","candidate_sec_fy","candidate_type","label"]].reset_index(drop=True)
            show.index.name = "Candidate index"
            st.dataframe(show, use_container_width=True)
            st.caption(
                "candidate_sec_fy is the SEC filing FY associated with the candidate; "
                "it is not automatically the target fiscal year."
            )

    with tabs[2]:
        st.markdown("### Original SEC Visual Evidence Lab")
        st.caption(
            "Browser-rendered original SEC table evidence archived during Meeting 04. "
            "These images are evidence artifacts only; changing controls does not retrain or rerun the model."
        )
        available = sorted(M04_VISUAL_DIR.glob("*.png")) if M04_VISUAL_DIR.exists() else []
        if not available:
            st.warning("No Meeting 04 SEC visual assets are bundled in this deployment yet.")
        else:
            tickers = sorted({x.name.split("_FY")[0] for x in available})
            vtick = st.selectbox("Evidence company", tickers, key="vistick")
            company_imgs = [x for x in available if x.name.startswith(vtick + "_FY")]
            years = sorted({int(x.name.split("_FY")[1].split("_TABLE")[0]) for x in company_imgs})
            vfy = st.selectbox("Evidence fiscal year", years, key="visfy")
            year_imgs = [x for x in company_imgs if f"_FY{vfy}_" in x.name]
            table_names = [x.name for x in year_imgs]
            selected_name = st.selectbox("SEC table region", table_names, key="vistable")
            selected_path = M04_VISUAL_DIR / selected_name
            left, right = st.columns([1.35, 1])
            with left:
                st.image(str(selected_path), caption=selected_name, use_container_width=True)
            with right:
                st.markdown("#### Grounding + archived prediction")
                cq = c[(c.ticker.eq(vtick)) & (c.target_fiscal_year.eq(vfy))].copy()
                metrics_here = sorted(cq.metric.unique())
                if metrics_here:
                    vmetric = st.selectbox("Financial concept", metrics_here, key="vismetric")
                    cq = cq[cq.metric.eq(vmetric)]
                    if len(cq):
                        gid = cq.grounding_id.iloc[0]
                        st.caption(f"Grounding ID: {gid}")
                        show = cq[["candidate_value","candidate_sec_fy","candidate_type","label"]].reset_index(drop=True)
                        show.index.name = "Candidate index"
                        st.dataframe(show, use_container_width=True)
                        vm = st.selectbox("Archived model", list(MODEL_NAMES), format_func=lambda x: MODEL_NAMES[x], key="vismodel")
                        vs = st.selectbox("Seed", [11,22,33,44,55], key="visseed")
                        pred = p[(p.model.eq(vm)) & (p.seed.eq(vs)) & (p.grounding_id.eq(gid))]
                        if len(pred):
                            r = pred.iloc[0]
                            a1,a2,a3 = st.columns(3)
                            a1.metric("True value", f"{r.true_candidate_value:,.0f}")
                            a2.metric("Predicted value", f"{r.predicted_candidate_value:,.0f}")
                            a3.metric("Joint", "CORRECT" if bool(r.joint_correct) else "WRONG")
                        else:
                            st.info("This grounding target is outside the clean Frozen-Test prediction cohort.")
                st.caption(
                    "The SEC image and candidate/prediction records are archived artifacts. "
                    "Table-region selection is visual inspection; candidate_sec_fy is filing FY metadata."
                )

    with tabs[3]:
        st.markdown("### Archived Model Playground")
        m = st.selectbox("Model", list(MODEL_NAMES), format_func=lambda x: MODEL_NAMES[x])
        seed = st.selectbox("Seed", [11,22,33,44,55])
        q = p[(p.model.eq(m)) & (p.seed.eq(seed))].copy()
        tick = st.selectbox("Frozen-Test company", sorted(q.ticker.unique()), key="playtick")
        q = q[q.ticker.eq(tick)]
        metric = st.selectbox("Concept", sorted(q.metric.unique()), key="playmetric")
        q = q[q.metric.eq(metric)]
        x,y,z = st.columns(3)
        x.metric("Value Accuracy", _pct(q.value_correct.mean()))
        y.metric("Concept Accuracy", _pct(q.concept_correct.mean()))
        z.metric("Joint Accuracy", _pct(q.joint_correct.mean()))
        st.dataframe(
            q[["target_fiscal_year","metric","true_candidate_value","predicted_candidate_value",
               "value_correct","concept_correct","joint_correct"]],
            hide_index=True, use_container_width=True
        )

    with tabs[4]:
        st.markdown("### Training Diagnostics")
        m = st.selectbox("Training model", list(MODEL_NAMES), format_func=lambda x: MODEL_NAMES[x], key="tm")
        seed = st.selectbox("Training seed", [11,22,33,44,55], key="ts")
        q = h[(h.model.eq(m)) & (h.seed.eq(seed))].sort_values("epoch")
        st.line_chart(q.set_index("epoch")[["train_loss","val_loss"]])
        st.line_chart(q.set_index("epoch")[["train_joint_accuracy","val_joint_accuracy"]])
        best = q.loc[q.val_joint_accuracy.idxmax()]
        x,y = st.columns(2)
        x.metric("Best epoch", int(best.epoch))
        y.metric("Best validation Joint", _pct(best.val_joint_accuracy))

    with tabs[5]:
        st.markdown("### Model Comparison — Live Filters Over Frozen Test")
        companies = st.multiselect("Companies", sorted(p.ticker.unique()), default=sorted(p.ticker.unique()))
        metrics = st.multiselect("Concepts", sorted(p.metric.unique()), default=sorted(p.metric.unique()))
        q = p[p.ticker.isin(companies) & p.metric.isin(metrics)]
        if len(q):
            agg = q.groupby("model").agg(
                Value=("value_correct","mean"),
                Concept=("concept_correct","mean"),
                Joint=("joint_correct","mean")
            ).reset_index()
            st.dataframe(
                agg.style.format({"Value":"{:.2%}","Concept":"{:.2%}","Joint":"{:.2%}"}),
                hide_index=True, use_container_width=True
            )
            st.bar_chart(agg.set_index("model")[["Value","Joint"]])
        st.info(
            "Full Frozen Test mean Joint: A 23.85% · B 25.69% · C 24.92% · D 27.54%. "
            "D − C = +2.62 pp. This is an observed experimental difference, not proof of superiority."
        )
        st.warning(
            "Primary bottleneck: value grounding/candidate discrimination. Concept prediction is near ceiling. "
            "94.27% of Model D errors selected a candidate associated with a different SEC filing FY."
        )

    with tabs[6]:
        st.markdown("### Power Analysis & Research Boundary")
        a,b,d = st.columns(3)
        a.metric("Observed D − C", "+2.62 pp")
        b.metric("Current Power", "19.9%")
        d.metric("Prospective Planning N", "816 targets")
        st.caption(
            "816 is a prospective planning estimate under paired-target assumptions, "
            "not a retrospective significance claim. Future confirmatory evaluation "
            "should expand both targets and companies."
        )
        st.warning(
            "Meeting 04 does not formally test the official Praxis H1/H2/H3. "
            "Model D is learned summary-level fusion; it is not yet token↔patch Adaptive Cross-Attention. "
            "Frozen Test was not used for training, tuning, or checkpoint selection."
        )
