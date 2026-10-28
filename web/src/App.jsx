import React, { useState, useRef } from "react";
import { Upload, AlertTriangle, CheckCircle2, ImageIcon, Loader2 } from "lucide-react";

// Change this to your current ngrok URL if it changes between sessions
const API_URL = "https://eradicate-suitor-candy.ngrok-free.dev";

const CLASS_LABELS = {
  akiec: "Actinic keratosis",
  bcc: "Basal cell carcinoma",
  bkl: "Benign keratosis",
  df: "Dermatofibroma",
  mel: "Melanoma",
  nv: "Melanocytic nevus",
  vasc: "Vascular lesion",
};

const RISK_LEVEL = {
  akiec: "moderate",
  bcc: "high",
  bkl: "low",
  df: "low",
  mel: "high",
  nv: "low",
  vasc: "low",
};

const LOCALIZATIONS = [
  "back", "lower extremity", "trunk", "upper extremity", "abdomen",
  "face", "chest", "foot", "unknown", "neck", "scalp", "hand",
  "ear", "genital", "acral",
];

const COLORS = {
  ink: "#1C2B2D",
  bg: "#F7F5F2",
  teal: "#2D6A6F",
  tealDark: "#1F4C50",
  terracotta: "#B8542E",
  terracottaLight: "#FAECE7",
  terracottaText: "#712B13",
  sage: "#4A7856",
  sageLight: "#EAF3DE",
  sageText: "#27500A",
  amber: "#8A6116",
  amberLight: "#FAEEDA",
  border: "#DDD6CC",
  paper: "#FFFFFF",
  muted: "#5B6664",
  faint: "#9A9E97",
};

function RiskBadge({ level }) {
  const map = {
    low: { bg: COLORS.sageLight, text: COLORS.sageText, label: "Low concern" },
    moderate: { bg: COLORS.amberLight, text: COLORS.amber, label: "Moderate concern" },
    high: { bg: COLORS.terracottaLight, text: COLORS.terracottaText, label: "High concern" },
  };
  const s = map[level] || map.low;
  return (
    <span
      style={{
        display: "inline-block",
        background: s.bg,
        color: s.text,
        fontSize: 12.5,
        fontWeight: 500,
        padding: "4px 12px",
        borderRadius: 20,
      }}
    >
      {s.label}
    </span>
  );
}

function getRecommendation(level, needsReview) {
  if (level === "high" || needsReview) {
    return {
      text: "Consult a dermatologist promptly. This finding (or the model's uncertainty) warrants an in-person evaluation.",
      bg: COLORS.terracottaLight,
      text_color: COLORS.terracottaText,
      border: "#F0997B",
    };
  }
  if (level === "moderate") {
    return {
      text: "Consider a dermatologist visit for confirmation, especially if the lesion has changed recently.",
      bg: COLORS.amberLight,
      text_color: COLORS.amber,
      border: "#EF9F27",
    };
  }
  return {
    text: "Low concern based on this analysis. Routine self-monitoring is reasonable; see a dermatologist if the lesion changes.",
    bg: COLORS.sageLight,
    text_color: COLORS.sageText,
    border: "#97C459",
  };
}

function StatCard({ label, value, sub, accent }) {
  return (
    <div style={{ background: COLORS.bg, borderRadius: 10, padding: "14px 16px", flex: 1 }}>
      <p style={{ fontSize: 12.5, color: COLORS.muted, margin: 0 }}>{label}</p>
      <p
        style={{
          fontFamily: "'Source Serif 4', serif",
          fontSize: 26,
          fontWeight: 600,
          margin: "4px 0 0",
          color: accent || COLORS.ink,
        }}
      >
        {value}
      </p>
      {sub && <p style={{ fontSize: 12, color: COLORS.faint, margin: "2px 0 0" }}>{sub}</p>}
    </div>
  );
}

export default function SkinLesionDemo() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [age, setAge] = useState("45");
  const [sex, setSex] = useState("male");
  const [localization, setLocalization] = useState("back");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreviewUrl(e.target.result);
    reader.readAsDataURL(f);
  };

  const onDrop = (e) => {
    e.preventDefault();
    handleFile(e.dataTransfer.files?.[0]);
  };

  const runAnalysis = async () => {
    if (!file) {
      setError("Choose an image first.");
      return;
    }
    if (!age || Number(age) <= 0) {
      setError("Enter a valid age.");
      return;
    }
    setError(null);
    setLoading(true);
    setResult(null);

    try {
      const form = new FormData();
      form.append("image", file);
      form.append("age", age);
      form.append("sex", sex);
      form.append("localization", localization);

      const res = await fetch(`${API_URL}/predict`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`Server responded with ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(
        "Couldn't reach the analysis server. Check that the Kaggle backend session is still running."
      );
    } finally {
      setLoading(false);
    }
  };

  const riskLevel = result ? RISK_LEVEL[result.predicted_class] || "low" : null;

  return (
    <div
      style={{
        minHeight: "100vh",
        background: COLORS.bg,
        color: COLORS.ink,
        fontFamily: "'Inter', system-ui, sans-serif",
      }}
    >
      <link
        rel="stylesheet"
        href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600&family=Inter:wght@400;500;600&display=swap"
      />

      <div style={{ maxWidth: 640, margin: "0 auto", padding: "48px 24px 80px" }}>
        <header style={{ marginBottom: 32, display: "flex", alignItems: "flex-start", gap: 14 }}>
          <div
            style={{
              width: 44,
              height: 44,
              borderRadius: 10,
              background: COLORS.tealDark,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            <ImageIcon size={22} color="#F7F5F2" aria-hidden="true" />
          </div>
          <div>
            <h1
              style={{
                fontFamily: "'Source Serif 4', serif",
                fontWeight: 600,
                fontSize: 26,
                margin: 0,
                lineHeight: 1.25,
              }}
            >
              Skin lesion risk triage
            </h1>
            <p style={{ color: COLORS.muted, fontSize: 14.5, marginTop: 6, lineHeight: 1.6 }}>
              Uploads a dermoscopic image, estimates the lesion type, and
              flags low-confidence predictions for expert review.
            </p>
          </div>
        </header>

        <section
          style={{
            background: COLORS.paper,
            border: `0.5px solid ${COLORS.border}`,
            borderRadius: 14,
            padding: 24,
          }}
        >
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
            style={{
              border: `1.5px dashed ${previewUrl ? COLORS.teal : COLORS.border}`,
              borderRadius: 10,
              padding: previewUrl ? 0 : "36px 16px",
              textAlign: "center",
              cursor: "pointer",
              overflow: "hidden",
              background: previewUrl ? "#00000005" : COLORS.bg,
            }}
          >
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Selected lesion"
                style={{ width: "100%", maxHeight: 260, objectFit: "contain", display: "block" }}
              />
            ) : (
              <>
                <Upload size={26} color={COLORS.muted} style={{ marginBottom: 8 }} aria-hidden="true" />
                <p style={{ fontSize: 15, fontWeight: 500, margin: 0 }}>
                  Drop an image here or tap to choose one
                </p>
                <p style={{ fontSize: 13, color: COLORS.faint, marginTop: 4 }}>
                  JPG or PNG, dermoscopic image
                </p>
              </>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              style={{ display: "none" }}
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 20 }}>
            <label style={{ fontSize: 13, color: COLORS.muted }}>
              Age
              <input
                type="number"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                style={{
                  display: "block",
                  width: "100%",
                  marginTop: 6,
                  padding: "9px 10px",
                  borderRadius: 8,
                  border: `0.5px solid ${COLORS.border}`,
                  fontSize: 14,
                  boxSizing: "border-box",
                }}
              />
            </label>

            <label style={{ fontSize: 13, color: COLORS.muted }}>
              Sex
              <select
                value={sex}
                onChange={(e) => setSex(e.target.value)}
                style={{
                  display: "block",
                  width: "100%",
                  marginTop: 6,
                  padding: "9px 10px",
                  borderRadius: 8,
                  border: `0.5px solid ${COLORS.border}`,
                  fontSize: 14,
                  boxSizing: "border-box",
                  background: COLORS.paper,
                }}
              >
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="unknown">Unknown</option>
              </select>
            </label>

            <label style={{ fontSize: 13, color: COLORS.muted, gridColumn: "1 / -1" }}>
              Lesion location
              <select
                value={localization}
                onChange={(e) => setLocalization(e.target.value)}
                style={{
                  display: "block",
                  width: "100%",
                  marginTop: 6,
                  padding: "9px 10px",
                  borderRadius: 8,
                  border: `0.5px solid ${COLORS.border}`,
                  fontSize: 14,
                  boxSizing: "border-box",
                  background: COLORS.paper,
                }}
              >
                {LOCALIZATIONS.map((loc) => (
                  <option key={loc} value={loc}>{loc}</option>
                ))}
              </select>
            </label>
          </div>

          {error && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginTop: 14,
                color: COLORS.terracottaText,
                fontSize: 13,
              }}
            >
              <AlertTriangle size={15} aria-hidden="true" />
              <span>{error}</span>
            </div>
          )}

          <button
            onClick={runAnalysis}
            disabled={loading}
            style={{
              marginTop: 20,
              width: "100%",
              padding: "12px 0",
              background: COLORS.tealDark,
              color: "#F7F5F2",
              border: "none",
              borderRadius: 8,
              fontSize: 15,
              fontWeight: 500,
              cursor: loading ? "default" : "pointer",
              opacity: loading ? 0.75 : 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 8,
            }}
          >
            {loading && <Loader2 size={16} className="spin" aria-hidden="true" />}
            {loading ? "Analyzing…" : "Analyze image"}
          </button>
        </section>

        {result && (
          <section style={{ marginTop: 24 }}>
            {result.needs_expert_review ? (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  background: COLORS.terracottaLight,
                  border: "0.5px solid #F0997B",
                  color: COLORS.terracottaText,
                  borderRadius: 10,
                  padding: "12px 16px",
                  fontSize: 14,
                  fontWeight: 500,
                  marginBottom: 16,
                }}
              >
                <AlertTriangle size={18} aria-hidden="true" />
                Flagged for expert review — confidence is below the safety threshold.
              </div>
            ) : (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  background: COLORS.sageLight,
                  border: "0.5px solid #97C459",
                  color: COLORS.sageText,
                  borderRadius: 10,
                  padding: "12px 16px",
                  fontSize: 14,
                  fontWeight: 500,
                  marginBottom: 16,
                }}
              >
                <CheckCircle2 size={18} aria-hidden="true" />
                Confidence is within the accepted range — no review flag raised.
              </div>
            )}

            <div
              style={{
                background: COLORS.paper,
                border: `0.5px solid ${COLORS.border}`,
                borderRadius: 14,
                padding: 24,
              }}
            >
              <div>
                <p style={{ fontSize: 12.5, color: COLORS.faint, margin: 0 }}>Predicted lesion type</p>
                <h2
                  style={{
                    fontFamily: "'Source Serif 4', serif",
                    fontWeight: 600,
                    fontSize: 24,
                    margin: "4px 0 8px",
                  }}
                >
                  {CLASS_LABELS[result.predicted_class] || result.predicted_class}
                </h2>
                <RiskBadge level={riskLevel} />
              </div>

              <div style={{ display: "flex", gap: 12, marginTop: 20 }}>
                <StatCard
                  label="Confidence"
                  value={`${(result.confidence * 100).toFixed(1)}%`}
                  accent={COLORS.teal}
                />
                <StatCard
                  label="Uncertainty"
                  value={`${(result.uncertainty * 100).toFixed(1)}%`}
                  accent={result.uncertainty > 0.3 ? COLORS.terracotta : COLORS.sage}
                  sub={result.uncertainty > 0.3 ? "Above safety threshold" : "Within accepted range"}
                />
              </div>

              {(() => {
                const rec = getRecommendation(riskLevel, result.needs_expert_review);
                return (
                  <div
                    style={{
                      marginTop: 20,
                      background: rec.bg,
                      border: `0.5px solid ${rec.border}`,
                      borderRadius: 10,
                      padding: "12px 16px",
                    }}
                  >
                    <p style={{ fontSize: 12, fontWeight: 500, color: rec.text_color, margin: "0 0 2px" }}>
                      Recommendation
                    </p>
                    <p style={{ fontSize: 13.5, color: rec.text_color, margin: 0, lineHeight: 1.5 }}>
                      {rec.text}
                    </p>
                  </div>
                );
              })()}

              <div style={{ marginTop: 24 }}>
                <p style={{ fontSize: 13, color: COLORS.muted, marginBottom: 10, fontWeight: 500 }}>
                  Probability by class
                </p>
                {Object.entries(result.class_probabilities)
                  .sort((a, b) => b[1] - a[1])
                  .map(([cls, prob]) => (
                    <div key={cls} style={{ marginBottom: 9 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5, marginBottom: 3 }}>
                        <span style={{ color: COLORS.muted }}>{CLASS_LABELS[cls] || cls}</span>
                        <span style={{ fontWeight: cls === result.predicted_class ? 500 : 400 }}>
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div style={{ height: 6, borderRadius: 3, background: COLORS.bg, overflow: "hidden" }}>
                        <div
                          style={{
                            height: "100%",
                            width: `${prob * 100}%`,
                            background: cls === result.predicted_class ? COLORS.teal : "#C9C3B6",
                            borderRadius: 3,
                          }}
                        />
                      </div>
                    </div>
                  ))}
              </div>

              <div
                style={{
                  marginTop: 20,
                  paddingTop: 16,
                  borderTop: `0.5px solid ${COLORS.border}`,
                  fontSize: 12.5,
                  color: COLORS.faint,
                  display: "flex",
                  justifyContent: "space-between",
                }}
              >
                <span>Patient: {age}y, {sex}</span>
                <span>Site: {localization}</span>
              </div>
            </div>
          </section>
        )}

        <p style={{ fontSize: 12, color: COLORS.faint, marginTop: 32, lineHeight: 1.6 }}>
          Research demo only — not a diagnostic device. Predictions should
          always be reviewed by a qualified dermatologist.
        </p>
      </div>

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .spin { animation: spin 0.8s linear infinite; }
      `}</style>
    </div>
  );
}
