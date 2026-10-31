import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Upload, AlertTriangle, CheckCircle2, Loader2, Sun, Moon, Activity,
  Stethoscope, ShieldCheck, Download, Printer, RotateCcw, Settings,
  Info, History, BookOpen, BarChart3, Wifi, WifiOff, X, Layers, ScanSearch,
} from "lucide-react";

/* ────────────────────────────────────────────────────────────────
   CONFIG
   The API URL can be set 3 ways (highest priority first):
   1. Settings panel in the app (saved in this browser)
   2. VITE_API_URL environment variable (set it in Vercel)
   3. The fallback below
   ──────────────────────────────────────────────────────────────── */
const FALLBACK_API = "https://eradicate-suitor-candy.ngrok-free.dev";
const DEFAULT_API = (import.meta.env && import.meta.env.VITE_API_URL) || FALLBACK_API;
const API_KEY_STORE = "triage_api_url";
const THEME_KEY = "triage_theme";
const MAX_FILE_MB = 10;

// Mirrors core/risk_triage.py — keep in sync if you change the backend.
const THRESH = { low: 0.25, mid: 0.45 };
// Mirrors core/quality_check.py
const MIN_RESOLUTION = 100;

/* ────────────────────────────────────────────────────────────────
   DATA
   ──────────────────────────────────────────────────────────────── */
const CLASSES = {
  akiec: { name: "Actinic keratosis / intraepithelial carcinoma", cat: "premalignant", blurb: "Rough, scaly patch from sun damage. Can progress to squamous cell carcinoma if untreated." },
  bcc: { name: "Basal cell carcinoma", cat: "malignant", blurb: "Most common skin cancer. Slow-growing and rarely spreads, but needs treatment." },
  bkl: { name: "Benign keratosis-like lesion", cat: "benign", blurb: "Includes seborrheic keratoses and solar lentigines. Harmless waxy or scaly spots." },
  df: { name: "Dermatofibroma", cat: "benign", blurb: "Firm, harmless skin nodule, often on the legs." },
  mel: { name: "Melanoma", cat: "malignant", blurb: "The most dangerous skin cancer. Early detection greatly improves outcomes." },
  nv: { name: "Melanocytic nevus (mole)", cat: "benign", blurb: "Common mole. Usually harmless, but monitor for changes." },
  vasc: { name: "Vascular lesion", cat: "benign", blurb: "Blood-vessel growths such as angiomas and cherry angiomas." },
};

const CAT = {
  benign: { label: "Benign class", tone: "sage" },
  premalignant: { label: "Pre-malignant class", tone: "amber" },
  malignant: { label: "Malignant class", tone: "red" },
};

const TRIAGE = {
  low_risk: {
    tone: "sage", Icon: ShieldCheck, title: "Low Risk",
    advice: "Routine self-monitoring is reasonable. See a dermatologist if the lesion changes in size, shape, colour or symptoms.",
  },
  suspicious: {
    tone: "amber", Icon: AlertTriangle, title: "Suspicious",
    advice: "Book a dermatologist visit for confirmation, especially if the lesion has changed recently.",
  },
  expert_review: {
    tone: "red", Icon: Stethoscope, title: "Expert Review Recommended",
    advice: "Consult a dermatologist promptly. Either the model is not confident or the image quality is poor.",
  },
};

const HIGH_STAKES = new Set(["mel", "bcc", "akiec"]);

// Mirrors core/risk_triage.py. Used only when the backend response has no valid
// risk_level (e.g. an older backend), so the UI never defaults to "Expert Review".
function deriveRiskLevel(r) {
  if (r && TRIAGE[r.risk_level]) return r.risk_level;
  if (r?.image_quality && r.image_quality.passed === false) return "expert_review";
  const u = Number(r?.uncertainty);
  if (r?.needs_expert_review === true && !(u <= THRESH.mid)) return "expert_review";
  if (u > THRESH.mid) return "expert_review";
  if (HIGH_STAKES.has(r?.predicted_class) || u > THRESH.low) return "suspicious";
  return "low_risk";
}

const LOCALIZATIONS = [
  "back", "lower extremity", "trunk", "upper extremity", "abdomen",
  "face", "chest", "foot", "unknown", "neck", "scalp", "hand",
  "ear", "genital", "acral",
];

const SKIN_TYPES = [
  ["", "Not specified"],
  ["type_I", "I: always burns, never tans"],
  ["type_II", "II: burns easily, tans slightly"],
  ["type_III", "III: burns, then tans"],
  ["type_IV", "IV: rarely burns, tans well"],
  ["type_V", "V: brown skin"],
  ["type_VI", "VI: dark brown / black skin"],
];
const EVOLUTION = [["", "Not specified"], ["stable", "Stable / unchanged"], ["changed_recently", "Changed recently"], ["unknown", "Don't know"]];
const SYMPTOMS = [["", "Not specified"], ["none", "None"], ["itching", "Itching"], ["bleeding", "Bleeding"], ["pain", "Pain"], ["other", "Other"]];
const HISTORY = [["", "Not specified"], ["none", "None"], ["prior_skin_cancer", "Prior skin cancer"], ["family_history", "Family history"], ["other", "Other"]];

// Figures copied from the project README (not computed live).
const METRICS_README = {
  accuracy: "93.4%",
  ece: "0.125",
  auroc: "0.87",
  uncCorrect: 0.19,
  uncWrong: 0.39,
  inferenceSingle: "~35 ms",
  inferenceBatched: "~13.5 ms",
  melBefore: "73%",
  melAfter: "82%",
};

const TIPS_ABCDE = [
  ["A", "Asymmetry", "One half does not match the other."],
  ["B", "Border", "Edges are ragged, notched or blurred."],
  ["C", "Colour", "Several shades of brown, black, red, white or blue."],
  ["D", "Diameter", "Larger than about 6 mm (a pencil eraser)."],
  ["E", "Evolving", "Changing in size, shape, colour, or starts to bleed or itch."],
];

/* ────────────────────────────────────────────────────────────────
   HELPERS
   ──────────────────────────────────────────────────────────────── */
const clamp = (v, lo = 0, hi = 100) => Math.min(hi, Math.max(lo, v));
const fmtPct = (v, d = 1) => `${(Number(v) * 100).toFixed(d)}%`;
const normUrl = (u) => (u || "").trim().replace(/\/+$/, "");
const asPct = (v) => (v == null ? "—" : `${(v <= 1 ? v * 100 : v).toFixed(1)}%`);

async function api(base, path, opts = {}) {
  const headers = { "ngrok-skip-browser-warning": "true", ...(opts.headers || {}) };
  return fetch(`${base}${path}`, { ...opts, headers });
}

function readImageDims(dataUrl) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve({ w: img.naturalWidth, h: img.naturalHeight });
    img.onerror = () => resolve(null);
    img.src = dataUrl;
  });
}

function getRedFlags(c) {
  const f = [];
  if (c.evolution === "changed_recently") f.push("Lesion has changed recently");
  if (["itching", "bleeding", "pain"].includes(c.symptoms)) f.push(`Reported symptom: ${c.symptoms}`);
  if (c.history === "prior_skin_cancer") f.push("Prior skin cancer");
  if (c.history === "family_history") f.push("Family history of skin disease");
  return f;
}

/* ────────────────────────────────────────────────────────────────
   SMALL COMPONENTS
   ──────────────────────────────────────────────────────────────── */
function Chip({ tone = "muted", children }) {
  return <span className={`chip chip-${tone}`}>{children}</span>;
}

function Card({ title, icon, right, children, className = "" }) {
  return (
    <section className={`card ${className}`}>
      {(title || right) && (
        <div className="card-h">
          <h3>{icon}{title}</h3>
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

function Select({ label, value, onChange, options, full }) {
  return (
    <label className={`field ${full ? "full" : ""}`}>
      <span className="lbl">{label}</span>
      <select className="input" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => {
          const [v, t] = Array.isArray(o) ? o : [o, o];
          return <option key={v} value={v}>{t}</option>;
        })}
      </select>
    </label>
  );
}

function Stat({ label, value, sub, tone }) {
  return (
    <div className="stat">
      <p className="stat-l">{label}</p>
      <p className={`stat-v ${tone ? `t-${tone}` : ""}`}>{value}</p>
      {sub && <p className="stat-s">{sub}</p>}
    </div>
  );
}

function UncertaintyGauge({ value }) {
  const pos = clamp(value * 100);
  return (
    <div>
      <div className="gauge" role="img" aria-label={`Uncertainty ${fmtPct(value)}`}>
        <i className="zone z-sage" style={{ width: `${THRESH.low * 100}%` }} />
        <i className="zone z-amber" style={{ width: `${(THRESH.mid - THRESH.low) * 100}%` }} />
        <i className="zone z-red" style={{ width: `${(1 - THRESH.mid) * 100}%` }} />
        <b className="needle" style={{ left: `${pos}%` }} />
      </div>
      <div className="gauge-l">
        <span style={{ left: 0 }}>0</span>
        <span style={{ left: `${THRESH.low * 100}%` }}>{THRESH.low}</span>
        <span style={{ left: `${THRESH.mid * 100}%` }}>{THRESH.mid}</span>
        <span style={{ right: 0 }}>1</span>
      </div>
    </div>
  );
}

function ProbBars({ probs, predicted }) {
  const rows = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  return (
    <div>
      {rows.map(([cls, p], i) => {
        const meta = CLASSES[cls];
        const tone = meta ? CAT[meta.cat].tone : "muted";
        return (
          <div key={cls} className="prob">
            <div className="prob-t">
              <span>
                {i < 3 && <em className="rank">{i + 1}</em>}
                {meta?.name || cls}
              </span>
              <strong className={cls === predicted ? "" : "dim"}>{fmtPct(p)}</strong>
            </div>
            <div className="track">
              <div className={`fill f-${tone} ${cls === predicted ? "" : "ghost"}`} style={{ width: `${clamp(p * 100)}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function QualityPanel({ server, dims }) {
  const clientLow = dims && Math.min(dims.w, dims.h) < MIN_RESOLUTION;
  const passed = server ? server.passed : !clientLow;
  const issues = server?.issues || [];
  return (
    <div>
      <div className="row-between">
        <Chip tone={passed ? "sage" : "red"}>{passed ? "Quality check passed" : "Quality issues found"}</Chip>
        {dims && <span className="muted-s">{dims.w} × {dims.h}px</span>}
      </div>
      {issues.length > 0 && (
        <ul className="issues">
          {issues.map((t) => <li key={t}>{t}</li>)}
        </ul>
      )}
      {!server && clientLow && (
        <p className="note" style={{ marginTop: 8 }}>
          Shorter side is under {MIN_RESOLUTION}px, so this will be flagged for expert review.
        </p>
      )}
      {server && (
        <p className="note" style={{ marginTop: 8 }}>
          Checks run: resolution, sharpness (Laplacian variance) and exposure.
        </p>
      )}
      {!server && (
        <p className="note" style={{ marginTop: 8 }}>
          Resolution is checked here. Sharpness and exposure are checked by the server when you analyze.
        </p>
      )}
    </div>
  );
}

/* ────────────────────────────────────────────────────────────────
   MAIN
   ──────────────────────────────────────────────────────────────── */
export default function SkinLesionTriage() {
  /* app state */
  const [tab, setTab] = useState("analyze");
  const [theme, setTheme] = useState(() => {
    try {
      const s = localStorage.getItem(THEME_KEY);
      if (s) return s;
    } catch { /* ignore */ }
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  const [apiUrl, setApiUrl] = useState(() => {
    try { return localStorage.getItem(API_KEY_STORE) || DEFAULT_API; } catch { return DEFAULT_API; }
  });
  const [apiDraft, setApiDraft] = useState(apiUrl);
  const [showSettings, setShowSettings] = useState(false);
  const [apiStatus, setApiStatus] = useState({ state: "checking", device: null });
  const [modelInfo, setModelInfo] = useState(null);

  /* form state */
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [dims, setDims] = useState(null);
  const [ctx, setCtx] = useState({
    age: "45", sex: "male", localization: "back",
    skin_color: "", evolution: "", symptoms: "", history: "",
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [showHeat, setShowHeat] = useState(false);
  const [heatOpacity, setHeatOpacity] = useState(60);
  const fileInputRef = useRef(null);
  const resultRef = useRef(null);

  const setField = (k) => (v) => setCtx((c) => ({ ...c, [k]: v }));

  /* theme persistence */
  useEffect(() => {
    try { localStorage.setItem(THEME_KEY, theme); } catch { /* ignore */ }
  }, [theme]);

  /* backend health + model info */
  const checkApi = useCallback(async (base) => {
    setApiStatus({ state: "checking", device: null });
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 12000);
      const res = await api(base, "/health", { signal: ctrl.signal });
      clearTimeout(t);
      if (!res.ok) throw new Error("bad status");
      const j = await res.json();
      setApiStatus({ state: "online", device: j.device || null });
      try {
        const mi = await api(base, "/model-info");
        if (mi.ok) setModelInfo(await mi.json());
      } catch { /* model-info is optional */ }
    } catch {
      setApiStatus({ state: "offline", device: null });
    }
  }, []);

  useEffect(() => { checkApi(apiUrl); }, [apiUrl, checkApi]);

  const saveApi = () => {
    const u = normUrl(apiDraft) || DEFAULT_API;
    setApiDraft(u);
    try { localStorage.setItem(API_KEY_STORE, u); } catch { /* ignore */ }
    if (u === apiUrl) checkApi(u); else setApiUrl(u);
  };
  const resetApi = () => {
    setApiDraft(DEFAULT_API);
    try { localStorage.removeItem(API_KEY_STORE); } catch { /* ignore */ }
    if (DEFAULT_API === apiUrl) checkApi(DEFAULT_API); else setApiUrl(DEFAULT_API);
  };

  /* file handling */
  const handleFile = useCallback((f) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) { setError("That file isn't an image. Use JPG, PNG or WEBP."); return; }
    if (f.size > MAX_FILE_MB * 1024 * 1024) { setError(`Image is larger than ${MAX_FILE_MB} MB.`); return; }
    setError(null);
    setFile(f);
    setResult(null);
    setShowHeat(false);
    const reader = new FileReader();
    reader.onload = async (e) => {
      setPreviewUrl(e.target.result);
      setDims(await readImageDims(e.target.result));
    };
    reader.readAsDataURL(f);
  }, []);

  /* paste an image from the clipboard (Ctrl/Cmd+V) */
  useEffect(() => {
    const onPaste = (e) => {
      const f = Array.from(e.clipboardData?.files || []).find((x) => x.type.startsWith("image/"));
      if (f) { setTab("analyze"); handleFile(f); }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [handleFile]);

  const onDrop = (e) => {
    e.preventDefault();
    handleFile(e.dataTransfer.files?.[0]);
  };

  const resetAll = () => {
    setFile(null); setPreviewUrl(null); setDims(null);
    setResult(null); setError(null); setShowHeat(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  /* analysis */
  const runAnalysis = async () => {
    if (!file) { setError("Choose an image first."); return; }
    const ageNum = Number(ctx.age);
    if (!ctx.age || !(ageNum > 0 && ageNum <= 120)) { setError("Enter a valid age between 1 and 120."); return; }
    setError(null);
    setLoading(true);
    setResult(null);
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 90000);
    try {
      const form = new FormData();
      form.append("image", file);
      form.append("age", ctx.age);
      form.append("sex", ctx.sex);
      form.append("localization", ctx.localization);
      // Extension-point fields: sent for forward-compatibility, but the
      // current model ignores them (see core/metadata_schema.py).
      if (ctx.skin_color) form.append("skin_color", ctx.skin_color);
      if (ctx.evolution) form.append("evolution", ctx.evolution);
      if (ctx.symptoms) form.append("symptoms", ctx.symptoms);
      if (ctx.history) form.append("previous_disease_history", ctx.history);

      const res = await api(apiUrl, "/predict", { method: "POST", body: form, signal: ctrl.signal });
      let data = null;
      try { data = await res.json(); } catch { /* non-JSON response */ }
      if (!res.ok) throw new Error(data?.error || `Server responded with ${res.status}`);
      setResult(data);
      setApiStatus((s) => ({ ...s, state: "online" }));
      setHistory((h) => [
        { id: Date.now(), at: new Date(), file, previewUrl, dims, ctx: { ...ctx }, result: data },
        ...h,
      ].slice(0, 6));
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 80);
    } catch (err) {
      if (err.name === "AbortError") {
        setError("The request timed out. The backend may be waking up or overloaded. Try again.");
      } else if (err.message && !/failed to fetch|networkerror|load failed/i.test(err.message)) {
        setError(`Analysis failed: ${err.message}`);
      } else {
        setApiStatus((s) => ({ ...s, state: "offline" }));
        setError("Couldn't reach the analysis server. Check that the backend is running, then update the URL in Settings.");
      }
    } finally {
      clearTimeout(timer);
      setLoading(false);
    }
  };

  const restore = (h) => {
    setFile(h.file); setPreviewUrl(h.previewUrl); setDims(h.dims);
    setCtx(h.ctx); setResult(h.result); setError(null); setShowHeat(false);
    setTab("analyze");
    setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 80);
  };

  const downloadJson = () => {
    const payload = {
      generated_at: new Date().toISOString(),
      inputs: ctx,
      image: file ? { name: file.name, size_bytes: file.size, ...(dims || {}) } : null,
      model_output: result,
      note: "Academic research prototype. Not a diagnostic device.",
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `triage-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  /* derived */
  const triage = result ? TRIAGE[deriveRiskLevel(result)] : null;
  const cls = result ? CLASSES[result.predicted_class] : null;
  const redFlags = getRedFlags(ctx);
  // Optional: shown only if your backend later returns a Grad-CAM overlay image.
  const heatSrc = result?.gradcam || result?.heatmap || result?.gradcam_overlay || null;
  const heatUrl = heatSrc ? (heatSrc.startsWith("data:") ? heatSrc : `data:image/png;base64,${heatSrc}`) : null;

  const statusLabel = { checking: "Connecting…", online: "Backend online", offline: "Backend offline" }[apiStatus.state];

  /* ──────────────────────────  RENDER  ────────────────────────── */
  return (
    <div className="app" data-theme={theme}>
      <style>{CSS}</style>

      <div className="wrap">
        {/* HEADER */}
        <header className="top no-print">
          <div className="brand">
            <div className="logo"><ScanSearch size={22} aria-hidden="true" /></div>
            <div>
              <h1>Skin lesion risk triage</h1>
              <p>Uncertainty-aware multimodal analysis</p>
            </div>
          </div>
          <div className="top-actions">
            <button className={`status s-${apiStatus.state}`} onClick={() => setShowSettings((s) => !s)} title="Connection settings">
              {apiStatus.state === "offline" ? <WifiOff size={14} /> : apiStatus.state === "checking" ? <Loader2 size={14} className="spin" /> : <Wifi size={14} />}
              <span>{statusLabel}</span>
            </button>
            <button className="icon-btn" onClick={() => setShowSettings((s) => !s)} aria-label="Settings"><Settings size={18} /></button>
            <button className="icon-btn" onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))} aria-label="Toggle theme">
              {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
        </header>

        {/* SETTINGS */}
        {showSettings && (
          <Card
            className="no-print"
            title="Connection settings"
            icon={<Settings size={16} />}
            right={<button className="icon-btn sm" onClick={() => setShowSettings(false)} aria-label="Close"><X size={16} /></button>}
          >
            <label className="field full">
              <span className="lbl">Backend API URL</span>
              <input className="input" value={apiDraft} onChange={(e) => setApiDraft(e.target.value)} placeholder="https://your-backend.example.com" autoCapitalize="off" autoCorrect="off" spellCheck={false} />
              <span className="note">
                Paste your current ngrok / Hugging Face / Render URL. Saved in this browser only.
                {apiStatus.state === "online" && apiStatus.device ? ` Running on: ${apiStatus.device}.` : ""}
              </span>
            </label>
            <div className="btn-row">
              <button className="btn btn-primary" onClick={saveApi}>Save &amp; test</button>
              <button className="btn btn-ghost" onClick={resetApi}>Reset to default</button>
            </div>
          </Card>
        )}

        {/* TABS */}
        <nav className="tabs no-print" role="tablist">
          {[
            ["analyze", "Analyze", <Activity size={15} key="a" />],
            ["model", "Model", <BarChart3 size={15} key="m" />],
            ["learn", "Learn", <BookOpen size={15} key="l" />],
          ].map(([id, label, icon]) => (
            <button key={id} role="tab" aria-selected={tab === id} className={`tab ${tab === id ? "on" : ""}`} onClick={() => setTab(id)}>
              {icon}{label}
            </button>
          ))}
        </nav>

        {/* ═══════════ ANALYZE ═══════════ */}
        {tab === "analyze" && (
          <>
            <div className="layout">
              {/* INPUT COLUMN */}
              <div className="col no-print">
                <Card title="1 · Lesion image" icon={<Upload size={16} />}>
                  <div
                    className={`drop ${previewUrl ? "has" : ""}`}
                    role="button"
                    tabIndex={0}
                    onClick={() => fileInputRef.current?.click()}
                    onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && fileInputRef.current?.click()}
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={onDrop}
                  >
                    {previewUrl ? (
                      <img src={previewUrl} alt="Selected lesion" />
                    ) : (
                      <div className="drop-empty">
                        <Upload size={28} aria-hidden="true" />
                        <p className="drop-t">Drop, tap, or paste an image</p>
                        <p className="drop-s">Dermoscopic JPG, PNG or WEBP · up to {MAX_FILE_MB} MB</p>
                      </div>
                    )}
                    <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={(e) => handleFile(e.target.files?.[0])} />
                  </div>
                  {file && (
                    <div className="row-between" style={{ marginTop: 10 }}>
                      <span className="muted-s ellipsis">{file.name} · {(file.size / 1024).toFixed(0)} KB</span>
                      <button className="link" onClick={resetAll}>Remove</button>
                    </div>
                  )}
                  {previewUrl && (
                    <div style={{ marginTop: 12 }}>
                      <QualityPanel server={null} dims={dims} />
                    </div>
                  )}
                </Card>

                <Card title="2 · Patient details" icon={<Stethoscope size={16} />} right={<Chip tone="teal">Used by model</Chip>}>
                  <div className="grid2">
                    <label className="field">
                      <span className="lbl">Age</span>
                      <input className="input" type="number" inputMode="numeric" min="1" max="120" value={ctx.age} onChange={(e) => setField("age")(e.target.value)} />
                    </label>
                    <Select label="Sex" value={ctx.sex} onChange={setField("sex")} options={[["male", "Male"], ["female", "Female"], ["unknown", "Unknown"]]} />
                    <Select full label="Lesion location" value={ctx.localization} onChange={setField("localization")} options={LOCALIZATIONS} />
                  </div>
                </Card>

                <Card title="3 · Clinical context" icon={<Layers size={16} />} right={<Chip tone="muted">Optional</Chip>}>
                  <div className="grid2">
                    <Select full label="Skin type (Fitzpatrick)" value={ctx.skin_color} onChange={setField("skin_color")} options={SKIN_TYPES} />
                    <Select label="Evolution" value={ctx.evolution} onChange={setField("evolution")} options={EVOLUTION} />
                    <Select label="Symptoms" value={ctx.symptoms} onChange={setField("symptoms")} options={SYMPTOMS} />
                    <Select full label="Previous history" value={ctx.history} onChange={setField("history")} options={HISTORY} />
                  </div>
                </Card>

                {error && (
                  <div className="alert" role="alert">
                    <AlertTriangle size={16} aria-hidden="true" />
                    <span>{error}</span>
                  </div>
                )}

                <button className="btn btn-primary btn-lg" onClick={runAnalysis} disabled={loading}>
                  {loading ? <Loader2 size={17} className="spin" /> : <Activity size={17} />}
                  {loading ? "Analyzing…" : "Analyze image"}
                </button>
                {loading && <p className="note center">The first request can take up to a minute if the backend is waking up.</p>}
              </div>

              {/* RESULT COLUMN */}
              <div className="col" ref={resultRef}>
                {!result && !loading && (
                  <Card className="empty no-print">
                    <ScanSearch size={34} aria-hidden="true" />
                    <h3>Your result will appear here</h3>
                    <p>Add an image and patient details, then press <b>Analyze image</b>. You'll get a risk tier, calibrated uncertainty, an image quality report and a downloadable summary.</p>
                  </Card>
                )}

                {loading && (
                  <Card className="empty">
                    <Loader2 size={34} className="spin" aria-hidden="true" />
                    <h3>Running the model…</h3>
                    <p>Checking image quality, extracting features and estimating uncertainty.</p>
                  </Card>
                )}

                {result && triage && (
                  <>
                    {/* print-only heading */}
                    <div className="print-only">
                      <h2>Skin lesion triage report</h2>
                      <p>{new Date().toLocaleString()}</p>
                    </div>

                    {/* TRIAGE BANNER */}
                    <div className={`banner b-${triage.tone}`}>
                      <triage.Icon size={26} aria-hidden="true" />
                      <div>
                        <p className="banner-k">Risk triage</p>
                        <h2>{result.risk_label || triage.title}</h2>
                        {result.risk_reason && <p className="banner-r">{result.risk_reason}</p>}
                      </div>
                    </div>

                    <Card>
                      <div className="pred">
                        {previewUrl && <img className="pred-img" src={previewUrl} alt="Analyzed lesion" />}
                        <div>
                          <p className="stat-l">Predicted lesion type</p>
                          <h2 className="pred-n">{result.predicted_class_full_name || cls?.name || result.predicted_class}</h2>
                          {cls && <Chip tone={CAT[cls.cat].tone}>{CAT[cls.cat].label}</Chip>}
                        </div>
                      </div>
                      {cls && <p className="blurb">{cls.blurb}</p>}

                      <div className="grid2 stats">
                        <Stat label="Confidence" value={fmtPct(result.confidence)} tone="teal" sub="Probability of top class" />
                        <Stat
                          label="Uncertainty"
                          value={fmtPct(result.uncertainty)}
                          tone={result.uncertainty > THRESH.mid ? "red" : result.uncertainty > THRESH.low ? "amber" : "sage"}
                          sub={result.uncertainty > THRESH.mid ? "Above safety threshold" : result.uncertainty > THRESH.low ? "Moderate" : "Within accepted range"}
                        />
                      </div>

                      <div style={{ marginTop: 18 }}>
                        <p className="lbl">Uncertainty gauge</p>
                        <UncertaintyGauge value={result.uncertainty} />
                        <p className="note" style={{ marginTop: 6 }}>
                          Evidential deep learning estimates how much evidence supports the prediction in a single pass. Higher means the model knows less.
                        </p>
                      </div>
                    </Card>

                    {/* RECOMMENDATION */}
                    <div className={`reco r-${triage.tone}`}>
                      <p className="reco-k">Recommendation</p>
                      <p>{triage.advice}</p>
                    </div>

                    {redFlags.length > 0 && (
                      <div className="reco r-red">
                        <p className="reco-k">Clinical red flags you reported</p>
                        <ul className="issues flat">{redFlags.map((f) => <li key={f}>{f}</li>)}</ul>
                        <p style={{ marginTop: 6 }}>These are not used by the model, but they are reasons to have the lesion examined in person whatever the result above says.</p>
                      </div>
                    )}

                    <Card title="Image quality" icon={<CheckCircle2 size={16} />}>
                      <QualityPanel server={result.image_quality} dims={dims} />
                    </Card>

                    {heatUrl && (
                      <Card title="Model attention (Grad-CAM)" icon={<ScanSearch size={16} />}
                        right={<button className="link" onClick={() => setShowHeat((s) => !s)}>{showHeat ? "Hide overlay" : "Show overlay"}</button>}>
                        <div className="heat">
                          <img src={previewUrl} alt="Original" />
                          {showHeat && <img className="heat-o" src={heatUrl} alt="Heatmap overlay" style={{ opacity: heatOpacity / 100 }} />}
                        </div>
                        {showHeat && (
                          <label className="field" style={{ marginTop: 10 }}>
                            <span className="lbl">Overlay opacity · {heatOpacity}%</span>
                            <input type="range" min="10" max="100" value={heatOpacity} onChange={(e) => setHeatOpacity(Number(e.target.value))} />
                          </label>
                        )}
                      </Card>
                    )}

                    <Card title="Probability by class" icon={<BarChart3 size={16} />}>
                      <ProbBars probs={result.class_probabilities} predicted={result.predicted_class} />
                    </Card>

                    <Card className="summary">
                      <div className="row-between wrap-s">
                        <span>Patient: {ctx.age}y, {ctx.sex}</span>
                        <span>Site: {ctx.localization}</span>
                        {ctx.skin_color && <span>Skin: {ctx.skin_color.replace("type_", "Type ")}</span>}
                      </div>
                    </Card>

                    <div className="btn-row no-print">
                      <button className="btn btn-ghost" onClick={downloadJson}><Download size={16} />Download JSON</button>
                      <button className="btn btn-ghost" onClick={() => window.print()}><Printer size={16} />Print report</button>
                      <button className="btn btn-ghost" onClick={resetAll}><RotateCcw size={16} />New analysis</button>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* HISTORY */}
            {history.length > 0 && (
              <Card className="no-print hist-card" title="This session" icon={<History size={16} />}
                right={<button className="link" onClick={() => setHistory([])}>Clear</button>}>
                <div className="hist">
                  {history.map((h) => {
                    const t = TRIAGE[deriveRiskLevel(h.result)];
                    return (
                      <button key={h.id} className="hist-i" onClick={() => restore(h)}>
                        <img src={h.previewUrl} alt="" />
                        <div>
                          <strong>{CLASSES[h.result.predicted_class]?.name || h.result.predicted_class}</strong>
                          <span><Chip tone={t.tone}>{h.result.risk_label || t.title}</Chip></span>
                          <span className="muted-s">{h.at.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} · {fmtPct(h.result.confidence, 0)} conf.</span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </Card>
            )}
          </>
        )}

        {/* ═══════════ MODEL ═══════════ */}
        {tab === "model" && (
          <div className="stack">
            <Card title="Live model" icon={<Activity size={16} />}
              right={<button className="link" onClick={() => checkApi(apiUrl)}>Refresh</button>}>
              {modelInfo ? (
                <>
                  <div className="grid2 stats" style={{ marginTop: 0 }}>
                    <Stat label="Validation accuracy" value={asPct(modelInfo.val_accuracy)} tone="teal" sub="From checkpoint" />
                    <Stat label="Melanoma recall" value={asPct(modelInfo.mel_recall)} tone="sage" sub="From checkpoint" />
                  </div>
                  <p className="note" style={{ marginTop: 10 }}>Checkpoint: <code>{modelInfo.model_file}</code></p>
                </>
              ) : (
                <p className="note">Not connected. Open Settings and set your backend URL to load live model details.</p>
              )}
            </Card>

            <Card title="Reported results" icon={<BarChart3 size={16} />} right={<Chip tone="muted">From project README</Chip>}>
              <div className="grid2 stats" style={{ marginTop: 0 }}>
                <Stat label="Calibration (ECE)" value={METRICS_README.ece} sub="Lower is better" />
                <Stat label="Error-detection AUROC" value={METRICS_README.auroc} sub="Uncertainty vs mistakes" />
                <Stat label="Inference (single image)" value={METRICS_README.inferenceSingle} sub="Batch size 1, GPU" />
                <Stat label="Inference (batched)" value={METRICS_README.inferenceBatched} sub="Per image, batch of 16" />
              </div>
              <p className="lbl" style={{ marginTop: 18 }}>Average uncertainty: correct vs wrong predictions</p>
              {[["Correct predictions", METRICS_README.uncCorrect, "sage"], ["Wrong predictions", METRICS_README.uncWrong, "red"]].map(([l, v, tone]) => (
                <div key={l} className="prob">
                  <div className="prob-t"><span>{l}</span><strong>{v.toFixed(2)}</strong></div>
                  <div className="track"><div className={`fill f-${tone}`} style={{ width: `${v * 100}%` }} /></div>
                </div>
              ))}
              <p className="note">
                The model is measurably more uncertain when it is wrong. That gap is what makes uncertainty-based triage meaningful.
                Melanoma recall improved from {METRICS_README.melBefore} to {METRICS_README.melAfter} after class-weighted fine-tuning.
              </p>
              <p className="note" style={{ marginTop: 10 }}>
                How these are measured: all figures come from the 1,521-image validation split (no image or lesion overlap with training).
                ECE uses 10 equal-width confidence bins. Uncertainty is u = K / S from the evidential head. AUROC measures how well
                uncertainty separates wrong from correct predictions. Single-image time is the model forward pass only (batch size 1,
                after warm-up); batched time is per image at batch size 16.
              </p>
            </Card>

            <Card title="Pipeline" icon={<Layers size={16} />}>
              <ol className="steps">
                <li><b>Quality pre-screen</b> checks resolution, blur and exposure.</li>
                <li><b>Swin Transformer</b> extracts visual features.</li>
                <li><b>Metadata encoder</b> turns age, sex and site into a vector.</li>
                <li><b>Cross-attention fusion</b> lets patient context attend to image features.</li>
                <li><b>Evidential head</b> outputs class evidence and uncertainty.</li>
                <li><b>Risk triage</b> maps both to Low Risk, Suspicious or Expert Review.</li>
              </ol>
            </Card>

            <Card title="Known limitations" icon={<AlertTriangle size={16} />}>
              <ul className="issues flat">
                <li>Trained on one source (HAM10000). Accuracy can drop on images from other devices (domain shift).</li>
                <li>Skin type, evolution, symptoms and history are schema-ready but not yet trained.</li>
                <li>Academic prototype. Not a diagnostic device.</li>
              </ul>
            </Card>
          </div>
        )}

        {/* ═══════════ LEARN ═══════════ */}
        {tab === "learn" && (
          <div className="stack">
            <Card title="How risk triage works" icon={<ShieldCheck size={16} />}>
              <div className="tiers">
                {[
                  ["sage", "Low Risk", `Benign class and uncertainty ≤ ${THRESH.low}`],
                  ["amber", "Suspicious", `Malignant or pre-malignant class, or uncertainty ${THRESH.low}–${THRESH.mid}`],
                  ["red", "Expert Review", `Uncertainty > ${THRESH.mid}, or the image failed quality checks`],
                ].map(([tone, t, d]) => (
                  <div key={t} className={`tier tier-${tone}`}><strong>{t}</strong><span>{d}</span></div>
                ))}
              </div>
              <p className="note" style={{ marginTop: 10 }}>These thresholds are a tunable policy in <code>core/risk_triage.py</code>.</p>
            </Card>

            <Card title="Lesion types the model knows" icon={<BookOpen size={16} />}>
              <div className="cls-list">
                {Object.entries(CLASSES).map(([k, c]) => (
                  <div key={k} className="cls">
                    <div className="row-between">
                      <strong>{c.name}</strong>
                      <Chip tone={CAT[c.cat].tone}>{CAT[c.cat].label.replace(" class", "")}</Chip>
                    </div>
                    <p>{c.blurb}</p>
                  </div>
                ))}
              </div>
            </Card>

            <Card title="The ABCDE self-check" icon={<Stethoscope size={16} />}>
              <div className="abcde">
                {TIPS_ABCDE.map(([l, t, d]) => (
                  <div key={l} className="ab"><b>{l}</b><div><strong>{t}</strong><span>{d}</span></div></div>
                ))}
              </div>
              <p className="note" style={{ marginTop: 10 }}>Any one of these is a reason to see a dermatologist.</p>
            </Card>

            <Card title="Tips for a good photo" icon={<ScanSearch size={16} />}>
              <ul className="issues flat">
                <li>Use even, natural light without flash glare.</li>
                <li>Keep the lesion in focus and fill most of the frame.</li>
                <li>Dermoscopic images work best. Ordinary phone photos may be less reliable.</li>
                <li>Avoid shadows, hair across the lesion, and heavy filters.</li>
              </ul>
            </Card>
          </div>
        )}

        <footer className="disc">
          <Info size={14} aria-hidden="true" />
          <p>Research demo only. This is not a diagnostic device. Predictions must always be reviewed by a qualified dermatologist.</p>
        </footer>
      </div>
    </div>
  );
}

/* ────────────────────────────────────────────────────────────────
   STYLES
   ──────────────────────────────────────────────────────────────── */
const LIGHT_VARS = `
  --bg:#F6F4EF; --surface:#FFFFFF; --surface2:#F0EDE6; --ink:#1C2B2D; --muted:#5B6664; --faint:#8E938C; --line:#E1DBD0;
  --teal:#2D6A6F; --teal-btn:#1F4C50; --teal-soft:#E3EEEE;
  --sage:#4A7856; --sage-soft:#EAF3DE; --sage-ink:#27500A;
  --amber:#8A6116; --amber-soft:#FAEEDA; --amber-ink:#6B4A0E;
  --red:#B8542E; --red-soft:#FAECE7; --red-ink:#712B13;
  --shadow:0 1px 2px rgba(28,43,45,.05), 0 6px 20px rgba(28,43,45,.05);
`;
const DARK_VARS = `
  --bg:#0F1719; --surface:#172124; --surface2:#1E2A2D; --ink:#E8EEEC; --muted:#9DB0AD; --faint:#6F8582; --line:#2A3A3D;
  --teal:#5FB3B8; --teal-btn:#2D6A6F; --teal-soft:#1D3639;
  --sage:#7DBE8A; --sage-soft:#1B2E21; --sage-ink:#A9DDB3;
  --amber:#E0B25A; --amber-soft:#33291A; --amber-ink:#F0CB85;
  --red:#E58A66; --red-soft:#352119; --red-ink:#F2B7A0;
  --shadow:0 1px 2px rgba(0,0,0,.3), 0 6px 20px rgba(0,0,0,.25);
`;

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600&family=Inter:wght@400;500;600&display=swap');
.app{${LIGHT_VARS}}
.app[data-theme="dark"]{${DARK_VARS}}
.app *{box-sizing:border-box}
body{margin:0}
.app{min-height:100vh;background:var(--bg);color:var(--ink);font-family:'Inter',system-ui,sans-serif;font-size:14.5px;line-height:1.5;transition:background .2s,color .2s}
.wrap{max-width:1040px;margin:0 auto;padding:20px 16px 64px}
.app h1,.app h2,.app h3,.app p{margin:0}
.top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:16px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:12px}
.logo{width:42px;height:42px;border-radius:12px;background:var(--teal-btn);color:#F7F5F2;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.brand h1{font-family:'Source Serif 4',serif;font-weight:600;font-size:21px;line-height:1.2}
.brand p{font-size:12.5px;color:var(--muted)}
.top-actions{display:flex;align-items:center;gap:8px}
.icon-btn{width:38px;height:38px;border-radius:10px;border:1px solid var(--line);background:var(--surface);color:var(--ink);display:flex;align-items:center;justify-content:center;cursor:pointer}
.icon-btn.sm{width:30px;height:30px;border-radius:8px}
.icon-btn:hover{background:var(--surface2)}
.status{display:flex;align-items:center;gap:6px;height:38px;padding:0 12px;border-radius:20px;border:1px solid var(--line);background:var(--surface);font:500 12.5px 'Inter',sans-serif;color:var(--muted);cursor:pointer}
.s-online{color:var(--sage);border-color:color-mix(in srgb,var(--sage) 40%,var(--line))}
.s-offline{color:var(--red);border-color:color-mix(in srgb,var(--red) 40%,var(--line))}
.tabs{display:flex;gap:4px;background:var(--surface2);padding:4px;border-radius:12px;margin:14px 0 16px}
.tab{flex:1;display:flex;align-items:center;justify-content:center;gap:6px;padding:9px 10px;border:0;background:transparent;color:var(--muted);font:500 13.5px 'Inter',sans-serif;border-radius:9px;cursor:pointer}
.tab.on{background:var(--surface);color:var(--ink);box-shadow:var(--shadow)}
.layout{display:grid;grid-template-columns:1fr;gap:16px}
@media(min-width:860px){.layout{grid-template-columns:minmax(0,5fr) minmax(0,6fr);align-items:start}}
.col,.stack{display:flex;flex-direction:column;gap:14px;min-width:0}
.stack{max-width:720px;margin:0 auto}
.hist-card{margin-top:16px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:var(--shadow)}
.card-h{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:14px}
.card-h h3{display:flex;align-items:center;gap:8px;font-size:14.5px;font-weight:600}
.card-h h3 svg{color:var(--teal)}
.lbl{display:block;font-size:12.5px;font-weight:500;color:var(--muted);margin-bottom:6px}
.note{font-size:12.5px;color:var(--faint);line-height:1.55}
.muted-s{font-size:12.5px;color:var(--faint)}
.center{text-align:center}
.ellipsis{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.row-between{display:flex;align-items:center;justify-content:space-between;gap:10px}
.wrap-s{flex-wrap:wrap;font-size:13px;color:var(--muted)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.full{grid-column:1/-1}
.field{display:block;min-width:0}
.input{width:100%;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:var(--surface);color:var(--ink);font:400 14.5px 'Inter',sans-serif;appearance:auto}
.input:focus,.drop:focus-visible,.btn:focus-visible,.tab:focus-visible{outline:2px solid var(--teal);outline-offset:2px}
.app input[type=range]{width:100%;accent-color:var(--teal)}
.drop{border:1.5px dashed var(--line);border-radius:12px;background:var(--surface2);cursor:pointer;overflow:hidden;transition:border-color .15s,background .15s}
.drop:hover{border-color:var(--teal)}
.drop.has{border-style:solid;border-color:var(--teal);background:#00000008}
.drop img{display:block;width:100%;max-height:280px;object-fit:contain}
.drop-empty{padding:34px 16px;text-align:center;color:var(--muted)}
.drop-t{font-weight:500;color:var(--ink);margin-top:8px}
.drop-s{font-size:12.5px;color:var(--faint);margin-top:2px}
.link{border:0;background:none;color:var(--teal);font:500 12.5px 'Inter',sans-serif;cursor:pointer;padding:0}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:10px 16px;border-radius:10px;border:1px solid transparent;font:500 14px 'Inter',sans-serif;cursor:pointer}
.btn:disabled{opacity:.7;cursor:default}
.btn-primary{background:var(--teal-btn);color:#fff}
.btn-primary:hover:not(:disabled){filter:brightness(1.12)}
.btn-ghost{background:var(--surface);border-color:var(--line);color:var(--ink)}
.btn-ghost:hover{background:var(--surface2)}
.btn-lg{width:100%;padding:14px 0;font-size:15.5px;border-radius:12px}
.btn-row{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.alert{display:flex;gap:8px;align-items:flex-start;background:var(--red-soft);color:var(--red-ink);border:1px solid color-mix(in srgb,var(--red) 35%,transparent);border-radius:12px;padding:12px 14px;font-size:13.5px}
.alert svg{flex-shrink:0;margin-top:2px}
.chip{display:inline-block;font-size:11.5px;font-weight:500;padding:3px 10px;border-radius:20px;white-space:nowrap}
.chip-sage{background:var(--sage-soft);color:var(--sage-ink)}
.chip-amber{background:var(--amber-soft);color:var(--amber-ink)}
.chip-red{background:var(--red-soft);color:var(--red-ink)}
.chip-teal{background:var(--teal-soft);color:var(--teal)}
.chip-muted{background:var(--surface2);color:var(--muted)}
.empty{text-align:center;padding:44px 22px;color:var(--muted)}
.empty svg{color:var(--teal);margin-bottom:10px}
.empty h3{font-family:'Source Serif 4',serif;font-size:18px;color:var(--ink);margin-bottom:6px}
.empty p{font-size:13.5px;max-width:380px;margin:0 auto}
.banner{display:flex;gap:14px;align-items:center;padding:18px;border-radius:16px;border:1px solid}
.banner svg{flex-shrink:0}
.banner h2{font-family:'Source Serif 4',serif;font-size:22px;font-weight:600;line-height:1.25}
.banner-k{font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;font-weight:600;opacity:.75}
.banner-r{font-size:13.5px;margin-top:3px}
.b-sage{background:var(--sage-soft);color:var(--sage-ink);border-color:color-mix(in srgb,var(--sage) 40%,transparent)}
.b-amber{background:var(--amber-soft);color:var(--amber-ink);border-color:color-mix(in srgb,var(--amber) 40%,transparent)}
.b-red{background:var(--red-soft);color:var(--red-ink);border-color:color-mix(in srgb,var(--red) 40%,transparent)}
.pred{display:flex;gap:14px;align-items:center}
.pred-img{width:72px;height:72px;object-fit:cover;border-radius:12px;border:1px solid var(--line);flex-shrink:0}
.pred-n{font-family:'Source Serif 4',serif;font-size:21px;font-weight:600;line-height:1.25;margin:3px 0 8px}
.blurb{margin-top:12px;font-size:13.5px;color:var(--muted)}
.stats{margin-top:16px}
.stat{background:var(--surface2);border-radius:12px;padding:12px 14px}
.stat-l{font-size:12px;color:var(--muted)}
.stat-v{font-family:'Source Serif 4',serif;font-size:26px;font-weight:600;line-height:1.2;margin-top:2px}
.stat-s{font-size:12px;color:var(--faint);margin-top:1px}
.t-teal{color:var(--teal)}.t-sage{color:var(--sage)}.t-amber{color:var(--amber)}.t-red{color:var(--red)}
.gauge{position:relative;display:flex;height:12px;border-radius:6px;margin-top:10px}
.zone{display:block;height:100%}
.zone:first-child{border-radius:6px 0 0 6px}.zone:nth-child(3){border-radius:0 6px 6px 0}
.z-sage{background:var(--sage)}.z-amber{background:var(--amber)}.z-red{background:var(--red)}
.needle{position:absolute;top:-5px;width:4px;height:22px;margin-left:-2px;border-radius:2px;background:var(--ink);box-shadow:0 0 0 2px var(--surface);transition:left .5s}
.gauge-l{position:relative;height:16px;margin:10px 0 4px;font-size:11px;color:var(--faint)}
.gauge-l span{position:absolute;transform:translateX(-50%)}
.gauge-l span:first-child,.gauge-l span:last-child{transform:none}
.reco{border-radius:14px;padding:14px 16px;font-size:13.5px;border:1px solid}
.reco-k{font-size:12px;font-weight:600;margin-bottom:3px}
.r-sage{background:var(--sage-soft);color:var(--sage-ink);border-color:color-mix(in srgb,var(--sage) 35%,transparent)}
.r-amber{background:var(--amber-soft);color:var(--amber-ink);border-color:color-mix(in srgb,var(--amber) 35%,transparent)}
.r-red{background:var(--red-soft);color:var(--red-ink);border-color:color-mix(in srgb,var(--red) 35%,transparent)}
.issues{margin:10px 0 0;padding-left:18px;font-size:13.5px;color:var(--muted)}
.issues li{margin-bottom:4px}
.issues.flat{margin-top:4px}
.r-red .issues{color:inherit}
.prob{margin-bottom:11px}
.prob-t{display:flex;justify-content:space-between;gap:10px;font-size:13px;margin-bottom:4px}
.prob-t span{color:var(--muted)}
.dim{font-weight:400;color:var(--muted)}
.rank{display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:50%;background:var(--surface2);font-style:normal;font-size:10px;font-weight:600;margin-right:6px;color:var(--muted)}
.track{height:7px;border-radius:4px;background:var(--surface2);overflow:hidden}
.fill{height:100%;border-radius:4px;transition:width .6s}
.f-sage{background:var(--sage)}.f-amber{background:var(--amber)}.f-red{background:var(--red)}.f-teal{background:var(--teal)}.f-muted{background:var(--faint)}
.fill.ghost{opacity:.45}
.heat{position:relative;border-radius:12px;overflow:hidden;background:#000}
.heat img{display:block;width:100%}
.heat .heat-o{position:absolute;inset:0;height:100%;object-fit:cover}
.hist{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}
.hist-i{display:flex;gap:10px;align-items:center;text-align:left;padding:10px;border:1px solid var(--line);border-radius:12px;background:var(--surface);color:var(--ink);cursor:pointer;font:inherit}
.hist-i:hover{background:var(--surface2)}
.hist-i img{width:52px;height:52px;border-radius:10px;object-fit:cover;flex-shrink:0}
.hist-i div{display:flex;flex-direction:column;gap:3px;min-width:0}
.hist-i strong{font-size:13px;line-height:1.3}
.steps{margin:0;padding-left:20px;font-size:13.5px;color:var(--muted)}
.steps li{margin-bottom:7px}.steps b{color:var(--ink)}
.tiers{display:flex;flex-direction:column;gap:8px}
.tier{display:flex;flex-direction:column;gap:2px;padding:11px 14px;border-radius:12px;font-size:13px;border-left:4px solid}
.tier-sage{background:var(--sage-soft);color:var(--sage-ink);border-color:var(--sage)}
.tier-amber{background:var(--amber-soft);color:var(--amber-ink);border-color:var(--amber)}
.tier-red{background:var(--red-soft);color:var(--red-ink);border-color:var(--red)}
.cls-list{display:flex;flex-direction:column;gap:10px}
.cls{padding:12px 14px;border:1px solid var(--line);border-radius:12px}
.cls p{font-size:13px;color:var(--muted);margin-top:4px}
.abcde{display:flex;flex-direction:column;gap:8px}
.ab{display:flex;gap:12px;align-items:center}
.ab b{width:34px;height:34px;border-radius:10px;background:var(--teal-soft);color:var(--teal);display:flex;align-items:center;justify-content:center;font-family:'Source Serif 4',serif;font-size:17px;flex-shrink:0}
.ab div{display:flex;flex-direction:column;font-size:13px;color:var(--muted)}
.ab strong{color:var(--ink)}
.app code{font-size:12px;background:var(--surface2);padding:1px 6px;border-radius:5px}
.summary{padding:12px 16px}
.disc{display:flex;gap:8px;align-items:flex-start;margin-top:28px;font-size:12px;color:var(--faint);line-height:1.6}
.disc svg{flex-shrink:0;margin-top:3px}
.print-only{display:none}
.spin{animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
@media(max-width:420px){.status span{display:none}.status{padding:0 11px}.pred-n{font-size:19px}}
@media print{
  .app,.app[data-theme="dark"]{${LIGHT_VARS}}
  .no-print{display:none!important}
  .print-only{display:block;margin-bottom:10px}
  .layout{display:block}
  .app{background:#fff}
  .card,.banner,.reco{box-shadow:none;break-inside:avoid;margin-bottom:10px}
}
@media(prefers-reduced-motion:reduce){.app *{transition:none!important}.spin{animation:none!important}}
`;
