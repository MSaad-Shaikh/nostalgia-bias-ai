'use client'
import { ChangeEvent, useEffect, useRef, useState, useCallback, useMemo, useReducer, memo } from "react";
import Markdown from "react-markdown";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

type SK = "Age Group" | "Gender" | "Ethnicity";
const LABELS: SK[] = ["Age Group", "Gender", "Ethnicity"];
interface GI { group: string; rate: number; count: number; percentage: number; sufficient?: boolean; }
interface Sev { tier: string; color: string; }
interface Conf { level: string; score: number; }
interface CatM {
  di_raw: number; di_smoothed: number; spd: number; eo: number;
  severity: Sev; confidence: Conf; bias_detected: boolean; di_unstable: boolean;
  bias_class: string; chi2: number | null; p_value: number | null;
  breakdown: GI[]; synthetic_flags: string[];
}
interface IX {
  label: string; disparate_impact: number; spd: number; severity: Sev;
  worst_group: string; best_group: string; breakdown: GI[];
}
interface Feat {
  column: string; correlation: number; abs_correlation: number;
  direction: string; strength: string; is_demographic: boolean;
  is_artifact: boolean; note: string;
}
interface Risk { level: string; color: string; worst: number; avg: number; }
interface Proxy { proxy_column: string; demographic_column: string; strength: number; risk: string; description: string; }
interface KeyInsight { icon: string; type: string; text: string; detail: string; }
interface ChatMsg { role: "user" | "ai"; text: string; }
interface Res {
  rows: number; columns: string[]; target_column: string; demographics_detected: number;
  metrics: Record<SK, CatM>; risk: Risk; synthetic_flags: string[]; simpsons_paradox: string[];
  intersectional: IX[]; features: Feat[]; confounder_strength: number; proxy_variables: Proxy[];
  key_insights: KeyInsight[]; data_preview: Record<string, unknown>[]; ai_report: string;
}

const rC = (l: string) =>
  l === "Critical" ? "#ef4444" : l === "High" ? "#f97316" : l === "Medium" ? "#f59e0b" : "#10b981";
const sC = (s: number) => (s >= 0.8 ? "#10b981" : s >= 0.6 ? "#f59e0b" : "#ef4444");
const bC = (r: number) => (r >= 0.8 ? "#10b981" : r >= 0.5 ? "#f59e0b" : "#ef4444");
const sevC = (c: string) => (c === "green" ? "#10b981" : c === "yellow" ? "#f59e0b" : "#ef4444");
const cC = (l: string) => (l === "HIGH" ? "#10b981" : l === "MEDIUM" ? "#f59e0b" : "#9095a0");
const classC = (c: string) =>
  c === "Structural Bias" ? "#ef4444"
  : c === "Conditional Bias" ? "#f97316"
  : c === "Confounded Bias" ? "#f59e0b"
  : c === "Data Artifact" ? "#7c5cff"
  : "#10b981";

/** Memoized progress ring — flat */
const Ring = memo(function Ring({ score, size = 86, duration = 1400 }: { score: number; size?: number; duration?: number }) {
  const r = 30, c = 2 * Math.PI * r;
  const target = Math.min(Math.max(score, 0), 1);
  const wrapRef = useRef<SVGSVGElement | null>(null);
  const [val, setVal] = useState(0);
  const startedRef = useRef(false);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") { setVal(target); return; }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting && !startedRef.current) {
          startedRef.current = true;
          io.unobserve(e.target);
          const start = performance.now();
          const tick = (t: number) => {
            const p = Math.min(1, (t - start) / duration);
            // easeOutCubic — speedometer-style decel
            const eased = 1 - Math.pow(1 - p, 3);
            setVal(target * eased);
            if (p < 1) requestAnimationFrame(tick);
          };
          requestAnimationFrame(tick);
        }
      });
    }, { threshold: 0.35 });
    io.observe(el);
    return () => io.disconnect();
  }, [target, duration]);
  const off = c - val * c;
  const col = sC(val);
  return (
    <svg ref={wrapRef} width={size} height={size} viewBox="0 0 76 76" className="animate-ring-fill" style={{ filter: `drop-shadow(0 0 6px ${col}33)` }}>
      <circle cx="38" cy="38" r={r} fill="none" stroke="#ececef" strokeWidth="5" />
      <circle
        cx="38" cy="38" r={r} fill="none" stroke={col} strokeWidth="5"
        strokeLinecap="round" strokeDasharray={c} strokeDashoffset={off}
        transform="rotate(-90 38 38)"
      />
      <text x="38" y="36" textAnchor="middle" fill={col} fontSize="14" fontWeight="700">
        {val.toFixed(2)}
      </text>
      <text x="38" y="47" textAnchor="middle" fill="#9095a0" fontSize="7">
        {target >= 0.8 ? "FAIR" : target >= 0.6 ? "MODERATE" : "HIGH RISK"}
      </text>
    </svg>
  );
});

/** Tooltip — light, flat */
const CTip = memo(function CTip({ active, payload }: { active?: boolean; payload?: Array<{ payload: GI }> }) {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-lg border border-[var(--border-subtle)] bg-white px-3 py-2 text-xs shadow-md">
      <p className="font-semibold text-[var(--text-primary)]">{d.group}</p>
      <p className="text-[var(--text-secondary)]">{d.percentage}% (n={d.count})</p>
    </div>
  );
});

/** Surface — flat replacement for the old TiltCard. Same API. */
const TiltCard = memo(function TiltCard({
  children,
  className = "",
  intensity: _intensity = 0,
  ...rest
}: {
  children: React.ReactNode;
  className?: string;
  intensity?: number;
} & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={`tilt-card ${className}`} {...rest}>
      {children}
    </div>
  );
});

/**
 * Reveal — IntersectionObserver scroll-trigger.
 * Compositor-only: opacity + translateY/scale.
 * Once revealed, .in-view persists (graphs stay visible).
 */
const Reveal = memo(function Reveal({
  children,
  className = "",
  delay = 0,
  variant = "up",
}: {
  children: React.ReactNode;
  className?: string;
  delay?: number;
  variant?: "up" | "scale" | "left";
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") { setShown(true); return; }
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) { setShown(true); io.unobserve(e.target); }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -80px 0px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);
  const base = variant === "scale" ? "reveal-scale" : variant === "left" ? "reveal-left" : "reveal";
  return (
    <div
      ref={ref}
      className={`${base} ${shown ? "in-view" : ""} ${className}`}
      style={{ transitionDelay: shown ? `${delay}ms` : "0ms" }}
    >
      {children}
    </div>
  );
});

/** CountUp — animates number once visible (transform/opacity-free, just text update) */
const CountUp = memo(function CountUp({ value, duration = 900 }: { value: number; duration?: number }) {
  const [n, setN] = useState(0);
  const ref = useRef<HTMLSpanElement | null>(null);
  const startedRef = useRef(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") { setN(value); return; }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting && !startedRef.current) {
          startedRef.current = true;
          io.unobserve(e.target);
          const start = performance.now();
          const from = 0, to = value;
          const tick = (t: number) => {
            const p = Math.min(1, (t - start) / duration);
            const eased = 1 - Math.pow(1 - p, 3);
            setN(Math.round(from + (to - from) * eased));
            if (p < 1) requestAnimationFrame(tick);
          };
          requestAnimationFrame(tick);
        }
      });
    }, { threshold: 0.4 });
    io.observe(el);
    return () => io.disconnect();
  }, [value, duration]);
  return <span ref={ref}>{n.toLocaleString()}</span>;
});

/**
 * ScrollBarChart — renders a horizontal Recharts BarChart whose bars
 * animate from 0% → real percentage when scrolled into view.
 * We hold the data at percentage:0 until visible, then swap to real values
 * so Recharts' built-in animation runs from 0 like a speedometer fill.
 */
const ScrollBarChart = memo(function ScrollBarChart({
  data,
  duration = 1400,
}: {
  data: GI[];
  duration?: number;
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [active, setActive] = useState(false);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") { setActive(true); return; }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          setActive(true);
          io.unobserve(e.target);
        }
      });
    }, { threshold: 0.25 });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  const display = active
    ? data
    : data.map((d) => ({ ...d, percentage: 0 }));
  return (
    <div ref={wrapRef}>
      <ResponsiveContainer width="100%" height={data.length * 42 + 10}>
        <BarChart data={display} layout="vertical" margin={{ top: 0, right: 10, bottom: 0, left: 0 }}>
          <XAxis type="number" domain={[0, 100]} tick={{ fill: "#9095a0", fontSize: 10 }} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="group" tick={{ fill: "#52525b", fontSize: 11 }} axisLine={false} tickLine={false} width={90} />
          <Tooltip content={<CTip />} cursor={{ fill: "#f5f5f7" }} />
          <Bar
            dataKey="percentage"
            radius={[0, 4, 4, 0]}
            barSize={16}
            isAnimationActive
            animationDuration={duration}
            animationBegin={120}
            animationEasing="ease-out"
          >
            {display.map((d, i) => <Cell key={i} fill={bC(d.rate)} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
});

export default function BiasInspector() {
  // ─── State (UNCHANGED) ─────────────────────────────────────────
  type Action =
    | { type: "SET_FILE"; file: File; name: string }
    | { type: "SET_COLS"; cols: string[]; target: string }
    | { type: "SET_TARGET"; target: string }
    | { type: "SET_DRAG"; dragging: boolean }
    | { type: "ANALYZE_START" }
    | { type: "ANALYZE_OK"; payload: Res }
    | { type: "ANALYZE_FAIL"; error: string }
    | { type: "SET_ERROR"; error: string | null }
    | { type: "HEADER_FAIL"; error: string };

  interface S {
    file: File | null; fn: string | null; cols: string[]; tgt: string;
    drag: boolean; ph: "idle" | "analyzing" | "complete"; err: string | null; res: Res | null;
  }

  const [s, dispatch] = useReducer(
    (prev: S, a: Action): S => {
      switch (a.type) {
        case "SET_FILE":      return { ...prev, file: a.file, fn: a.name, err: null, ph: "idle", res: null };
        case "SET_COLS":      return { ...prev, cols: a.cols, tgt: a.target };
        case "SET_TARGET":    return { ...prev, tgt: a.target };
        case "SET_DRAG":      return { ...prev, drag: a.dragging };
        case "ANALYZE_START": return { ...prev, err: null, ph: "analyzing", res: null };
        case "ANALYZE_OK":
          return { ...prev, ph: "complete", res: a.payload, cols: a.payload.columns || prev.cols, tgt: a.payload.target_column || prev.tgt };
        case "ANALYZE_FAIL":  return { ...prev, ph: "idle", err: a.error };
        case "SET_ERROR":     return { ...prev, err: a.error };
        case "HEADER_FAIL":   return { ...prev, cols: [], tgt: "", err: a.error };
        default:              return prev;
      }
    },
    { file: null, fn: null, cols: [], tgt: "", drag: false, ph: "idle", err: null, res: null }
  );

  const [chatMsgs, setChatMsgs] = useState<ChatMsg[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [highlightedCard, setHighlightedCard] = useState<SK | null>(null);

  const ref = useRef<HTMLInputElement | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  const analyzingRef = useRef(false);
  const chatLockRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);

  // ─── Handlers (UNCHANGED) ──────────────────────────────────────
  // ─── Feature 3: Highlight referenced metric card ─────────────
  const detectHighlight = useCallback((q: string) => {
    const lower = q.toLowerCase();
    const map: [SK, string[]][] = [
      ["Ethnicity", ["ethnicity", "race", "ethnic"]],
      ["Gender", ["gender", "sex", "male", "female"]],
      ["Age Group", ["age", "age group", "old", "young"]],
    ];
    for (const [label, keywords] of map) {
      if (keywords.some((kw) => lower.includes(kw))) {
        setHighlightedCard(label);
        setTimeout(() => setHighlightedCard(null), 3000);
        return;
      }
    }
  }, []);

  const sendChat = useCallback(() => {
    const q = chatInput.trim();
    if (!q || chatLockRef.current) return;
    chatLockRef.current = true;
    setChatMsgs((p) => [...p, { role: "user", text: q }]);
    setChatInput("");
    setChatLoading(true);

    // Feature 1: Instant placeholder response
    const placeholderIdx = { current: -1 };
    setChatMsgs((p) => {
      placeholderIdx.current = p.length;
      return [...p, { role: "ai", text: "⏳ Analyzing your dataset..." }];
    });

    // Feature 3: Highlight referenced card
    detectHighlight(q);

    // Feature 2: Send metrics context along with question
    const payload: Record<string, unknown> = { question: q };
    if (s.res) {
      payload.metrics = s.res.metrics;
      payload.risk = s.res.risk;
      payload.key_insights = s.res.key_insights;
    }

    fetch("http://127.0.0.1:5000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => r.json())
      .then((d) => {
        // Replace placeholder with real response
        setChatMsgs((p) => {
          const updated = [...p];
          const idx = placeholderIdx.current;
          if (idx >= 0 && idx < updated.length) {
            updated[idx] = { role: "ai", text: d.answer || "No response." };
          } else {
            updated.push({ role: "ai", text: d.answer || "No response." });
          }
          return updated;
        });
        setChatLoading(false);
        chatLockRef.current = false;
        setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
      })
      .catch(() => {
        setChatMsgs((p) => {
          const updated = [...p];
          const idx = placeholderIdx.current;
          if (idx >= 0 && idx < updated.length) {
            updated[idx] = { role: "ai", text: "Failed to get response." };
          }
          return updated;
        });
        setChatLoading(false);
        chatLockRef.current = false;
      });
  }, [chatInput, s.res, detectHighlight]);

  const loadH = useCallback(async (f: File) => {
    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    const fd = new FormData();
    fd.append("file", f);
    const r = await fetch("http://127.0.0.1:5000/api/headers", { method: "POST", body: fd, signal: ctrl.signal });
    if (!r.ok) throw new Error(`${r.status}`);
    const d = await r.json();
    const c: string[] = Array.isArray(d?.columns) ? d.columns.map(String) : [];
    if (!c.length) throw new Error("No columns.");
    dispatch({ type: "SET_COLS", cols: c, target: c[c.length - 1] });
  }, []);

  const handleF = useCallback(async (f: File) => {
    dispatch({ type: "SET_FILE", file: f, name: f.name });
    try { await loadH(f); }
    catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      dispatch({ type: "HEADER_FAIL", error: e instanceof Error ? e.message : "Failed." });
    }
  }, [loadH]);

  const click = useCallback(() => { ref.current?.click(); }, []);

  const analyze = useCallback(() => {
    if (!s.file) { dispatch({ type: "SET_ERROR", error: "Upload a CSV first." }); return; }
    if (analyzingRef.current) return;
    analyzingRef.current = true;
    dispatch({ type: "ANALYZE_START" });
    const fd = new FormData();
    fd.append("file", s.file);
    if (s.tgt) fd.append("target_column", s.tgt);
    fetch("http://127.0.0.1:5000/api/analyze", { method: "POST", body: fd })
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((d: Res) => {
        if (!d.metrics) throw new Error("No metrics.");
        dispatch({ type: "ANALYZE_OK", payload: d });
        analyzingRef.current = false;
      })
      .catch((e) => {
        dispatch({ type: "ANALYZE_FAIL", error: e instanceof Error ? e.message : "Failed." });
        analyzingRef.current = false;
      });
  }, [s.file, s.tgt]);

  const statsCards = useMemo(() => {
    if (!s.res) return [];
    return [
      { l: "Total Rows", v: s.res.rows, i: "📊", numeric: true },
      { l: "Columns", v: s.res.columns.length, i: "📋", numeric: true },
      { l: "Demographics", v: s.res.demographics_detected, i: "👥", numeric: true },
      { l: "Risk Level", v: s.res.risk.level, numeric: false,
        i: s.res.risk.level === "Critical" ? "🔴" : s.res.risk.level === "High" ? "🟠" : s.res.risk.level === "Medium" ? "🟡" : "🟢" },
    ];
  }, [s.res]);

  return (
    <main className="bias-app-shell relative min-h-screen pb-20">
      <div className="relative mx-auto max-w-5xl px-5 pt-16">
        {/* HERO */}
        <header className="mb-14 text-center animate-fade-in-up">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-[var(--border-subtle)] bg-white px-3.5 py-1.5 text-[11px] font-medium text-[var(--text-secondary)] shadow-[0_1px_2px_rgba(16,24,40,0.04)]">
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[var(--accent-blue)] opacity-60" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-[var(--accent-blue)]" />
            </span>
            Powered by Gemini AI
          </div>
          <h1 className="text-4xl sm:text-6xl font-semibold tracking-tight text-[var(--text-primary)] leading-[1.05]">
            AI Bias <span className="gradient-text">Inspector</span>
          </h1>
          <p className="mt-5 text-base sm:text-lg text-[var(--text-secondary)] max-w-xl mx-auto leading-relaxed">
            Measure, flag, and fix harmful bias before your AI impacts real people.
          </p>
        </header>

        {/* DROPZONE */}
        <section className="mb-6 animate-fade-in-up stagger-1 no-print">
          <div
            role="button"
            tabIndex={0}
            onClick={click}
            onDragOver={(e) => { e.preventDefault(); dispatch({ type: "SET_DRAG", dragging: true }); }}
            onDragLeave={(e) => { e.preventDefault(); dispatch({ type: "SET_DRAG", dragging: false }); }}
            onDrop={(e) => {
              e.preventDefault();
              dispatch({ type: "SET_DRAG", dragging: false });
              const f = e.dataTransfer.files[0];
              if (f?.name.toLowerCase().endsWith(".csv")) void handleF(f);
            }}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); click(); } }}
            className={`drop-zone relative cursor-pointer p-10 text-center ${s.drag ? "dragging" : ""}`}
          >
            <input
              ref={ref}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={(e: ChangeEvent<HTMLInputElement>) => {
                const f = e.target.files?.[0];
                if (f) void handleF(f);
              }}
              onClick={(e) => e.stopPropagation()}
            />
            <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-[var(--bg-subtle)] animate-float">
              <svg className="h-6 w-6 text-[var(--accent-blue)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <p className="font-medium text-[var(--text-primary)] text-base">
              {s.fn ? <><span className="text-[var(--accent-blue)]">{s.fn}</span> selected</> : "Drop your CSV dataset here"}
            </p>
            <p className="mt-1 text-sm text-[var(--text-muted)]">or click to browse files</p>
          </div>
        </section>

        {/* TARGET + ANALYZE */}
        <div className="mb-10 grid grid-cols-1 gap-3 sm:grid-cols-[1fr_auto] animate-fade-in-up stagger-2 no-print">
          <div>
            <label htmlFor="tc" className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
              Target Outcome Column
            </label>
            <select
              id="tc"
              value={s.tgt}
              onChange={(e) => dispatch({ type: "SET_TARGET", target: e.target.value })}
              className="w-full rounded-lg border border-[var(--border-subtle)] bg-white px-3 py-2.5 text-sm text-[var(--text-primary)] focus:border-[var(--accent-blue)] focus:outline-none focus:ring-2 focus:ring-[var(--accent-blue)]/20 transition"
            >
              {s.cols.length === 0
                ? <option value="" disabled>Upload CSV…</option>
                : s.cols.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div className="flex items-end">
            <button
              type="button"
              onClick={analyze}
              disabled={s.ph === "analyzing"}
              className="gradient-btn flex h-[42px] items-center justify-center gap-2 px-7 text-sm font-medium"
            >
              {s.ph === "analyzing" ? (
                <>
                  <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Analyzing…
                </>
              ) : s.ph === "complete" ? <>✓ Complete</> : <>Analyze Data →</>}
            </button>
          </div>
        </div>

        {s.err && (
          <p className="mb-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {s.err}
          </p>
        )}

        {(s.ph === "analyzing" || s.res) && (
          <div className="animate-fade-in-up stagger-3">
            {/* Key Insights */}
            {s.res?.key_insights?.length ? (
              <Reveal>
                <section className="mb-8">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-3">Key Insights</h3>
                  <div className="space-y-1.5">
                    {s.res!.key_insights.map((ki, i) => {
                      const accent =
                        ki.type === "critical" ? "border-l-red-500"
                        : ki.type === "warning" ? "border-l-amber-500"
                        : ki.type === "proxy" ? "border-l-violet-500"
                        : "border-l-blue-500";
                      return (
                        <Reveal key={i} delay={i * 60} variant="left">
                          <div className={`flex items-start gap-3 border-l-2 ${accent} bg-white px-3 py-2 rounded-r border border-[var(--border-subtle)] border-l-2 transition-shadow hover:shadow-md`}>
                            <span className="text-sm mt-0.5 shrink-0">{ki.icon}</span>
                            <div>
                              <p className="text-sm font-medium text-[var(--text-primary)]">{ki.text}</p>
                              <p className="text-xs text-[var(--text-muted)] mt-0.5">{ki.detail}</p>
                            </div>
                          </div>
                        </Reveal>
                      );
                    })}
                  </div>
                </section>
              </Reveal>
            ) : null}

            {/* Stats — interactive cards with count-up */}
            {statsCards.length > 0 && (
              <div className="mb-12 grid grid-cols-2 gap-3 sm:grid-cols-4">
                {statsCards.map((card, idx) => (
                  <Reveal key={card.l} delay={idx * 80} variant="scale">
                    <div className="stat-card p-4 h-full">
                      <div className="relative z-10">
                        <div className="flex items-center justify-between mb-1">
                          <div className="text-xs uppercase tracking-wider text-[var(--text-muted)]">{card.l}</div>
                          <span className="text-base opacity-80">{card.i}</span>
                        </div>
                        <div
                          className="text-2xl font-semibold"
                          style={card.l === "Risk Level" ? { color: rC(String(card.v)) } : { color: "var(--text-primary)" }}
                        >
                          {card.numeric ? <CountUp value={Number(card.v)} /> : card.v}
                        </div>
                      </div>
                    </div>
                  </Reveal>
                ))}
              </div>
            )}

            {/* Proxy badges */}
            {s.res?.proxy_variables?.length ? (
              <Reveal>
                <section className="mb-6">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-2">Proxy Variable Detection</h3>
                  <div className="flex flex-wrap gap-1.5">
                    {s.res!.proxy_variables.map((p, i) => (
                      <span
                        key={i}
                        className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-medium transition-transform hover:scale-105 ${
                          p.risk === "HIGH" ? "bg-red-50 text-red-700 border border-red-200" : "bg-amber-50 text-amber-700 border border-amber-200"
                        }`}
                      >
                        {p.proxy_column} → {p.demographic_column}
                        <span className="opacity-60">({(p.strength * 100).toFixed(0)}%)</span>
                      </span>
                    ))}
                  </div>
                </section>
              </Reveal>
            ) : null}

            {s.res?.synthetic_flags?.length ? (
              <Reveal>
                <section className="mb-6">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-2">⚠️ Synthetic Pattern Risk</h3>
                  <ul className="space-y-1">
                    {s.res!.synthetic_flags.map((f, i) => (
                      <li key={i} className="text-sm text-[var(--text-secondary)] pl-3 border-l border-amber-300">{f}</li>
                    ))}
                  </ul>
                </section>
              </Reveal>
            ) : null}

            {s.res?.simpsons_paradox?.length ? (
              <Reveal>
                <section className="mb-8">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-2">Simpson&apos;s Paradox Detected</h3>
                  <ul className="space-y-1">
                    {s.res!.simpsons_paradox.map((f, i) => (
                      <li key={i} className="text-sm text-[var(--text-secondary)] pl-3 border-l border-violet-300">{f}</li>
                    ))}
                  </ul>
                </section>
              </Reveal>
            ) : null}

            {/* Multi-metric */}
            <h2 className="mb-4 text-lg font-semibold text-[var(--text-primary)] section-heading">Multi-Metric Bias Analysis</h2>
            <div className="mb-12 grid grid-cols-1 gap-4 md:grid-cols-3">
              {s.ph === "analyzing"
                ? [0, 1, 2].map((i) => <div key={i} className="tilt-card p-5 h-72 animate-shimmer" />)
                : s.res &&
                  LABELS.map((label, idx) => {
                    const m = s.res!.metrics[label];
                    const isGlowing = highlightedCard === label;
                    return (
                      <Reveal key={label} delay={idx * 100} variant="scale">
                        <TiltCard className={`p-5 ${isGlowing ? "card-glow" : ""}`} data-metric={label}>
                          <div className="flex items-start justify-between mb-3">
                            <h3 className="text-sm font-semibold text-[var(--text-primary)]">{label}</h3>
                            <span
                              className="text-[10px] font-semibold px-2 py-0.5 rounded-full"
                              style={{ backgroundColor: `${classC(m.bias_class)}15`, color: classC(m.bias_class) }}
                            >
                              {m.bias_class}
                            </span>
                          </div>
                          <div className="flex justify-center mb-3"><Ring score={m.di_smoothed} /></div>
                          <div className="text-center mb-3">
                            <span
                              className="text-[10px] font-semibold px-2.5 py-1 rounded-full"
                              style={{ backgroundColor: `${sevC(m.severity.color)}15`, color: sevC(m.severity.color) }}
                            >
                              {m.severity.tier}
                            </span>
                          </div>
                          {m.di_unstable && (
                            <p className="text-[10px] text-amber-600 text-center mb-2">⚠ Small sample — DI unstable</p>
                          )}
                          <div className="space-y-1.5 text-[11px]">
                            <div className="flex justify-between">
                              <span className="text-[var(--text-muted)]">DI (raw)</span>
                              <span className="font-mono font-medium" style={{ color: sC(m.di_raw) }}>{m.di_raw.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-[var(--text-muted)]">DI (smoothed)</span>
                              <span className="font-mono font-medium" style={{ color: sC(m.di_smoothed) }}>{m.di_smoothed.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-[var(--text-muted)]">Stat. Parity Diff</span>
                              <span className="font-mono font-medium" style={{ color: m.spd > 0.1 ? "#ef4444" : "#10b981" }}>{m.spd.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-[var(--text-muted)]">Equal Opp. Proxy</span>
                              <span className="font-mono font-medium" style={{ color: m.eo > 0.1 ? "#ef4444" : "#10b981" }}>{m.eo.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between pt-1.5 border-t border-[var(--border-subtle)]">
                              <span className="text-[var(--text-muted)]">p-value</span>
                              <span className="font-mono font-medium" style={{ color: m.p_value !== null && m.p_value < 0.05 ? "#10b981" : "#9095a0" }}>
                                {m.p_value !== null ? (m.p_value < 0.001 ? "<0.001" : m.p_value.toFixed(3)) : "N/A"}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-[var(--text-muted)]">Confidence</span>
                              <span className="font-medium" style={{ color: cC(m.confidence.level) }}>{m.confidence.level}</span>
                            </div>
                          </div>
                        </TiltCard>
                      </Reveal>
                    );
                  })}
            </div>

            {/* Bar charts — scroll-revealed graphs (scale + lift) */}
            {/* Feature 4: Loading skeleton for graphs */}
            {s.ph === "analyzing" && (
              <>
                <h2 className="mb-4 text-lg font-semibold text-[var(--text-primary)] section-heading">Per-Group Approval Rates</h2>
                <div className="mb-12 grid grid-cols-1 gap-4 lg:grid-cols-3">
                  {[0, 1, 2].map((i) => (
                    <div key={i} className="tilt-card p-5">
                      <div className="animate-shimmer h-3 w-24 mb-4 rounded" />
                      <div className="space-y-3">
                        {[75, 60, 90, 45].map((w, j) => (
                          <div key={j} className="flex items-center gap-3">
                            <div className="animate-shimmer h-3 w-16 rounded" />
                            <div className="flex-1 animate-shimmer h-4 rounded" style={{ maxWidth: `${w}%` }} />
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
            {s.res?.metrics && (
              <>
                <h2 className="mb-4 text-lg font-semibold text-[var(--text-primary)] section-heading">Per-Group Approval Rates</h2>
                <div className="mb-12 grid grid-cols-1 gap-4 lg:grid-cols-3">
                  {LABELS.map((label, idx) => {
                    const data = s.res!.metrics[label]?.breakdown;
                    if (!data?.length)
                      return (
                        <Reveal key={label} delay={idx * 120} variant="scale">
                          <TiltCard className="p-5">
                            <h3 className="text-sm font-semibold text-[var(--text-primary)]">{label}</h3>
                            <p className="text-xs text-[var(--text-muted)] mt-2">Not detected</p>
                          </TiltCard>
                        </Reveal>
                      );
                    return (
                      <Reveal key={label} delay={idx * 120} variant="scale">
                        <TiltCard className="p-5 chart-3d">
                          <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-3">{label}</h3>
                          <ScrollBarChart data={data} duration={1400} />
                          <div className="mt-2 space-y-0.5">
                            {data.map((d) => (
                              <div key={d.group} className="flex justify-between text-[10px]">
                                <span className="text-[var(--text-muted)]">{d.group}</span>
                                <span className="text-[var(--text-secondary)]">
                                  {d.percentage}% <span className="text-[var(--text-muted)]">(n={d.count})</span>
                                </span>
                              </div>
                            ))}
                          </div>
                        </TiltCard>
                      </Reveal>
                    );
                  })}
                </div>
              </>
            )}

            {/* Feature Importance — bars fill on reveal */}
            {s.res?.features?.length ? (
              <>
                <h2 className="mb-4 text-lg font-semibold text-[var(--text-primary)] section-heading">Feature Importance Ranking</h2>
                <Reveal variant="scale">
                  <TiltCard className="mb-12 p-5">
                    <div className="space-y-2.5">
                      {s.res!.features.map((f, i) => (
                        <div key={i} className="flex items-center gap-3">
                          <span className="text-xs font-medium text-[var(--text-muted)] w-5">{i + 1}</span>
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="text-sm font-medium text-[var(--text-primary)]">{f.column}</span>
                              <span
                                className={`text-[9px] font-semibold px-1.5 py-0.5 rounded ${
                                  f.is_demographic ? "bg-blue-50 text-blue-700" : "bg-emerald-50 text-emerald-700"
                                }`}
                              >
                                {f.is_demographic ? "demographic" : "feature"}
                              </span>
                              {f.is_artifact && (
                                <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-red-50 text-red-700">ARTIFACT</span>
                              )}
                            </div>
                            <div className="h-1.5 rounded-full bg-[var(--bg-subtle)] overflow-hidden">
                              <div
                                className="h-full rounded-full transition-[width] duration-1000 ease-out"
                                style={{
                                  width: `${f.abs_correlation * 100}%`,
                                  backgroundColor: f.is_artifact ? "#7c5cff" : sC(1 - f.abs_correlation),
                                  transitionDelay: `${i * 60}ms`,
                                }}
                              />
                            </div>
                          </div>
                          <span className="text-xs font-mono font-medium text-[var(--text-secondary)] w-12 text-right">
                            {f.correlation.toFixed(2)}
                          </span>
                        </div>
                      ))}
                      {s.res!.confounder_strength > 0.3 && (
                        <p className="mt-3 text-[11px] text-amber-700 border-t border-[var(--border-subtle)] pt-3">
                          ⚠ Confounder influence: {(s.res!.confounder_strength * 100).toFixed(0)}% — bias severity should be interpreted with caution
                        </p>
                      )}
                    </div>
                  </TiltCard>
                </Reveal>
              </>
            ) : null}

            {/* Intersectional */}
            {s.res?.intersectional?.length ? (
              <>
                <h2 className="mb-4 text-lg font-semibold text-[var(--text-primary)] section-heading">Intersectional Analysis</h2>
                <div className="mb-12 space-y-3">
                  {s.res!.intersectional.map((ix, i) => (
                    <Reveal key={i} delay={i * 80} variant="scale">
                      <TiltCard className="p-5">
                        <div className="flex items-start justify-between mb-3">
                          <div>
                            <h3 className="text-sm font-semibold text-[var(--text-primary)]">{ix.label}</h3>
                            <p className="text-[11px] text-[var(--text-muted)] mt-0.5">
                              Worst: <span className="text-red-600">{ix.worst_group}</span> · Best:{" "}
                              <span className="text-emerald-600">{ix.best_group}</span>
                            </p>
                          </div>
                          <div className="text-right">
                            <span className="text-xs font-mono font-semibold" style={{ color: sC(ix.disparate_impact) }}>
                              DI: {ix.disparate_impact.toFixed(2)}
                            </span>
                            <br />
                            <span className="text-[10px] font-medium" style={{ color: sevC(ix.severity.color) }}>
                              {ix.severity.tier}
                            </span>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                          {ix.breakdown.map((g) => (
                            <div key={g.group} className="rounded-md bg-[var(--bg-subtle)] p-2.5 text-center transition-transform hover:scale-105">
                              <p className="text-[10px] text-[var(--text-muted)] truncate" title={g.group}>{g.group}</p>
                              <p className="text-sm font-semibold" style={{ color: bC(g.rate) }}>{g.percentage}%</p>
                              <p className="text-[9px] text-[var(--text-muted)]">n={g.count}</p>
                            </div>
                          ))}
                        </div>
                      </TiltCard>
                    </Reveal>
                  ))}
                </div>
              </>
            ) : null}

            {/* Data Preview — STYLE ONLY (no Reveal animation, no hover-lift) */}
            {s.res?.data_preview?.length ? (
              <>
                <h2 className="mb-4 text-lg font-semibold text-[var(--text-primary)] section-heading">Dataset Preview</h2>
                <div className="flat-card mb-12 overflow-auto max-h-[340px] p-0">
                  <table className="data-table">
                    <thead>
                      <tr>{s.res!.columns.map((c) => <th key={c}>{c}</th>)}</tr>
                    </thead>
                    <tbody>
                      {s.res!.data_preview.map((row, i) => (
                        <tr key={i}>
                          {s.res!.columns.map((c) => <td key={c}>{row[c] != null ? String(row[c]) : "—"}</td>)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : null}

            {/* AI Report — STYLE ONLY (no Reveal animation, no hover-lift) */}
            <h2 className="mb-4 text-lg font-semibold text-[var(--text-primary)] section-heading">AI Analysis Report</h2>
            <div className="flat-card p-6 md:p-8 mb-8">
              {s.ph === "analyzing" ? (
                <div className="space-y-3">
                  {[100, 95, 88, 92, 78, 85].map((w, i) => (
                    <div key={i} className="animate-shimmer h-3" style={{ width: `${w}%` }} />
                  ))}
                </div>
              ) : s.res?.ai_report ? (
                <div className="prose-dark"><Markdown>{s.res!.ai_report}</Markdown></div>
              ) : (
                <p className="text-sm text-[var(--text-muted)]">Run analysis to generate report.</p>
              )}
              {s.res?.ai_report && (
                <button
                  onClick={() => window.print()}
                  className="no-print mt-6 flex items-center gap-2 rounded-md border border-[var(--border-subtle)] px-3 py-1.5 text-xs font-medium text-[var(--text-secondary)] hover:border-[var(--accent-blue)] hover:text-[var(--accent-blue)] transition-colors"
                >
                  📄 Export Report
                </button>
              )}
            </div>
          </div>
        )}

        <footer className="mt-20 border-t border-[var(--border-subtle)] pt-8 text-center">
          <p className="text-sm text-[var(--text-muted)]">AI Bias Inspector — Ensuring fairness in automated decisions</p>
          <p className="mt-2 text-xs text-[var(--text-muted)]">
            built by <span className="gradient-text font-semibold">team nostalgia</span>
          </p>
        </footer>
      </div>

      {/* Floating Chat Bubble */}
      {s.res && (
        <div className="fixed bottom-6 right-6 z-50 no-print">
          {chatOpen && (
            <div className="absolute bottom-16 right-0 w-[380px] rounded-xl border border-[var(--border-subtle)] bg-white shadow-[0_12px_32px_-8px_rgba(16,24,40,0.18)] overflow-hidden animate-fade-in-up">
              <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border-subtle)] bg-[var(--bg-subtle)]">
                <div>
                  <h3 className="text-sm font-semibold text-[var(--text-primary)]">Ask about bias</h3>
                  <p className="text-[10px] text-[var(--text-muted)]">Ask anything about bias in your data</p>
                </div>
                <button onClick={() => setChatOpen(false)} className="text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors text-lg leading-none">✕</button>
              </div>
              <div className="h-[300px] overflow-y-auto p-3 space-y-2 chat-scroll">
                {chatMsgs.length === 0 && (
                  <div className="flex flex-col items-center justify-center h-full text-center gap-2">
                    <span className="text-2xl opacity-60">🔍</span>
                    <p className="text-xs text-[var(--text-muted)]">Try asking:</p>
                    <div className="space-y-1 w-full">
                      {["Explain gender bias", "Why is ethnicity biased?", "Is there hidden bias?"].map((q) => (
                        <button
                          key={q}
                          onClick={() => setChatInput(q)}
                          className="block w-full text-[11px] text-[var(--accent-blue)] hover:bg-[var(--bg-subtle)] rounded-md px-3 py-2 transition-colors text-left"
                        >
                          {q}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
                {chatMsgs.map((m, i) => (
                  <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div
                      className={`max-w-[85%] rounded-2xl px-3 py-2 text-[12px] leading-relaxed ${
                        m.role === "user"
                          ? "bg-[var(--accent-blue)] text-white rounded-br-md"
                          : "bg-[var(--bg-subtle)] text-[var(--text-primary)] rounded-bl-md"
                      }`}
                    >
                      {m.role === "ai" ? (
                        <div className="prose-dark chat-prose"><Markdown>{m.text}</Markdown></div>
                      ) : m.text}
                    </div>
                  </div>
                ))}
                {chatLoading && (
                  <div className="flex justify-start">
                    <div className="rounded-2xl rounded-bl-md bg-[var(--bg-subtle)] px-3 py-2 text-xs text-[var(--text-muted)]">
                      <span className="inline-flex gap-1">
                        <span className="animate-bounce" style={{ animationDelay: "0ms" }}>·</span>
                        <span className="animate-bounce" style={{ animationDelay: "150ms" }}>·</span>
                        <span className="animate-bounce" style={{ animationDelay: "300ms" }}>·</span>
                      </span>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
              <div className="border-t border-[var(--border-subtle)] p-3 bg-white">
                <div className="flex gap-2">
                  <input
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter") sendChat(); }}
                    placeholder="Ask about the analysis..."
                    className="flex-1 rounded-lg border border-[var(--border-subtle)] bg-white px-3 py-2 text-xs text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:border-[var(--accent-blue)] focus:outline-none focus:ring-2 focus:ring-[var(--accent-blue)]/20"
                  />
                  <button
                    onClick={sendChat}
                    disabled={chatLoading || !chatInput.trim()}
                    className="gradient-btn rounded-lg px-3 py-2 text-xs font-semibold disabled:opacity-40"
                  >
                    ↑
                  </button>
                </div>
              </div>
            </div>
          )}
          <button
            onClick={() => setChatOpen((o) => !o)}
            className="h-[52px] w-[52px] rounded-full gradient-btn flex items-center justify-center text-lg shadow-[0_8px_20px_-4px_rgba(59,108,255,0.4)] transition-transform hover:scale-110"
          >
            {chatOpen ? "✕" : "💬"}
          </button>
        </div>
      )}
    </main>
  );
}
