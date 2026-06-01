import { ChangeEvent, DragEvent, PointerEvent, useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { Variants } from "framer-motion";
import {
  AlertCircle,
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  ClipboardCheck,
  FileImage,
  HeartPulse,
  ImageUp,
  Layers3,
  Loader2,
  Microscope,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  UploadCloud,
} from "lucide-react";
import { getHealth, getMetrics, predictImage, HealthResponse, MetricsResponse, PredictionResponse } from "./api";

type AnalysisPhase = "idle" | "analyzing" | "complete" | "result";

const classLabels: Record<string, string> = {
  benign: "良性",
  malignant: "恶性",
  normal: "正常",
};

const classDescriptions: Record<string, string> = {
  benign: "模型倾向于良性表现，仍建议结合影像医师意见综合判断。",
  malignant: "模型提示需要重点关注，建议尽快携带检查资料咨询专业医生。",
  normal: "模型倾向于未见明显异常表现，如有症状仍建议按医嘱随访。",
};

const classTone: Record<string, string> = {
  benign: "tone-benign",
  malignant: "tone-malignant",
  normal: "tone-normal",
};

const probabilityTone: Record<string, string> = {
  benign: "bar-benign",
  malignant: "bar-malignant",
  normal: "bar-normal",
};

const container: Variants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08, delayChildren: 0.08 },
  },
};

const rise: Variants = {
  hidden: { opacity: 0, y: 20, filter: "blur(8px)" },
  show: { opacity: 1, y: 0, filter: "blur(0px)", transition: { duration: 0.65, ease: "easeOut" } },
};

function formatPercent(value?: number) {
  if (typeof value !== "number" || Number.isNaN(value)) return "--";
  return `${(value * 100).toFixed(1)}%`;
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function labelFor(className: string) {
  return classLabels[className] ?? className;
}

function phaseLabel(phase: AnalysisPhase) {
  if (phase === "analyzing") return "模型正在分析";
  if (phase === "complete") return "结果生成完成";
  if (phase === "result") return "分析结果已就绪";
  return "等待上传影像";
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [analysisPhase, setAnalysisPhase] = useState<AnalysisPhase>("idle");
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState("");
  const [visualTilt, setVisualTilt] = useState({ rx: 0, ry: 0, x: 52, y: 46 });

  const isAnalyzing = analysisPhase === "analyzing";
  const isCompleting = analysisPhase === "complete";
  const isActiveMotion = isAnalyzing || isCompleting;

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([getHealth(), getMetrics()]).then(([healthResult, metricsResult]) => {
      if (cancelled) return;
      if (healthResult.status === "fulfilled") setHealth(healthResult.value);
      if (metricsResult.status === "fulfilled") setMetrics(metricsResult.value);
      if (healthResult.status === "rejected") setError("后端服务暂未连接，请先启动 FastAPI。");
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl("");
      return;
    }
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [selectedFile]);

  const platformState = useMemo(() => {
    if (!health) return "服务连接中";
    if (!health.checkpoint_exists) return "模型待部署";
    return "研究演示版";
  }, [health]);

  function acceptFile(file: File | undefined) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("请上传 PNG、JPG、JPEG、BMP 等图像文件。");
      return;
    }
    setSelectedFile(file);
    setPrediction(null);
    setAnalysisPhase("idle");
    setError("");
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    acceptFile(event.target.files?.[0]);
    event.target.value = "";
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(false);
    acceptFile(event.dataTransfer.files?.[0]);
  }

  async function analyze() {
    if (!selectedFile || isAnalyzing) return;
    setAnalysisPhase("analyzing");
    setError("");
    try {
      const result = await predictImage(selectedFile);
      setPrediction(result);
      setAnalysisPhase("complete");
      const refreshedHealth = await getHealth().catch(() => null);
      if (refreshedHealth) setHealth(refreshedHealth);
      window.setTimeout(() => setAnalysisPhase("result"), 1050);
    } catch (exc) {
      setAnalysisPhase("idle");
      setError(exc instanceof Error ? exc.message : "分析失败，请检查后端服务。");
    }
  }

  const currentDescription = prediction ? classDescriptions[prediction.predicted_class] : "";

  return (
    <main className={`app-shell phase-${analysisPhase}`}>
      <div className="lab-grid" />
      <div className="lab-orbit orbit-one" />
      <div className="lab-orbit orbit-two" />

      <motion.header className="site-header" variants={rise} initial="hidden" animate="show">
        <a className="logo-lockup" href="#analysis" aria-label="BreastAI Insight 首页">
          <span className="logo-mark">
            <HeartPulse size={22} />
          </span>
          <span>
            <strong>BreastAI Insight</strong>
            <em>乳腺超声 AI 辅助分析平台</em>
          </span>
        </a>
        <nav className="site-nav" aria-label="页面导航">
          <a href="#analysis">影像分析</a>
          <a href="#guidance">辅助说明</a>
          <a href="#guidance">判断依据</a>
          <a href="#limits">系统边界</a>
        </nav>
      </motion.header>

      <motion.section className="hero-section" variants={container} initial="hidden" animate="show">
        <motion.div className="hero-copy" variants={rise}>
          <p className="eyebrow">AI Medical Imaging Product</p>
          <h1>面向乳腺超声影像的智能分类与可解释分析</h1>
          <p className="hero-lede">
            上传乳腺超声图像后，系统会给出模型分析倾向、三分类概率与 Grad-CAM 关注区域，用于课程展示、科研演示和辅助理解模型行为。
          </p>
          <div className="hero-actions">
            <a className="primary-link" href="#analysis">
              开始影像分析
              <ArrowRight size={18} />
            </a>
            <span className="state-chip">{platformState}</span>
          </div>
        </motion.div>
        <motion.div className="hero-panel" variants={rise}>
          <span>Gradient Response</span>
          <div
            className="imaging-visual"
            aria-label="乳腺超声 AI 分析示意图"
            style={
              {
                "--tilt-x": `${visualTilt.rx}deg`,
                "--tilt-y": `${visualTilt.ry}deg`,
                "--focus-x": `${visualTilt.x}%`,
                "--focus-y": `${visualTilt.y}%`,
              } as React.CSSProperties
            }
            onPointerMove={(event: PointerEvent<HTMLDivElement>) => {
              const rect = event.currentTarget.getBoundingClientRect();
              const x = ((event.clientX - rect.left) / rect.width) * 100;
              const y = ((event.clientY - rect.top) / rect.height) * 100;
              setVisualTilt({
                rx: (50 - y) / 8,
                ry: (x - 50) / 8,
                x: Math.min(82, Math.max(18, x)),
                y: Math.min(76, Math.max(22, y)),
              });
            }}
            onPointerLeave={() => setVisualTilt({ rx: 0, ry: 0, x: 52, y: 46 })}
          >
            <span className="ultrasound-texture" />
            <span className="lesion-focus" />
            <span className="focus-ring ring-one" />
            <span className="focus-ring ring-two" />
            <div className="ai-readout">
              <b>Gradient Focus</b>
              <em>pointer reactive activation</em>
            </div>
          </div>
          <p>以产品化方式呈现分类倾向、概率分布与模型关注区域。</p>
        </motion.div>
      </motion.section>

      <motion.section id="analysis" className="analysis-shell" variants={container} initial="hidden" animate="show">
        <motion.div className="section-heading" variants={rise}>
          <p className="eyebrow">Image Analysis Console</p>
          <h2>影像分析工作台</h2>
          <span>仅用于课程与科研演示，不作为医学诊断依据。</span>
        </motion.div>

        <div className="dashboard-grid">
          <motion.section className={`panel upload-panel ${isDragging ? "dragging" : ""} ${isActiveMotion ? "motion-active" : ""}`} variants={rise}>
            <PanelTitle label="Input" title="上传乳腺超声图像" icon={<ImageUp size={22} />} />
            <motion.label
              className={`drop-zone ${previewUrl ? "has-preview" : ""}`}
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              animate={isActiveMotion ? { scale: [1, 1.006, 1] } : { scale: 1 }}
              transition={isActiveMotion ? { repeat: Infinity, duration: 1.6, ease: "easeInOut" } : { duration: 0.2 }}
            >
              <span className="analysis-glow" />
              {previewUrl ? (
                <div className="image-stage">
                  <img src={previewUrl} alt="上传的乳腺超声图像预览" />
                  <AnimatePresence>{isActiveMotion && <motion.span className="lab-scan" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} />}</AnimatePresence>
                  <AnimatePresence>{isCompleting && <motion.span className="completion-flare" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} />}</AnimatePresence>
                </div>
              ) : (
                <div className="empty-upload">
                  <UploadCloud size={42} />
                  <strong>拖拽或点击上传图像</strong>
                  <span>支持 PNG / JPG / JPEG / BMP</span>
                </div>
              )}
              <input type="file" accept="image/*" onChange={handleFileChange} />
            </motion.label>

            <div className="file-row">
              <FileImage size={18} />
              <div>
                <strong>{selectedFile?.name ?? "尚未选择图像"}</strong>
                <span>{selectedFile ? formatBytes(selectedFile.size) : "请选择一张清晰、区域完整的超声图像"}</span>
              </div>
            </div>

            <button className="analyze-button" onClick={analyze} disabled={!selectedFile || isAnalyzing}>
              {isAnalyzing ? <Loader2 className="spin" size={19} /> : <Sparkles size={19} />}
              {isAnalyzing ? "分析生成中" : isCompleting ? "结果生成完成" : "开始分析"}
            </button>
          </motion.section>

          <motion.section className={`panel result-panel ${isCompleting ? "result-reveal" : ""}`} variants={rise}>
            <PanelTitle label="Prediction" title="模型分析倾向" icon={<BrainCircuit size={22} />} />
            <div className="phase-line">
              <span className={`phase-dot ${isActiveMotion ? "active" : ""} ${analysisPhase === "result" ? "done" : ""}`} />
              {phaseLabel(analysisPhase)}
            </div>

            <AnimatePresence mode="wait">
              {prediction ? (
                <motion.div key="prediction" className="prediction-stack" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
                  <div className={`verdict ${classTone[prediction.predicted_class] ?? ""}`}>
                    <span>模型分析倾向</span>
                    <strong>{labelFor(prediction.predicted_class)}</strong>
                    <em>{formatPercent(prediction.confidence)} 置信度</em>
                    <p>{currentDescription}</p>
                  </div>

                  <div className="probability-list">
                    {prediction.probabilities.map((item, index) => (
                      <motion.div className="probability-item" key={item.class_name} initial={{ opacity: 0, x: -14 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: index * 0.08 }}>
                        <div className="probability-label">
                          <span>{labelFor(item.class_name)}</span>
                          <strong>{formatPercent(item.probability)}</strong>
                        </div>
                        <div className="probability-track">
                          <motion.span
                            className={probabilityTone[item.class_name] ?? ""}
                            initial={{ width: 0 }}
                            animate={{ width: `${item.probability * 100}%` }}
                            transition={{ duration: 0.85, ease: "easeOut", delay: index * 0.1 }}
                          />
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              ) : (
                <motion.div key="empty-result" className="placeholder-state" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <ScanSearch size={40} />
                  <strong>{isAnalyzing ? "正在解析影像纹理" : "等待分析结果"}</strong>
                  <span>{isAnalyzing ? "模型正在生成分类概率与可解释热力图" : "上传图像后点击开始分析"}</span>
                </motion.div>
              )}
            </AnimatePresence>

            {error && (
              <motion.div className="error-box" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
                <AlertCircle size={18} />
                <span>{error}</span>
              </motion.div>
            )}
          </motion.section>

          <motion.section className={`panel gradcam-panel ${isActiveMotion ? "motion-active" : ""}`} variants={rise}>
            <PanelTitle label="Explanation" title="Grad-CAM 可解释热力图" icon={<Layers3 size={22} />} />
            <div className="heatmap-stage">
              <AnimatePresence mode="wait">
                {prediction?.gradcam_image ? (
                  <motion.img
                    key={prediction.gradcam_image}
                    src={prediction.gradcam_image}
                    alt="Grad-CAM 热力图"
                    initial={{ opacity: 0, scale: 0.97, filter: "blur(12px)" }}
                    animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.75, ease: "easeOut" }}
                  />
                ) : (
                  <motion.div key="heatmap-empty" className="placeholder-state compact" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    <Microscope size={40} />
                    <strong>热力图待生成</strong>
                    <span>分析完成后显示模型关注区域</span>
                  </motion.div>
                )}
              </AnimatePresence>
              {isActiveMotion && <span className="lab-scan heatmap-scan" />}
            </div>
            <p className="explain-note">高亮区域表示模型关注位置，不等同于病灶边界或医学诊断结论。</p>
          </motion.section>
        </div>
      </motion.section>

      <motion.section id="guidance" className="content-section" variants={container} initial="hidden" whileInView="show" viewport={{ once: true, margin: "-80px" }}>
        <SectionIntro title="判断依据" text="以下内容用于解释模型输出的常见依据，帮助理解三分类结果与 Grad-CAM 关注区域。" />
        <div className="evidence-grid">
          <InfoCard icon={<CheckCircle2 size={22} />} title="形态与边界" text="模型会响应肿块轮廓、边缘清晰度、形态规则性等纹理特征，这些信息会影响良性或恶性倾向。" />
          <InfoCard icon={<ClipboardCheck size={22} />} title="内部回声模式" text="超声图像中的灰阶分布、低回声区域、回声均匀性等变化，是分类模型学习的重要视觉线索。" />
          <InfoCard icon={<ScanSearch size={22} />} title="关注区域解释" text="Grad-CAM 高亮区域表示模型对结果贡献较大的位置，用于观察模型是否关注到可疑影像区域。" />
          <InfoCard icon={<AlertCircle size={22} />} title="结果置信度" text="三分类概率展示模型在正常、良性、恶性之间的相对倾向；概率接近时代表模型区分度有限。" />
        </div>
      </motion.section>

      <motion.section id="limits" className="content-section closing-section" variants={container} initial="hidden" whileInView="show" viewport={{ once: true, margin: "-80px" }}>
        <SectionIntro title="系统边界" text="AI 结果适合辅助展示和模型解释，不替代医生诊断、病理检查或临床决策。" />
        <div className="boundary-panel">
          <ShieldCheck size={28} />
          <div>
            <h3>面向教学与科研演示的辅助分析工具</h3>
            <p>请将模型输出理解为“当前图像下的算法倾向”。任何医疗判断都应结合医生阅片、体征、病史、其他检查和病理结果。</p>
          </div>
          <Microscope size={28} />
        </div>
      </motion.section>
    </main>
  );
}

function PanelTitle({ label, title, icon }: { label: string; title: string; icon: React.ReactNode }) {
  return (
    <div className="panel-heading">
      <div>
        <p className="eyebrow">{label}</p>
        <h3>{title}</h3>
      </div>
      {icon}
    </div>
  );
}

function SectionIntro({ title, text }: { title: string; text: string }) {
  return (
    <motion.div className="section-intro" variants={rise}>
      <p className="eyebrow">Clinical Context</p>
      <h2>{title}</h2>
      <span>{text}</span>
    </motion.div>
  );
}

function InfoCard({ icon, title, text }: { icon: React.ReactNode; title: string; text: string }) {
  return (
    <motion.article className="info-card" variants={rise}>
      <div>{icon}</div>
      <h3>{title}</h3>
      <p>{text}</p>
    </motion.article>
  );
}
