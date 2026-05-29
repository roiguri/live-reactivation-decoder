import { useState, useEffect, useCallback } from 'react';
import { Check, FolderOpen, MousePointer2, Download, ArrowLeft, Play, Cpu, FileJson, SlidersHorizontal } from 'lucide-react';
import { useWalkthroughAction } from '../walkthrough';

const EMPTY_PREPROC = {
  bandpass: { l_freq: '', h_freq: '', method: '', notch: null },
  resample: { target_rate: '' },
  ica: { n_components: '', method: '', random_state: '' },
  epochs: { tmin: '', tmax: '', baseline: [null, 0], reject: '' },
  autoreject: { random_state: '' },
  annotations: [],
};
const EMPTY_MODEL = { model: '', cvFolds: '', decoders: [], params: {} };

const LDA_DEFAULTS = { shrinkage: 'auto', solver: 'svd' };

export default function Phase1Screen({ onFinish, onBack, initialNode = 1 }) {
  const [activeNode, setActiveNode] = useState(initialNode);
  const [subStep, setSubStep] = useState(1); // Used for nodes with multiple views
  const [node3Decoders, setNode3Decoders] = useState([]);
  const [node3CVFolds, setNode3CVFolds] = useState(5);
  const [confirmedTimepoint, setConfirmedTimepoint] = useState(null);
  const [forceDecoderTab, setForceDecoderTab] = useState(null); // For tutorial to switch tabs

  // Settings state (Node 1: Settings workspace)
  const [preprocSettings, setPreprocSettings] = useState(EMPTY_PREPROC);
  const [modelSettings, setModelSettings] = useState(EMPTY_MODEL);
  const [configLoaded, setConfigLoaded] = useState(false);

  const handleLoadConfig = useCallback(() => {
    setPreprocSettings({
      bandpass: { l_freq: 1.0, h_freq: 40.0, method: 'iir', notch: 50.0 },
      resample: { target_rate: 250 },
      ica: { n_components: 20, method: 'fastica', random_state: 42 },
      epochs: { tmin: -0.2, tmax: 0.8, baseline: [null, 0], reject: 100e-6 },
      autoreject: { random_state: 42 },
      annotations: DEFAULT_ANNOTATIONS,
    });
    setModelSettings({ model: 'lda', cvFolds: 5, decoders: INITIAL_DECODERS, params: LDA_DEFAULTS });
    setConfigLoaded(true);
  }, []);

  // Register tutorial actions for internal navigation
  useWalkthroughAction('phase1-goto-settings', useCallback(() => {
    setActiveNode(1);
    setSubStep(1);
  }, []));

  useWalkthroughAction('phase1-load-config', handleLoadConfig);

  useWalkthroughAction('phase1-goto-data-ingestion', useCallback(() => {
    setActiveNode(2);
    setSubStep(1);
  }, []));

  useWalkthroughAction('phase1-goto-preproc-run', useCallback(() => {
    setActiveNode(3);
    setSubStep(1);
  }, []));

  useWalkthroughAction('phase1-goto-ica', useCallback(() => {
    setActiveNode(3);
    setSubStep(3);
  }, []));

  useWalkthroughAction('phase1-goto-eval-run', useCallback(() => {
    setActiveNode(4);
    setSubStep(1);
    setNode3Decoders([
      { id: 1, name: 'Red', positive: ['Red'], negative: ['Green', 'Yellow'] },
      { id: 2, name: 'Green', positive: ['Green'], negative: ['Red', 'Yellow'] },
      { id: 3, name: 'Yellow', positive: ['Yellow'], negative: ['Red', 'Green'] },
    ]);
  }, []));

  useWalkthroughAction('phase1-goto-eval-results', useCallback(() => {
    setActiveNode(4);
    setSubStep(3);
    setNode3Decoders([
      { id: 1, name: 'Red', positive: ['Red'], negative: ['Green', 'Yellow'] },
      { id: 2, name: 'Green', positive: ['Green'], negative: ['Red', 'Yellow'] },
      { id: 3, name: 'Yellow', positive: ['Yellow'], negative: ['Red', 'Green'] },
    ]);
  }, []));

  // These actions don't change state - just allow Next to proceed (element is already visible)
  useWalkthroughAction('phase1-focus-timepoint', useCallback(() => {}, []));
  useWalkthroughAction('phase1-show-decoder-tab', useCallback(() => {
    setForceDecoderTab(1);
  }, []));

  useWalkthroughAction('phase1-goto-deploy', useCallback(() => {
    setActiveNode(5);
    setSubStep(2);
    setNode3Decoders([
      { id: 1, name: 'Red', positive: ['Red'], negative: ['Green', 'Yellow'] },
      { id: 2, name: 'Green', positive: ['Green'], negative: ['Red', 'Yellow'] },
      { id: 3, name: 'Yellow', positive: ['Yellow'], negative: ['Red', 'Green'] },
    ]);
  }, []));

  useWalkthroughAction('phase1-go-live', useCallback(() => onFinish(), [onFinish]));

  // Reverse actions for Back button - restore previous views
  useWalkthroughAction('phase1-restore-settings', useCallback(() => {
    setActiveNode(1);
    setSubStep(1);
  }, []));

  useWalkthroughAction('phase1-restore-data-ingestion', useCallback(() => {
    setActiveNode(2);
    setSubStep(1);
  }, []));

  useWalkthroughAction('phase1-restore-preproc-run', useCallback(() => {
    setActiveNode(3);
    setSubStep(1);
  }, []));

  useWalkthroughAction('phase1-restore-ica', useCallback(() => {
    setActiveNode(3);
    setSubStep(3);
  }, []));

  useWalkthroughAction('phase1-restore-eval-run', useCallback(() => {
    setActiveNode(4);
    setSubStep(1);
    setForceDecoderTab('summary');
  }, []));

  useWalkthroughAction('phase1-restore-eval-results', useCallback(() => {
    setActiveNode(4);
    setSubStep(3);
    setForceDecoderTab('summary');
  }, []));

  useWalkthroughAction('phase1-restore-summary-tab', useCallback(() => {
    setActiveNode(4);
    setSubStep(3);
    setForceDecoderTab('summary');
  }, []));

  useWalkthroughAction('phase1-restore-decoder-tab', useCallback(() => {
    setActiveNode(4);
    setSubStep(3);
    setForceDecoderTab(1);
  }, []));

  const advanceNode = () => {
    setActiveNode(prev => prev + 1);
    setSubStep(1);
    if (activeNode === 4) {
      setConfirmedTimepoint(null);
    }
  };

  const handleBack = () => {
    if (activeNode === 1) {
      if (onBack) onBack();
    } else if (activeNode === 2) {
      setActiveNode(1);
    } else if (activeNode === 3) {
      if (subStep === 1) setActiveNode(2);
      else if (subStep === 3) setSubStep(1);
      else if (subStep === 5) setSubStep(3);
    } else if (activeNode === 4) {
      if (subStep === 1) {
        setActiveNode(3);
        setSubStep(5);
      } else if (subStep === 3 && confirmedTimepoint !== null) {
        setConfirmedTimepoint(null);
      } else if (subStep === 3) setSubStep(1);
    } else if (activeNode === 5) {
      if (subStep === 1 || subStep === 2) {
        setActiveNode(4);
        setSubStep(3);
      }
    }
  };

  return (
    <div className="h-full flex">
      {/* LEFT PANEL: Dynamic Workspace */}
      <div className="flex-1 p-6 flex flex-col h-full overflow-hidden">
        <div className="flex-1 bg-white shadow-sm border border-gray-200 rounded-md flex flex-col overflow-hidden">
          {/* Header */}
          <div className="h-12 border-b border-gray-100 flex items-center px-6 bg-gray-50/50">
            {onBack && (
              <button onClick={handleBack} className="mr-4 text-gray-400 hover:text-gray-700 transition" title="Go Back">
                <ArrowLeft className="w-5 h-5" />
              </button>
            )}
            <h2 className="text-lg font-medium text-gray-800">
              {activeNode === 1 && "Pipeline Settings"}
              {activeNode === 2 && "Data Ingestion"}
              {activeNode === 3 && (
                subStep === 1 ? "Preprocessing — Ready" :
                subStep === 2 ? "Running Preprocessing…" :
                subStep === 3 ? "ICA Artifact Rejection" :
                subStep === 4 ? "Finalizing Preprocessing…" :
                "Preprocessing Complete"
              )}
              {activeNode === 4 && (
                subStep === 1 ? "Model Evaluation — Ready" :
                subStep === 2 ? "Training Decoders…" :
                "Model Evaluation — Results"
              )}
              {activeNode === 5 && (
                subStep === 1 ? "Training Final Decoders…" :
                "Final Review & Deployment"
              )}
            </h2>
          </div>

          {/* Workspace Content */}
          <div className="flex-1 p-6 overflow-y-auto">
            {activeNode === 1 && <WorkspaceNode1Settings onLoadConfig={handleLoadConfig} preprocSettings={preprocSettings} modelSettings={modelSettings} configLoaded={configLoaded} />}
            {activeNode === 2 && <WorkspaceNode2 />}
            {activeNode === 3 && subStep === 1 && <WorkspaceNode3Run onRun={() => setSubStep(2)} />}
            {activeNode === 3 && subStep === 2 && <WorkspaceNode2Progress startFrom={0} onDone={() => setSubStep(3)} />}
            {activeNode === 3 && subStep === 3 && <WorkspaceNode2ICA />}
            {activeNode === 3 && subStep === 4 && <WorkspaceNode2Progress startFrom={3} onDone={() => setSubStep(5)} />}
            {activeNode === 3 && subStep === 5 && <WorkspaceNode2Complete />}
            {activeNode === 4 && subStep === 1 && <WorkspaceNode4Run onRun={() => { setNode3Decoders(modelSettings.decoders.length ? modelSettings.decoders : INITIAL_DECODERS); setNode3CVFolds(modelSettings.cvFolds || 5); setSubStep(2); }} />}
            {activeNode === 4 && subStep === 2 && <WorkspaceNode3CVProgress decoders={node3Decoders} cvFolds={node3CVFolds} onDone={() => setSubStep(3)} />}
            {activeNode === 4 && subStep === 3 && <WorkspaceNode3Results decoders={node3Decoders} confirmedTimepoint={confirmedTimepoint} setConfirmedTimepoint={setConfirmedTimepoint} forceDecoderTab={forceDecoderTab} />}
            {activeNode === 5 && subStep === 1 && <WorkspaceNode4Progress decoders={node3Decoders} onDone={() => setSubStep(2)} />}
            {activeNode === 5 && subStep === 2 && <WorkspaceNode4 />}
          </div>
        </div>
      </div>

      {/* RIGHT PANEL: The Journey Path */}
      <div data-tour="pipeline-overview" className="w-80 bg-white border-l border-gray-200 p-6 shadow-[-4px_0_15px_-3px_rgba(0,0,0,0.05)] z-10 flex flex-col">
        <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-8">Training Pipeline</h3>

        <div className="flex-1 relative">
          {/* Connecting Line */}
          <div className="absolute left-4 top-4 bottom-12 w-0.5 bg-gray-200" />

          <JourneyNode title="Settings" nodeNum={1} activeNode={activeNode}
            description={configLoaded ? "Config loaded" : "Configure pipeline"}
            actionText="Continue" onAction={advanceNode} />

          <JourneyNode title="Load Data" nodeNum={2} activeNode={activeNode}
            description="Ingest Localizer EEG" actionText="Load & Continue" onAction={advanceNode} />

          <JourneyNode title="Preprocessing" nodeNum={3} activeNode={activeNode}
            description={
              subStep === 1 ? "Ready to run" :
              subStep === 2 ? "Running pipeline…" :
              subStep === 3 ? "Review ICA components" :
              subStep === 4 ? "Completing pipeline…" :
              "Pipeline complete"
            }
            actionText={subStep === 3 ? "Confirm & Continue" : subStep === 5 ? "Continue to Evaluation" : null}
            onAction={subStep === 3 ? () => setSubStep(4) : advanceNode} />

          <JourneyNode title="Model Evaluation" nodeNum={4} activeNode={activeNode}
            description={
              subStep === 1 ? "Ready to run" :
              subStep === 2 ? "Training in progress…" :
              "Review results"
            }
            actionText={subStep === 3 ? (confirmedTimepoint !== null ? "Approve & Continue" : "Select Timepoint") : null}
            isDisabled={subStep === 3 && confirmedTimepoint === null}
            onAction={subStep === 3 && confirmedTimepoint !== null ? advanceNode : undefined} />

          <JourneyNode title="Train & Save" nodeNum={5} activeNode={activeNode}
            description={
              subStep === 1 ? "Training final decoders…" :
              "Finalize decoders"
            }
            actionText={subStep === 2 ? "Go to Live Mode" : null}
            actionTourId={subStep === 2 ? "go-live-btn" : undefined}
            onAction={onFinish} isLast />
        </div>
      </div>
    </div>
  );
}

// --- WORKSPACE COMPONENTS ---

function WorkspaceNode1Settings({ onLoadConfig, preprocSettings, modelSettings, configLoaded }) {
  return (
    <div data-tour="step-settings" className="max-w-2xl mx-auto py-6 space-y-8">
      {/* Load Config Button */}
      <div className="flex items-center gap-4 pb-4 border-b border-gray-100">
        <button
          data-tour="load-config-btn"
          onClick={onLoadConfig}
          className="flex items-center gap-2 bg-[#0078D4] hover:bg-[#006CBE] text-white px-5 py-2 rounded-sm text-sm font-medium transition-colors"
        >
          <FileJson className="w-4 h-4" />
          Load Config File
        </button>
        {configLoaded && (
          <span className="text-xs text-green-600 font-medium flex items-center gap-1">
            <Check className="w-3 h-3" /> Config loaded
          </span>
        )}
        {!configLoaded && (
          <span className="text-xs text-gray-400">Load a config file to auto-populate settings below</span>
        )}
      </div>

      {/* Preprocessing Settings */}
      <div data-tour="preproc-settings-section">
        <div className="flex items-center gap-2 mb-4">
          <SlidersHorizontal className="w-4 h-4 text-gray-500" />
          <h3 className="text-sm font-semibold text-gray-700">Preprocessing</h3>
        </div>
        <div className="space-y-4 pl-6">
          <div>
            <label className="block text-xs font-semibold text-gray-500 mb-1 uppercase">Bandpass</label>
            <div className="flex items-center gap-2">
              <input readOnly value={preprocSettings.bandpass?.l_freq} placeholder="—"
                className="w-20 border border-gray-200 rounded-sm px-3 py-2 font-mono text-sm bg-gray-50 text-gray-600 placeholder-gray-300" />
              <span className="text-gray-400 text-sm">to</span>
              <input readOnly value={preprocSettings.bandpass?.h_freq} placeholder="—"
                className="w-20 border border-gray-200 rounded-sm px-3 py-2 font-mono text-sm bg-gray-50 text-gray-600 placeholder-gray-300" />
              <span className="text-xs text-gray-500">Hz</span>
              <span className="ml-3 text-[10px] text-gray-500">Method: {preprocSettings.bandpass?.method ?? '—'}</span>
              <span className="ml-2 text-[10px] text-gray-500">Notch: {preprocSettings.bandpass?.notch ?? '—'}</span>
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-gray-500 mb-1 uppercase">Resample</label>
            <div className="flex items-center gap-2">
              <input readOnly value={preprocSettings.resample?.target_rate} placeholder="—"
                className="w-28 border border-gray-200 rounded-sm px-3 py-2 font-mono text-sm bg-gray-50 text-gray-600 placeholder-gray-300" />
              <span className="text-xs text-gray-500">Hz</span>
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-gray-500 mb-1 uppercase">ICA</label>
            <div className="flex items-center gap-3">
              <input readOnly value={preprocSettings.ica?.n_components} placeholder="—"
                className="w-24 border border-gray-200 rounded-sm px-3 py-2 font-mono text-sm bg-gray-50 text-gray-600 placeholder-gray-300" />
              <span className="text-[10px] text-gray-500">components</span>
              <span className="ml-3 text-[10px] text-gray-500">Method: {preprocSettings.ica?.method ?? '—'}</span>
              <span className="ml-2 text-[10px] text-gray-500">Seed: {preprocSettings.ica?.random_state ?? '—'}</span>
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-gray-500 mb-1 uppercase">Epoch Size (ms)</label>
            <div className="flex items-center gap-2">
              <input readOnly value={preprocSettings.epochs?.tmin} placeholder="—"
                className="w-28 border border-gray-200 rounded-sm px-3 py-2 font-mono text-sm bg-gray-50 text-gray-600 placeholder-gray-300" />
              <span className="text-gray-400 text-sm">to</span>
              <input readOnly value={preprocSettings.epochs?.tmax} placeholder="—"
                className="w-28 border border-gray-200 rounded-sm px-3 py-2 font-mono text-sm bg-gray-50 text-gray-600 placeholder-gray-300" />
            </div>
            <div className="mt-2 text-[10px] text-gray-500">Baseline: {String(preprocSettings.epochs?.baseline?.[0])} to {String(preprocSettings.epochs?.baseline?.[1])} — Reject: {preprocSettings.epochs?.reject ?? '—'}</div>
          </div>

          {/* Smoothing and Normalization removed per design — fields no longer displayed */}

          <div>
            <label className="block text-xs font-semibold text-gray-500 mb-2 uppercase">Autoreject</label>
            <div className="text-[10px] text-gray-500">Seed: {preprocSettings.autoreject?.random_state ?? '—'}</div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-gray-500 mb-2 uppercase">Annotations Mapping</label>
            {preprocSettings.annotations.length === 0 ? (
              <p className="text-sm text-gray-300 italic">No annotations loaded</p>
            ) : (
              <div className="border border-gray-200 rounded-sm overflow-hidden">
                <div className="grid grid-cols-[1fr_1.8fr] bg-gray-50 border-b border-gray-200 px-3 py-1.5">
                  <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide">Code</span>
                  <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide">Class Label</span>
                </div>
                {preprocSettings.annotations.map((a, idx) => (
                  <div key={idx} className="grid grid-cols-[1fr_1.8fr] items-center px-3 py-1.5 gap-3 border-b border-gray-100 last:border-b-0">
                    <span className="font-mono text-sm text-gray-600">{a.code}</span>
                    <span className="text-sm text-gray-600">{a.label}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Model Evaluation Settings */}
      <div data-tour="model-eval-settings-section">
        <div className="flex items-center gap-2 mb-4">
          <Cpu className="w-4 h-4 text-gray-500" />
          <h3 className="text-sm font-semibold text-gray-700">Model Evaluation</h3>
        </div>
        <div className="space-y-4 pl-6">
          <div>
            <label className="block text-xs font-semibold text-gray-500 mb-2 uppercase">Model</label>
            <div className="flex gap-2 mb-3">
              {[['lda', 'LDA'], ['logreg', 'Logistic Regression'], ['svm', 'SVM']].map(([key, label]) => (
                <span key={key} className={`px-3 py-1.5 rounded-sm text-sm border ${
                  modelSettings.model === key
                    ? 'bg-[#0078D4] border-[#0078D4] text-white font-medium'
                    : 'bg-gray-50 border-gray-200 text-gray-400'
                }`}>{label}</span>
              ))}
            </div>
            {Object.keys(modelSettings.params).length > 0 && (
              <div className="grid grid-cols-2 gap-x-6 gap-y-1 pl-1">
                {Object.entries(modelSettings.params).map(([k, v]) => (
                  <div key={k} className="flex justify-between items-center">
                    <span className="text-[11px] text-gray-500 capitalize">{k}</span>
                    <span className="text-[11px] font-mono text-gray-700">{String(v)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
          <div>
            <label className="block text-xs font-semibold text-gray-500 mb-1 uppercase">Cross-Validation Folds</label>
            <input readOnly value={modelSettings.cvFolds} placeholder="—"
              className="w-24 border border-gray-200 rounded-sm px-3 py-2 font-mono text-sm bg-gray-50 text-gray-600 placeholder-gray-300" />
          </div>
          <div>
            <label className="block text-xs font-semibold text-gray-500 mb-2 uppercase">Decoders</label>
            {modelSettings.decoders.length === 0 ? (
              <p className="text-sm text-gray-300 italic">No decoders loaded</p>
            ) : (
              <div className="space-y-2">
                {modelSettings.decoders.map(d => (
                  <div key={d.id} className="border border-gray-200 rounded-sm px-3 py-2 bg-gray-50">
                    <p className="text-xs font-semibold text-gray-700 mb-1.5">{d.name}</p>
                    <div className="flex flex-wrap gap-1 mb-1">
                      {d.positive.map(c => (
                        <span key={c} className="text-[10px] bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded-sm">+{c}</span>
                      ))}
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {d.negative.map(c => (
                        <span key={c} className="text-[10px] bg-red-100 text-red-600 px-1.5 py-0.5 rounded-sm">−{c}</span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function WorkspaceNode2() {
  return (
    <div data-tour="step-data-ingestion" className="h-full flex flex-col items-center justify-center text-center">
      <div className="w-24 h-24 bg-blue-50 rounded-full flex items-center justify-center mb-6">
        <FolderOpen className="w-10 h-10 text-blue-600" />
      </div>
      <h3 className="text-xl font-medium mb-2">Select EEG Data File</h3>
      <p className="text-gray-500 text-sm mb-6 max-w-sm">Locate the .xdf or .fif file generated from the Stage 1 Functional Localizer.</p>
      <div className="flex items-center gap-2 border border-gray-300 rounded-sm bg-gray-50 p-2 w-96 mb-4">
        <span className="text-gray-400 text-sm flex-1 text-left px-2 font-mono">Waiting for file...</span>
        <button className="bg-white border border-gray-300 px-4 py-1 text-sm rounded-sm hover:bg-gray-100">Browse</button>
      </div>
    </div>
  );
}

function WorkspaceNode3Run({ onRun }) {
  return (
    <div data-tour="step-preproc-run" className="h-full flex flex-col items-center justify-center text-center">
      <div className="w-24 h-24 bg-blue-50 rounded-full flex items-center justify-center mb-6">
        <Play className="w-10 h-10 text-blue-600" />
      </div>
      <h3 className="text-xl font-medium mb-2">Ready to Preprocess</h3>
      <p className="text-gray-500 text-sm mb-6 max-w-sm">Settings configured. Click Start to begin the preprocessing pipeline.</p>
      <button
        onClick={onRun}
        className="bg-[#0078D4] hover:bg-[#006CBE] text-white px-8 py-2.5 rounded-sm text-sm font-medium transition-colors flex items-center gap-2"
      >
        <Play className="w-4 h-4" /> Start Preprocessing
      </button>
    </div>
  );
}

const DEFAULT_ANNOTATIONS = [
  { code: '11', label: 'Red' },
  { code: '12', label: 'Green' },
  { code: '13', label: 'Yellow' },
  { code: '21', label: 'Kitchen' },
  { code: '22', label: 'Living Room' },
  { code: '23', label: 'Bathroom' },
];


// Preprocessing stages and timing
const PREP_STAGES = [
  { id: 'epoching',     label: 'Epoching',     detail: 'Segmenting continuous signal' },
  { id: 'filtering',   label: 'Filtering',    detail: 'Band-pass 1–40 Hz' },
  { id: 'ica',         label: 'ICA',          detail: 'Decomposing components' },
  { id: 'normalizing', label: 'Normalizing',  detail: 'Z-score baseline correction' },
];
const STAGE_DURATIONS = [800, 900, 700]; // ms each stage runs before completing (last stage pauses)

function WorkspaceNode2Progress({ onDone, startFrom = 0 }) {
  const initialStatuses = PREP_STAGES.map((_, i) =>
    i < startFrom ? 'done' : i === startFrom ? 'running' : 'pending'
  );
  const [statuses, setStatuses] = useState(initialStatuses);

  useEffect(() => {
    const timers = [];
    let elapsed = 0;

    if (startFrom === 0) {
      // Run Epoching and Filtering, then pause at ICA
      [0, 1].forEach(i => {
        timers.push(setTimeout(() => {
          setStatuses(prev => {
            const next = [...prev];
            next[i] = 'done';
            next[i + 1] = i + 1 === 2 ? 'paused' : 'running';
            return next;
          });
          if (i === 1) timers.push(setTimeout(onDone, 600));
        }, elapsed += STAGE_DURATIONS[i]));
      });
    } else {
      // Resume from startFrom and run remaining stages to completion
      for (let i = startFrom; i < PREP_STAGES.length; i++) {
        const dur = 900;
        timers.push(setTimeout(() => {
          setStatuses(prev => {
            const next = [...prev];
            next[i] = 'done';
            if (i + 1 < PREP_STAGES.length) next[i + 1] = 'running';
            return next;
          });
          if (i === PREP_STAGES.length - 1) timers.push(setTimeout(onDone, 400));
        }, elapsed += dur));
      }
    }

    return () => timers.forEach(clearTimeout);
  }, []);

  return (
    <div className="h-full flex flex-col items-center justify-center px-8">
      <p className="text-sm text-gray-500 mb-10">Running preprocessing pipeline…</p>

      {/* Stage path */}
      <div className="w-full max-w-xl">
        {/* Connecting track */}
        <div className="relative flex items-start justify-between mb-2">
          {/* Background line */}
          <div className="absolute top-4 left-4 right-4 h-0.5 bg-gray-200" />
          {/* Progress line */}
          <div
            className="absolute top-4 left-4 h-0.5 bg-blue-500 transition-all duration-500"
            style={{
              width: `calc(${
                statuses.filter(s => s === 'done').length /
                (PREP_STAGES.length - 1) * 100
              }% - 2rem)`,
            }}
          />

          {PREP_STAGES.map((stage, i) => {
            const status = statuses[i];
            const circleClass =
              status === 'done'    ? 'bg-green-500 border-green-500 text-white' :
              status === 'running' ? 'bg-blue-500 border-blue-500 text-white animate-pulse' :
              status === 'paused'  ? 'bg-amber-400 border-amber-400 text-white' :
                                     'bg-white border-gray-300 text-gray-400';

            return (
              <div key={stage.id} className="relative flex flex-col items-center z-10 w-20">
                <div className={`w-8 h-8 rounded-full border-2 flex items-center justify-center transition-all duration-300 ${circleClass}`}>
                  {status === 'done'
                    ? <Check className="w-4 h-4" />
                    : <span className="text-[10px] font-bold">{i + 1}</span>}
                </div>
                <span className={`mt-2 text-xs font-semibold text-center ${
                  status === 'done' ? 'text-green-600' :
                  status === 'running' ? 'text-blue-600' :
                  status === 'paused' ? 'text-amber-600' : 'text-gray-400'
                }`}>{stage.label}</span>
                <span className="mt-0.5 text-[10px] text-gray-400 text-center leading-tight">{stage.detail}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// Stubbed ICA suggestions: id → { artifact, suggestedReject }
const ICA_SUGGESTIONS = {
  2:  { artifact: 'Eye blink (EOG)',  suggestedReject: true },
  5:  { artifact: 'Muscle (EMG)',     suggestedReject: true },
  8:  { artifact: 'Line noise (60Hz)', suggestedReject: true },
};

function WorkspaceNode2ICA() {
  const initial = Array.from({ length: 12 }, (_, i) => {
    const id = i + 1;
    const suggestion = ICA_SUGGESTIONS[id];
    return {
      id,
      artifact: suggestion?.artifact ?? 'Neural',
      suggestedReject: suggestion?.suggestedReject ?? false,
      rejected: suggestion?.suggestedReject ?? false, // start at suggested value
    };
  });
  const [components, setComponents] = useState(initial);

  const toggle = (id) => {
    setComponents(prev => prev.map(c => c.id === id ? { ...c, rejected: !c.rejected } : c));
  };

  const rejectedCount = components.filter(c => c.rejected).length;

  return (
    <div data-tour="step-ica">
      <div className="flex justify-between items-end mb-4">
        <p className="text-sm text-gray-500">
          Review suggested component classifications. Override any decision before confirming.
        </p>
        <span className="text-xs font-mono text-gray-500">{rejectedCount} Rejected</span>
      </div>
      <div className="grid grid-cols-4 gap-3">
        {components.map(c => {
          const isOverridden = c.rejected !== c.suggestedReject;
          return (
            <div
              key={c.id}
              className={`border rounded-sm p-3 ${
                c.rejected ? 'border-red-300 bg-red-50' : 'border-gray-200 bg-white'
              }`}
            >
              <div className="flex justify-between items-center mb-1">
                <span className="text-xs font-bold font-mono text-gray-600">
                  IC{c.id.toString().padStart(2, '0')}
                </span>
                {isOverridden && (
                  <span className="text-[9px] font-bold text-amber-600 uppercase tracking-wide">Override</span>
                )}
              </div>

              {/* Artifact label */}
              <p className={`text-[10px] mb-2 truncate ${c.rejected ? 'text-red-500 font-semibold' : 'text-gray-400'}`}>
                {c.artifact}
              </p>

              {/* Fake topoplot + timeseries */}
              <div className="h-16 w-full flex items-center justify-center opacity-60 mb-2">
                <div className="w-10 h-10 rounded-full border border-gray-300 bg-gradient-to-br from-blue-100 via-yellow-50 to-red-100 shrink-0" />
                <svg className="w-14 h-7 ml-2" viewBox="0 0 100 40">
                  <polyline
                    points="0,20 10,15 20,25 30,10 40,30 50,20 60,18 70,22 80,5 90,35 100,20"
                    fill="none" stroke="currentColor" strokeWidth="1.5"
                    className={c.rejected ? 'text-red-400' : 'text-gray-400'}
                  />
                </svg>
              </div>

              {/* Suggested badge + toggle */}
              <div className="flex items-center justify-between gap-1">
                <span className={`text-[9px] px-1.5 py-0.5 rounded-sm font-semibold uppercase ${
                  c.suggestedReject
                    ? 'bg-red-100 text-red-600'
                    : 'bg-gray-100 text-gray-500'
                }`}>
                  {c.suggestedReject ? 'Sugg: Reject' : 'Sugg: Keep'}
                </span>
                <button
                  onClick={() => toggle(c.id)}
                  className={`text-[10px] uppercase px-2 py-0.5 rounded-sm font-bold transition-colors ${
                    c.rejected
                      ? 'bg-red-200 text-red-800 hover:bg-red-300'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {c.rejected ? 'Reject' : 'Keep'}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function WorkspaceNode2Complete() {
  return (
    <div className="h-full flex flex-col items-center justify-center text-center px-8">
      <div className="w-16 h-16 bg-green-50 rounded-full flex items-center justify-center mb-5 border border-green-100">
        <Check className="w-8 h-8 text-green-500" />
      </div>
      <h3 className="text-xl font-medium text-gray-800 mb-2">Preprocessing Complete</h3>
      <p className="text-sm text-gray-500 mb-8 max-w-sm">
        All pipeline stages finished successfully. Cleaned epochs are ready. You may download them before proceeding to model evaluation.
      </p>

      <div className="w-full max-w-sm space-y-3">
        {/* Stats summary */}
        <div className="bg-gray-50 border border-gray-200 rounded-sm p-4 text-left text-sm">
          <div className="grid grid-cols-2 gap-y-2">
            <span className="text-gray-500">Epochs retained</span>
            <span className="font-mono text-right font-medium">342 / 360</span>
            <span className="text-gray-500">ICA components removed</span>
            <span className="font-mono text-right font-medium">3</span>
            <span className="text-gray-500">Classes</span>
            <span className="font-mono text-right font-medium">6</span>
          </div>
        </div>

        <button className="w-full bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 py-2 rounded-sm text-sm font-medium flex items-center justify-center gap-2 transition-colors">
          <Download className="w-4 h-4" /> Download Cleaned Epochs (.fif)
        </button>
      </div>
    </div>
  );
}

const INITIAL_DECODERS = [
  { id: 1, name: 'Red',    positive: ['Red'],    negative: ['Green', 'Yellow'] },
  { id: 2, name: 'Green',  positive: ['Green'],  negative: ['Red',   'Yellow'] },
  { id: 3, name: 'Yellow', positive: ['Yellow'], negative: ['Red',   'Green']  },
];

function WorkspaceNode4Run({ onRun }) {
  return (
    <div data-tour="step-eval-run" className="h-full flex flex-col items-center justify-center text-center">
      <div className="w-24 h-24 bg-purple-50 rounded-full flex items-center justify-center mb-6">
        <Cpu className="w-10 h-10 text-purple-600" />
      </div>
      <h3 className="text-xl font-medium mb-2">Ready to Evaluate</h3>
      <p className="text-gray-500 text-sm mb-6 max-w-sm">Settings configured. Click Start to run cross-validation across all decoders.</p>
      <button
        onClick={onRun}
        className="bg-[#0078D4] hover:bg-[#006CBE] text-white px-8 py-2.5 rounded-sm text-sm font-medium transition-colors flex items-center gap-2"
      >
        <Play className="w-4 h-4" /> Start Evaluation
      </button>
    </div>
  );
}

// --- CV PROGRESS ---

function WorkspaceNode3CVProgress({ decoders, cvFolds, onDone }) {
  const [foldProgress, setFoldProgress] = useState(decoders.map(() => 0));

  useEffect(() => {
    const timers = [];
    let lastFinish = 0;

    decoders.forEach((_, di) => {
      const stagger = di * 120;
      for (let fold = 1; fold <= cvFolds; fold++) {
        const delay = stagger + fold * 450;
        if (delay > lastFinish) lastFinish = delay;
        timers.push(setTimeout(() => {
          setFoldProgress(prev => {
            const next = [...prev];
            next[di] = fold;
            return next;
          });
        }, delay));
      }
    });

    timers.push(setTimeout(onDone, lastFinish + 700));
    return () => timers.forEach(clearTimeout);
  }, []);

  const totalFolds = decoders.length * cvFolds;
  const completedFolds = foldProgress.reduce((s, f) => s + f, 0);
  const overallPct = Math.round((completedFolds / totalFolds) * 100);

  return (
    <div className="h-full flex flex-col items-center justify-center px-8">
      <p className="text-sm text-gray-500 mb-2">Training decoders in parallel…</p>
      <p className="text-xs font-mono text-gray-400 mb-8">{overallPct}% complete</p>

      <div className="w-full max-w-2xl">
        {/* Overall bar */}
        <div className="h-1 w-full bg-gray-100 rounded-full mb-10 overflow-hidden">
          <div
            className="h-full bg-[#0078D4] rounded-full transition-all duration-300"
            style={{ width: `${overallPct}%` }}
          />
        </div>

        {/* Per-decoder cards */}
        <div className="grid grid-cols-3 gap-4">
          {decoders.map((dec, di) => {
            const done = foldProgress[di] >= cvFolds;
            const pct = (foldProgress[di] / cvFolds) * 100;
            return (
              <div key={dec.id} className={`border rounded-sm p-4 transition-colors ${done ? 'border-green-200 bg-green-50/40' : 'border-gray-200 bg-white'}`}>
                <div className="flex justify-between items-center mb-3">
                  <span className="text-sm font-medium text-gray-700 truncate">{dec.name}</span>
                  {done
                    ? <Check className="w-4 h-4 text-green-500 shrink-0" />
                    : <span className="text-[10px] font-mono text-blue-500 shrink-0">▶</span>}
                </div>
                <div className="h-1.5 w-full bg-gray-100 rounded-full overflow-hidden mb-2">
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${done ? 'bg-green-500' : 'bg-[#0078D4]'}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <p className="text-[10px] font-mono text-gray-400">
                  {done ? 'Complete' : `Fold ${foldProgress[di]} / ${cvFolds}`}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// --- RESULTS: STUB DATA & HELPERS ---

// Stub AUC values per decoder index
const STUB_AUCS    = [0.82, 0.88, 0.75, 0.79, 0.85, 0.71];
const STUB_PEAKS   = [450, 380, 310, 420, 395, 460];
const STUB_BAL_ACC = [0.79, 0.84, 0.72, 0.76, 0.82, 0.68];

const DECODER_COLORS = ['#C41E3A', '#228B22', '#B8860B', '#0078D4', '#8B008B', '#CC5500'];

// Gaussian-ish AUC curve: 21 points from -200ms to 800ms
function stubAUCCurve(peakMs, maxAUC) {
  return Array.from({ length: 21 }, (_, k) => {
    const t = -200 + k * 50;
    const signal = (maxAUC - 0.5) * Math.exp(-((t - peakMs) ** 2) / (2 * 150 ** 2));
    return { t, auc: Math.max(0.48, 0.5 + signal) };
  });
}

// Interpolate AUC from a curve at an arbitrary time
function aucAtTime(curve, t) {
  const clamped = Math.max(-200, Math.min(800, t));
  const idx = (clamped + 200) / 50;
  const lo = Math.floor(idx), hi = Math.min(20, Math.ceil(idx));
  if (lo === hi) return curve[lo].auc;
  const frac = idx - lo;
  return curve[lo].auc * (1 - frac) + curve[hi].auc * frac;
}

// SVG chart: converts time → x pixel and AUC → y pixel
const CHART = { w: 460, h: 120, pad: { l: 32, r: 8, t: 10, b: 24 } };
function timeToX(t) {
  return CHART.pad.l + ((t + 200) / 1000) * (CHART.w - CHART.pad.l - CHART.pad.r);
}
function aucToY(auc) {
  return CHART.pad.t + (1 - auc) * (CHART.h - CHART.pad.t - CHART.pad.b);
}
function xToTime(x) {
  const frac = (x - CHART.pad.l) / (CHART.w - CHART.pad.l - CHART.pad.r);
  return Math.round((-200 + frac * 1000) / 50) * 50; // snap to 50ms
}

function WorkspaceNode3Results({ decoders, confirmedTimepoint, setConfirmedTimepoint, forceDecoderTab }) {
  const suggestedTimepoint = Math.round(
    decoders.reduce((sum, _, i) => sum + STUB_PEAKS[i % STUB_PEAKS.length], 0) / Math.max(1, decoders.length)
  );
  const [activeTab, setActiveTab] = useState('summary');
  const [selectedTime, setSelectedTime] = useState(suggestedTimepoint);
  const [decoderTime, setDecoderTime] = useState(null); // per-decoder override

  // Reset decoder-specific timepoint when tab changes
  useEffect(() => { setDecoderTime(null); }, [activeTab]);

  // Tutorial can force switch to a specific decoder tab or back to summary
  useEffect(() => {
    if (forceDecoderTab === 'summary') {
      setActiveTab('summary');
    } else if (forceDecoderTab !== null) {
      setActiveTab(forceDecoderTab);
    }
  }, [forceDecoderTab]);

  const activeDecoder = decoders.find(d => d.id === activeTab);

  // Build per-decoder curves once
  const curves = decoders.map((_, i) =>
    stubAUCCurve(STUB_PEAKS[i % STUB_PEAKS.length], STUB_AUCS[i % STUB_AUCS.length])
  );

  // SVG click handler → snap to nearest 50ms
  const handleChartClick = (e, setter) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (CHART.w / rect.width);
    const t = Math.max(-200, Math.min(800, xToTime(x)));
    setter(t);
  };

  // Tab bar
  const TabBar = () => (
    <div data-tour="decoder-tab" className="flex border-b border-gray-200 mb-3 overflow-x-auto shrink-0">
      <button onClick={() => setActiveTab('summary')}
        className={`px-4 py-2 text-sm font-medium whitespace-nowrap border-b-2 transition-colors ${activeTab === 'summary' ? 'border-[#0078D4] text-[#0078D4]' : 'border-transparent text-gray-500 hover:text-gray-700'}`}>
        Summary
      </button>
      {decoders.map((d, i) => (
        <button key={d.id} onClick={() => setActiveTab(d.id)}
          className={`px-4 py-2 text-sm font-medium whitespace-nowrap border-b-2 transition-colors flex items-center gap-1.5 ${activeTab === d.id ? 'border-[#0078D4] text-[#0078D4]' : 'border-transparent text-gray-500 hover:text-gray-700'}`}>
          <span className="w-2 h-2 rounded-full shrink-0" style={{ background: DECODER_COLORS[i % DECODER_COLORS.length] }} />
          {d.name}
        </button>
      ))}
    </div>
  );

  // ── SUMMARY TAB ──
  if (activeTab === 'summary') {
    const aucsAtSelected = decoders.map((_, i) => aucAtTime(curves[i], selectedTime));
    const avgAUC = aucsAtSelected.reduce((s, v) => s + v, 0) / Math.max(1, aucsAtSelected.length);
    const devFromSuggested = Math.abs(selectedTime - suggestedTimepoint);

    return (
      <div data-tour="step-eval-results" className="h-full flex flex-col min-h-0">
        <TabBar />
        <div className="overflow-y-auto flex-1">
          {/* AUC chart + stats panel */}
          <div className="flex gap-4 mb-5">
            {/* SVG chart */}
            <div className="flex-1 border border-gray-200 rounded-sm bg-white p-2 overflow-hidden">
              <p className="text-[10px] font-semibold text-gray-500 uppercase mb-1 px-1">AUC over Time — All Decoders</p>
              <svg
                viewBox={`0 0 ${CHART.w} ${CHART.h}`}
                className="w-full cursor-crosshair"
                style={{ height: CHART.h }}
                onClick={e => handleChartClick(e, setSelectedTime)}
              >
                {/* Chance line */}
                <line x1={CHART.pad.l} y1={aucToY(0.5)} x2={CHART.w - CHART.pad.r} y2={aucToY(0.5)}
                  stroke="#9ca3af" strokeWidth="0.8" strokeDasharray="3,3" />
                <text x={CHART.pad.l - 2} y={aucToY(0.5) + 3} textAnchor="end" fontSize="7" fill="#9ca3af">0.5</text>
                <text x={CHART.pad.l - 2} y={aucToY(1.0) + 3} textAnchor="end" fontSize="7" fill="#9ca3af">1.0</text>

                {/* Decoder lines */}
                {curves.map((curve, i) => (
                  <polyline key={i}
                    points={curve.map(pt => `${timeToX(pt.t)},${aucToY(pt.auc)}`).join(' ')}
                    fill="none" stroke={DECODER_COLORS[i % DECODER_COLORS.length]} strokeWidth="1.5" strokeLinejoin="round" opacity="0.85" />
                ))}

                {/* Selected time marker */}
                <line x1={timeToX(selectedTime)} y1={CHART.pad.t} x2={timeToX(selectedTime)} y2={CHART.h - CHART.pad.b}
                  stroke="#0078D4" strokeWidth="1.5" />

                {/* X-axis ticks */}
                {[-200, 0, 200, 400, 600, 800].map(t => (
                  <g key={t}>
                    <line x1={timeToX(t)} y1={CHART.h - CHART.pad.b} x2={timeToX(t)} y2={CHART.h - CHART.pad.b + 3} stroke="#d1d5db" strokeWidth="0.8" />
                    <text x={timeToX(t)} y={CHART.h - 4} textAnchor="middle" fontSize="7" fill="#9ca3af">{t}ms</text>
                  </g>
                ))}
              </svg>

              {/* Legend */}
              <div className="flex flex-wrap gap-3 px-1 pt-1">
                {decoders.map((d, i) => (
                  <div key={d.id} className="flex items-center gap-1">
                    <span className="w-4 h-0.5 inline-block rounded-full" style={{ background: DECODER_COLORS[i % DECODER_COLORS.length] }} />
                    <span className="text-[10px] text-gray-500">{d.name}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Stats panel */}
            <div data-tour="suggested-timepoint" className="w-44 shrink-0 border border-gray-200 rounded-sm bg-white p-3 flex flex-col gap-2">
              <div>
                <p className="text-[10px] font-semibold text-gray-500 uppercase mb-1">Selected</p>
                <p className={`text-base font-mono font-bold ${devFromSuggested > 50 ? 'text-amber-500' : 'text-[#0078D4]'}`}>
                  {selectedTime} ms
                </p>
                {devFromSuggested > 50 && (
                  <p className="text-[9px] text-amber-500">±{devFromSuggested}ms from suggested</p>
                )}
              </div>

              <div className="border-t border-gray-100 pt-2 space-y-1 flex-1">
                {decoders.map((d, i) => (
                  <div key={d.id} className="flex justify-between items-center">
                    <div className="flex items-center gap-1 min-w-0">
                      <span className="w-2 h-2 rounded-full shrink-0" style={{ background: DECODER_COLORS[i % DECODER_COLORS.length] }} />
                      <span className="text-[10px] text-gray-600 truncate">{d.name}</span>
                    </div>
                    <span className="text-[10px] font-mono font-semibold text-gray-700 shrink-0 ml-1">
                      {aucsAtSelected[i].toFixed(2)}
                    </span>
                  </div>
                ))}
                <div className="border-t border-gray-100 pt-1 flex justify-between items-center">
                  <span className="text-[10px] text-gray-500 font-semibold">Avg</span>
                  <span className="text-[10px] font-mono font-bold text-gray-700">{avgAUC.toFixed(2)}</span>
                </div>
              </div>

              <div className="border-t border-gray-100 pt-2">
                <p className="text-[9px] text-gray-400 mb-1.5">Suggested: {suggestedTimepoint}ms (avg peak)</p>
                <div className="flex flex-col gap-1.5">
                  <button
                    onClick={() => setConfirmedTimepoint(selectedTime)}
                    className={`w-full py-1 text-[10px] font-semibold text-white rounded-sm transition-colors ${confirmedTimepoint === selectedTime ? 'bg-green-600' : 'bg-[#0078D4] hover:bg-[#006CBE]'}`}
                  >
                    {confirmedTimepoint === selectedTime ? "Timepoint Confirmed" : "Confirm Timepoint"}
                  </button>
                  {selectedTime !== suggestedTimepoint && (
                    <button
                      onClick={() => setSelectedTime(suggestedTimepoint)}
                      className="w-full py-1 text-[10px] font-semibold text-[#0078D4] border border-[#0078D4]/30 hover:bg-blue-50 rounded-sm transition-colors"
                    >
                      Reset to suggested
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Existing decoder table */}
          <div className="border border-gray-200 rounded-sm overflow-hidden">
            <table className="w-full text-left text-sm">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-2.5 text-xs font-semibold text-gray-600">Decoder</th>
                  <th className="px-4 py-2.5 text-xs font-semibold text-gray-600">Positive</th>
                  <th className="px-4 py-2.5 text-xs font-semibold text-gray-600">Negative</th>
                  <th className="px-4 py-2.5 text-xs font-semibold text-gray-600 text-right font-mono">AUC</th>
                  <th className="px-4 py-2.5 text-xs font-semibold text-gray-600 text-right font-mono">Bal. Acc</th>
                  <th className="px-4 py-2.5 text-xs font-semibold text-gray-600 text-right font-mono">Peak (ms)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {decoders.map((d, i) => {
                  const auc = STUB_AUCS[i % STUB_AUCS.length];
                  return (
                    <tr key={d.id} onClick={() => setActiveTab(d.id)} className="hover:bg-blue-50/40 cursor-pointer transition-colors">
                      <td className="px-4 py-2.5 font-medium text-gray-800 flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: DECODER_COLORS[i % DECODER_COLORS.length] }} />
                        {d.name}
                      </td>
                      <td className="px-4 py-2.5">
                        <div className="flex flex-wrap gap-1">
                          {d.positive.map(c => <span key={c} className="text-[10px] bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded-sm">{c}</span>)}
                          {d.positive.length === 0 && <span className="text-[10px] text-gray-400">—</span>}
                        </div>
                      </td>
                      <td className="px-4 py-2.5">
                        <div className="flex flex-wrap gap-1">
                          {d.negative.map(c => <span key={c} className="text-[10px] bg-red-100 text-red-600 px-1.5 py-0.5 rounded-sm">{c}</span>)}
                          {d.negative.length === 0 && <span className="text-[10px] text-gray-400">—</span>}
                        </div>
                      </td>
                      <td className={`px-4 py-2.5 font-mono text-right font-semibold ${auc >= 0.70 ? 'text-green-600' : 'text-red-500'}`}>{auc.toFixed(2)}</td>
                      <td className="px-4 py-2.5 font-mono text-right text-gray-600">{STUB_BAL_ACC[i % STUB_BAL_ACC.length].toFixed(2)}</td>
                      <td className="px-4 py-2.5 font-mono text-right text-gray-600">{STUB_PEAKS[i % STUB_PEAKS.length]}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  }

  // ── PER-DECODER TAB ──
  if (activeDecoder) {
    const i = decoders.indexOf(activeDecoder);
    const suggestedT = STUB_PEAKS[i % STUB_PEAKS.length];
    const effectiveT = decoderTime ?? suggestedT;
    const curve = curves[i];
    const color = DECODER_COLORS[i % DECODER_COLORS.length];

    // Crosshair position as percentage inside TGM
    const tgmPct = `${((effectiveT + 200) / 1000) * 100}%`;

    return (
      <div data-tour="decoder-detail-view" className="h-full flex flex-col min-h-0">
        <TabBar />

        {/* Header */}
        <div className="flex items-center gap-2 mb-3 pb-2 border-b border-gray-100 shrink-0 flex-wrap">
          {activeDecoder.positive.map(c => <span key={c} className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-sm font-medium">{c}</span>)}
          <span className="text-xs text-gray-400">vs</span>
          {activeDecoder.negative.map(c => <span key={c} className="text-xs bg-red-100 text-red-600 px-2 py-0.5 rounded-sm font-medium">{c}</span>)}
          <span className="ml-auto text-xs font-mono bg-gray-100 px-2 py-0.5 rounded-sm text-gray-500">
            Peak AUC: {STUB_AUCS[i % STUB_AUCS.length].toFixed(2)} @ {suggestedT}ms
          </span>
        </div>

        {/* AUC curve */}
        <div className="border border-gray-200 rounded-sm bg-white p-2 mb-3 shrink-0">
          <p className="text-[10px] font-semibold text-gray-500 uppercase mb-1 px-1">AUC over Time</p>
          <svg
            viewBox={`0 0 ${CHART.w} ${CHART.h}`}
            className="w-full cursor-crosshair"
            style={{ height: CHART.h }}
            onClick={e => handleChartClick(e, setDecoderTime)}
          >
            {/* Chance line */}
            <line x1={CHART.pad.l} y1={aucToY(0.5)} x2={CHART.w - CHART.pad.r} y2={aucToY(0.5)}
              stroke="#9ca3af" strokeWidth="0.8" strokeDasharray="3,3" />
            <text x={CHART.pad.l - 2} y={aucToY(0.5) + 3} textAnchor="end" fontSize="7" fill="#9ca3af">0.5</text>
            <text x={CHART.pad.l - 2} y={aucToY(1.0) + 3} textAnchor="end" fontSize="7" fill="#9ca3af">1.0</text>

            {/* AUC line */}
            <polyline
              points={curve.map(pt => `${timeToX(pt.t)},${aucToY(pt.auc)}`).join(' ')}
              fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" />

            {/* Suggested timepoint — dotted */}
            <line x1={timeToX(suggestedT)} y1={CHART.pad.t} x2={timeToX(suggestedT)} y2={CHART.h - CHART.pad.b}
              stroke="#6b7280" strokeWidth="1" strokeDasharray="4,3" />
            <text x={timeToX(suggestedT) + 3} y={CHART.pad.t + 9} fontSize="7" fill="#6b7280">suggested</text>

            {/* Selected timepoint — solid */}
            <line x1={timeToX(effectiveT)} y1={CHART.pad.t} x2={timeToX(effectiveT)} y2={CHART.h - CHART.pad.b}
              stroke="#0078D4" strokeWidth="1.5" />

            {/* Dot at intersection */}
            <circle cx={timeToX(effectiveT)} cy={aucToY(aucAtTime(curve, effectiveT))} r="3"
              fill="#0078D4" stroke="white" strokeWidth="1" />

            {/* X-axis ticks */}
            {[-200, 0, 200, 400, 600, 800].map(t => (
              <g key={t}>
                <line x1={timeToX(t)} y1={CHART.h - CHART.pad.b} x2={timeToX(t)} y2={CHART.h - CHART.pad.b + 3} stroke="#d1d5db" strokeWidth="0.8" />
                <text x={timeToX(t)} y={CHART.h - 4} textAnchor="middle" fontSize="7" fill="#9ca3af">{t}ms</text>
              </g>
            ))}
          </svg>

          <div className="flex items-center justify-between px-1 pt-0.5">
            <span className="text-[10px] text-[#0078D4] font-mono">T = {effectiveT}ms  |  AUC = {aucAtTime(curve, effectiveT).toFixed(3)}</span>
            {decoderTime !== null && (
              <button onClick={() => setDecoderTime(null)}
                className="text-[10px] text-gray-400 hover:text-[#0078D4] transition-colors">
                Reset to suggested
              </button>
            )}
          </div>
        </div>

        {/* TGM */}
        <div className="flex-1 flex flex-col gap-4 min-h-0">
          <p className="text-xs font-semibold text-gray-500 uppercase mb-1.5">Temporal Generalization Matrix</p>
          <div className="flex-1 bg-gradient-to-br from-[#0a3663] via-white to-[#8b0000] border border-gray-200 rounded-sm relative cursor-crosshair">
            <span className="absolute -left-6 top-1/2 -rotate-90 text-[10px] text-gray-400">Test Time</span>
            <span className="absolute bottom-[-18px] left-1/2 -translate-x-1/2 text-[10px] text-gray-400">Train Time</span>
            {/* Dynamic crosshair at effectiveT */}
            <div className="absolute top-0 bottom-0 border-l border-white/90 border-dashed mix-blend-difference pointer-events-none"
              style={{ left: tgmPct }} />
            <div className="absolute left-0 right-0 border-t border-white/90 border-dashed mix-blend-difference pointer-events-none"
              style={{ top: tgmPct }} />
            <div className="absolute w-3 h-3 border-2 border-white rounded-full -translate-x-1.5 -translate-y-1.5 mix-blend-difference pointer-events-none"
              style={{ left: tgmPct, top: tgmPct }} />
          </div>
        </div>
      </div>
    );
  }

  return null;
}

function WorkspaceNode4() {
  return (
    <div data-tour="step-deploy" className="max-w-lg mx-auto py-8 text-center">
      <div className="w-16 h-16 bg-green-50 rounded-full flex items-center justify-center mb-6 mx-auto border border-green-100">
        <Check className="w-8 h-8 text-green-600" />
      </div>
      <h3 className="text-xl font-medium mb-2">Pipeline Finalized</h3>
      <p className="text-gray-500 text-sm mb-8">All parameters and decoder weights are locked. Ready to deploy models to the live streaming environment.</p>

      <div className="bg-gray-50 border border-gray-200 rounded-sm p-4 text-left mb-8">
        <h4 className="text-xs font-semibold text-gray-500 uppercase mb-3">Deployment Summary</h4>
        <div className="grid grid-cols-2 gap-y-2 text-sm">
          <span className="text-gray-500">Decoders trained:</span>
          <span className="font-mono text-right font-medium">6 (One-vs-Rest)</span>
          <span className="text-gray-500">Mean target AUC:</span>
          <span className="font-mono text-right font-medium">0.81</span>
          <span className="text-gray-500">Spatial Filters (ICA):</span>
          <span className="font-mono text-right font-medium">Frozen (10 comps)</span>
        </div>
      </div>
    </div>
  );
}

// --- FINAL TRAINING PROGRESS ---

function WorkspaceNode4Progress({ decoders, onDone }) {
  const [progress, setProgress] = useState(decoders.map(() => 0));

  useEffect(() => {
    const timers = [];
    let lastFinish = 0;

    decoders.forEach((_, di) => {
      const stagger = di * 250;
      const delay = stagger + 800; // Single step simulating training over full dataset
      if (delay > lastFinish) lastFinish = delay;
      timers.push(setTimeout(() => {
        setProgress(prev => {
          const next = [...prev];
          next[di] = 1;
          return next;
        });
      }, delay));
    });

    timers.push(setTimeout(onDone, lastFinish + 700));
    return () => timers.forEach(clearTimeout);
  }, []);

  const totalDecoders = decoders.length;
  const completedDecoders = progress.reduce((s, p) => s + p, 0);
  const overallPct = Math.round((completedDecoders / totalDecoders) * 100);

  return (
    <div className="h-full flex flex-col items-center justify-center px-8">
      <p className="text-sm text-gray-500 mb-2">Training final decoders on full dataset…</p>
      <p className="text-xs font-mono text-gray-400 mb-8">{overallPct}% complete</p>

      <div className="w-full max-w-2xl">
        <div className="h-1 w-full bg-gray-100 rounded-full mb-10 overflow-hidden">
          <div
            className="h-full bg-[#0078D4] rounded-full transition-all duration-300"
            style={{ width: `${overallPct}%` }}
          />
        </div>

        <div className="grid grid-cols-3 gap-4">
          {decoders.map((dec, di) => {
            const done = progress[di] === 1;
            const pct = progress[di] * 100;
            return (
              <div key={dec.id} className={`border rounded-sm p-4 transition-colors ${done ? 'border-green-200 bg-green-50/40' : 'border-gray-200 bg-white'}`}>
                <div className="flex justify-between items-center mb-3">
                  <span className="text-sm font-medium text-gray-700 truncate">{dec.name}</span>
                  {done
                    ? <Check className="w-4 h-4 text-green-500 shrink-0" />
                    : <span className="text-[10px] font-mono text-blue-500 shrink-0">▶</span>}
                </div>
                <div className="h-1.5 w-full bg-gray-100 rounded-full overflow-hidden mb-2">
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${done ? 'bg-green-500' : 'bg-[#0078D4]'}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <p className="text-[10px] font-mono text-gray-400">
                  {done ? 'Complete' : 'Training...'}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// --- JOURNEY NODE ---

function JourneyNode({ title, nodeNum, activeNode, description, actionText, onAction, isLast, isDisabled, actionTourId }) {
  const isActive = activeNode === nodeNum;
  const isPast = activeNode > nodeNum;
  const isLocked = activeNode < nodeNum;

  let circleClass = "bg-white border-gray-300 text-gray-400";
  if (isPast) circleClass = "bg-green-500 border-green-500 text-white";
  if (isActive) circleClass = "bg-[#0078D4] border-[#0078D4] text-white shadow-[0_0_0_4px_rgba(0,120,212,0.15)]";

  return (
    <div className="relative pl-12 mb-8 last:mb-0">
      <div className={`absolute left-[7px] top-0 w-6 h-6 rounded-full border-2 flex items-center justify-center z-10 transition-all duration-300 ${circleClass}`}>
        {isPast ? <Check className="w-3.5 h-3.5" /> : <span className="text-[10px] font-bold">{nodeNum}</span>}
      </div>

      <div className={`transition-opacity duration-300 ${isLocked ? 'opacity-40' : 'opacity-100'}`}>
        <h4 className={`font-semibold ${isActive ? 'text-[#0078D4]' : 'text-gray-700'}`}>{title}</h4>

        {isActive && (
          <div className="mt-2 bg-gray-50 border border-gray-200 rounded-sm p-3 shadow-sm">
            <p className="text-xs text-gray-600 mb-3">{description}</p>
            {actionText && (
              <button
                data-tour={actionTourId}
                onClick={onAction}
                disabled={isDisabled}
                className={`w-full py-1.5 rounded-sm text-xs font-semibold transition-colors ${
                  isDisabled
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-[#0078D4] hover:bg-[#006CBE] text-white'
                }`}>
                {actionText}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
