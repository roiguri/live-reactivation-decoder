import { useState, useCallback, useEffect } from 'react';
import { FolderOpen, Play, FilePlus, FolderInput, ArrowLeft, Settings2, User } from 'lucide-react';
import { useWalkthroughAction, useWalkthrough } from '../walkthrough';

async function pickDirectory() {
  if (!window.showDirectoryPicker) return null;
  try {
    const handle = await window.showDirectoryPicker({ mode: 'read' });
    return handle.name;
  } catch {
    // user cancelled
    return null;
  }
}

export default function WelcomeScreen({ onStart, initialView = 'initial' }) {
  const [view, setView] = useState(initialView); // 'initial' | 'create' | 'load'

  // Register tutorial actions - these get called when user clicks "Next" in the tutorial
  useWalkthroughAction('welcome-click-create-new', useCallback(() => setView('create'), []));
  useWalkthroughAction('welcome-click-back', useCallback(() => setView('initial'), []));
  useWalkthroughAction('welcome-click-load-existing', useCallback(() => setView('load'), []));
  useWalkthroughAction('welcome-start-experiment', useCallback(() => onStart(), [onStart]));

  // Reverse actions for Back button
  useWalkthroughAction('welcome-restore-initial', useCallback(() => setView('initial'), []));
  useWalkthroughAction('welcome-restore-create', useCallback(() => setView('create'), []));
  useWalkthroughAction('welcome-restore-load', useCallback(() => setView('load'), []));

  return (
    <div className="h-full flex flex-col items-center justify-center p-8">
      <div className="bg-white shadow-sm border border-gray-200 p-10 rounded-md w-full max-w-2xl">
        {view === 'initial' && <InitialView onCreateNew={() => setView('create')} onLoadExisting={() => setView('load')} />}
        {view === 'create' && <CreateNewView onBack={() => setView('initial')} onStart={onStart} />}
        {view === 'load' && <LoadExistingView onBack={() => setView('initial')} onStart={onStart} />}
      </div>
    </div>
  );
}

// --- INITIAL VIEW ---

function InitialView({ onCreateNew, onLoadExisting }) {
  return (
    <div>
      <h1 className="text-3xl font-light mb-10 text-gray-800 text-center">Reactivation Decoder</h1>

      <div className="grid grid-cols-2 gap-6">
        <button
          data-tour="create-new-btn"
          onClick={onCreateNew}
          className="group flex flex-col items-center gap-4 p-8 border-2 border-gray-200 rounded-sm hover:border-[#0078D4] hover:bg-blue-50/40 transition-all"
        >
          <div className="w-14 h-14 rounded-full bg-blue-50 flex items-center justify-center group-hover:bg-blue-100 transition-colors">
            <FilePlus className="w-7 h-7 text-[#0078D4]" />
          </div>
          <div className="text-center">
            <div className="font-semibold text-gray-800 mb-1">Create New Experiment</div>
            <div className="text-xs text-gray-400">Configure settings and train a new decoder</div>
          </div>
        </button>

        <button
          data-tour="load-existing-btn"
          onClick={onLoadExisting}
          className="group flex flex-col items-center gap-4 p-8 border-2 border-gray-200 rounded-sm hover:border-gray-400 hover:bg-gray-50 transition-all"
        >
          <div className="w-14 h-14 rounded-full bg-gray-100 flex items-center justify-center group-hover:bg-gray-200 transition-colors">
            <FolderInput className="w-7 h-7 text-gray-600" />
          </div>
          <div className="text-center">
            <div className="font-semibold text-gray-800 mb-1">Load Existing Experiment</div>
            <div className="text-xs text-gray-400">Continue from a previously saved session</div>
          </div>
        </button>
      </div>
    </div>
  );
}

// --- CREATE NEW VIEW ---

function CreateNewView({ onBack, onStart }) {
  const [saveDir, setSaveDir] = useState('');
  const [expName, setExpName] = useState('');

  const canStart = saveDir.trim() !== '' && expName.trim() !== '';
  const fullPath = saveDir && expName ? `${saveDir}/${expName}` : null;

  const browse = async () => {
    const name = await pickDirectory();
    if (name) setSaveDir(name);
  };

  return (
    <div>
      <div className="flex items-center gap-3 mb-8">
        <button data-tour="back-arrow-btn" onClick={onBack} className="text-gray-400 hover:text-gray-700 transition-colors">
          <ArrowLeft className="w-4 h-4" />
        </button>
        <h1 className="text-xl font-medium text-gray-800">New Experiment</h1>
      </div>

      <div className="space-y-8">
        {/* Save Location */}
        <Section icon={<FolderOpen className="w-4 h-4" />} title="Save Location">
          <div className="space-y-2" data-tour="exp-location-input">
            <div className="flex">
              <input
                type="text"
                value={saveDir}
                onChange={e => setSaveDir(e.target.value)}
                placeholder="Select parent folder..."
                className="flex-1 border border-gray-300 rounded-l-sm px-3 py-1.5 text-sm focus:outline-none focus:border-blue-500 font-mono text-gray-600"
              />
              <button
                onClick={browse}
                className="bg-gray-200 border border-l-0 border-gray-300 px-3 rounded-r-sm hover:bg-gray-300"
              >
                <FolderOpen className="w-4 h-4 text-gray-600" />
              </button>
            </div>
            <div data-tour="exp-name-input">
              <label className="block text-xs text-gray-500 mb-1">Experiment Name</label>
              <input
                type="text"
                value={expName}
                onChange={e => setExpName(e.target.value)}
                placeholder="e.g. BMR_Exp_April"
                className="w-full border border-gray-300 rounded-sm px-3 py-1.5 text-sm focus:outline-none focus:border-blue-500 font-mono"
              />
            </div>
            {fullPath && (
              <p className="text-[11px] text-gray-400 font-mono pt-0.5">
                Will create: <span className="text-gray-600">{fullPath}/</span>
              </p>
            )}
          </div>
        </Section>

        {/* Settings Placeholders */}
        <Section icon={<Settings2 className="w-4 h-4" />} title="Pipeline Settings">
          <div data-tour="pipeline-settings" className="grid grid-cols-2 gap-4">
            <SettingsPlaceholder label="Preprocessing Settings" />
            <SettingsPlaceholder label="Model Evaluation Settings" />
          </div>
        </Section>

        {/* Subject */}
        <Section icon={<User className="w-4 h-4" />} title="Subject">
          <div data-tour="subject-section">
            <label className="block text-xs text-gray-500 mb-1">Subject ID</label>
            <input
              type="text"
              defaultValue="Subject_001"
              className="w-full border border-gray-300 rounded-sm px-3 py-1.5 text-sm focus:outline-none focus:border-blue-500 font-mono"
            />
          </div>
        </Section>
      </div>

      <div className="mt-8 pt-6 border-t border-gray-100 flex justify-end">
        <button
          data-tour="start-exp-btn"
          onClick={onStart}
          disabled={!canStart}
          className="bg-[#0078D4] hover:bg-[#006CBE] disabled:bg-gray-300 disabled:cursor-not-allowed text-white py-2 px-8 rounded-sm transition-colors text-sm font-medium flex items-center gap-2"
        >
          Start Experiment <Play className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

// --- LOAD EXISTING VIEW ---

const STUB_SUBJECTS = [
  { id: 'Subject_001', lastSession: '2026-04-10', trials: 3 },
  { id: 'Subject_003', lastSession: '2026-04-11', trials: 5 },
  { id: 'Subject_005', lastSession: '2026-04-12', trials: 2 },
  { id: 'Subject_007', lastSession: '2026-04-14', trials: 4 },
];

function LoadExistingView({ onBack, onStart }) {
  const [loadDir, setLoadDir] = useState('');
  const [selectedAction, setSelectedAction] = useState(null); // null | 'continue' | 'new'
  const [selectedSubject, setSelectedSubject] = useState(null);
  const [newSubjectId, setNewSubjectId] = useState('');
  const { isActive } = useWalkthrough();

  // For tutorial: pre-populate directory to show action cards
  useEffect(() => {
    if (isActive && loadDir === '') {
      setLoadDir('BMR_Experiment_2026');
    }
  }, [isActive, loadDir]);

  const dirPicked = loadDir.trim() !== '';

  const browse = async () => {
    const name = await pickDirectory();
    if (name) {
      setLoadDir(name);
      setSelectedAction(null);
      setSelectedSubject(null);
    }
  };

  const selectAction = (action) => {
    setSelectedAction(action);
    setSelectedSubject(null);
    setNewSubjectId('');
  };

  return (
    <div>
      <div className="flex items-center gap-3 mb-8">
        <button onClick={onBack} className="text-gray-400 hover:text-gray-700 transition-colors">
          <ArrowLeft className="w-4 h-4" />
        </button>
        <h1 className="text-xl font-medium text-gray-800">Load Experiment</h1>
      </div>

      {/* Directory Picker */}
      <Section icon={<FolderOpen className="w-4 h-4" />} title="Experiment Directory">
        <div data-tour="load-exp-directory" className="flex">
          <input
            type="text"
            value={loadDir}
            onChange={e => { setLoadDir(e.target.value); setSelectedAction(null); }}
            placeholder="Select experiment folder..."
            className="flex-1 border border-gray-300 rounded-l-sm px-3 py-1.5 text-sm focus:outline-none focus:border-blue-500 font-mono text-gray-600"
          />
          <button
            onClick={browse}
            className="bg-gray-200 border border-l-0 border-gray-300 px-3 rounded-r-sm hover:bg-gray-300"
          >
            <FolderOpen className="w-4 h-4 text-gray-600" />
          </button>
        </div>
      </Section>

      {/* Action cards — shown once directory is picked */}
      {dirPicked && (
        <div className="mt-8 space-y-5">
          <p className="text-xs text-gray-500 uppercase font-semibold tracking-wider">How would you like to continue?</p>

          <div data-tour="load-action-cards" className="grid grid-cols-2 gap-4">
            {/* Continue existing subject */}
            <button
              onClick={() => selectAction('continue')}
              className={`flex flex-col items-start gap-2 p-5 border-2 rounded-sm transition-all text-left ${
                selectedAction === 'continue'
                  ? 'border-[#0078D4] bg-blue-50/40'
                  : 'border-gray-200 hover:border-[#0078D4] hover:bg-blue-50/40'
              }`}
            >
              <div className={`w-9 h-9 rounded-full flex items-center justify-center transition-colors ${
                selectedAction === 'continue' ? 'bg-blue-100' : 'bg-blue-50'
              }`}>
                <User className="w-4 h-4 text-[#0078D4]" />
              </div>
              <div>
                <div className="font-semibold text-sm text-gray-800">Continue Subject</div>
                <div className="text-xs text-gray-400 mt-0.5">Select from existing subjects</div>
              </div>
            </button>

            {/* New subject */}
            <button
              onClick={() => selectAction('new')}
              className={`flex flex-col items-start gap-2 p-5 border-2 rounded-sm transition-all text-left ${
                selectedAction === 'new'
                  ? 'border-gray-500 bg-gray-50'
                  : 'border-gray-200 hover:border-gray-400 hover:bg-gray-50'
              }`}
            >
              <div className="w-9 h-9 rounded-full bg-gray-100 flex items-center justify-center">
                <FilePlus className="w-4 h-4 text-gray-600" />
              </div>
              <div>
                <div className="font-semibold text-sm text-gray-800">New Subject</div>
                <div className="text-xs text-gray-400 mt-0.5">Start a new trial in this experiment</div>
              </div>
            </button>
          </div>

          {/* Continue: subject list */}
          {selectedAction === 'continue' && (
            <div className="border border-gray-200 rounded-sm overflow-hidden">
              <table className="w-full text-left text-sm">
                <thead className="bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase">Subject ID</th>
                    <th className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase">Last Session</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {STUB_SUBJECTS.map(s => (
                    <tr
                      key={s.id}
                      onClick={() => setSelectedSubject(s.id)}
                      className={`cursor-pointer transition-colors ${
                        selectedSubject === s.id
                          ? 'bg-blue-50 text-[#0078D4]'
                          : 'hover:bg-gray-50 text-gray-700'
                      }`}
                    >
                      <td className="px-4 py-2.5 font-mono font-medium">{s.id}</td>
                      <td className="px-4 py-2.5 text-gray-500 text-xs">{s.lastSession}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="p-3 bg-gray-50 border-t border-gray-200 flex justify-end">
                <button
                  onClick={onStart}
                  disabled={!selectedSubject}
                  className="bg-[#0078D4] hover:bg-[#006CBE] disabled:bg-gray-300 disabled:cursor-not-allowed text-white py-1.5 px-6 rounded-sm text-xs font-semibold transition-colors flex items-center gap-1.5"
                >
                  <Play className="w-3 h-3" /> Continue with {selectedSubject ?? '—'}
                </button>
              </div>
            </div>
          )}

          {/* New subject: ID input */}
          {selectedAction === 'new' && (
            <div className="border border-gray-200 rounded-sm p-4 space-y-3">
              <label className="block text-xs text-gray-500">Subject ID</label>
              <div className="flex gap-3">
                <input
                  type="text"
                  value={newSubjectId}
                  onChange={e => setNewSubjectId(e.target.value)}
                  placeholder="e.g. Subject_009"
                  className="flex-1 border border-gray-300 rounded-sm px-3 py-1.5 text-sm focus:outline-none focus:border-blue-500 font-mono"
                />
                <button
                  onClick={onStart}
                  disabled={newSubjectId.trim() === ''}
                  className="bg-gray-800 hover:bg-gray-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white py-1.5 px-6 rounded-sm text-xs font-semibold transition-colors flex items-center gap-1.5"
                >
                  <Play className="w-3 h-3" /> Start
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// --- SHARED HELPERS ---

function Section({ icon, title, children }) {
  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        <span className="text-gray-400">{icon}</span>
        <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">{title}</h2>
      </div>
      {children}
    </div>
  );
}

function SettingsPlaceholder({ label }) {
  return (
    <div className="border border-dashed border-gray-300 rounded-sm p-4 bg-gray-50 flex flex-col items-center justify-center gap-2 min-h-[80px]">
      <Settings2 className="w-5 h-5 text-gray-300" />
      <span className="text-xs text-gray-400 text-center">{label}</span>
    </div>
  );
}
