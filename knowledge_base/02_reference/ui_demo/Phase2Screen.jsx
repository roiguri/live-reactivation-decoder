import { useState, useCallback } from 'react';
import { Settings2, AlertCircle, Play, Square, ArrowLeft } from 'lucide-react';
import { useWalkthroughAction } from '../walkthrough';

const INITIAL_DECODERS = [
  { id: 1, name: 'Decoder 1 (Red)', color: '#C41E3A', visible: true, path: "M0,60 Q50,40 100,65 T200,55 T300,70 T400,60 T500,45 T600,60 T700,10" },
  { id: 2, name: 'Decoder 2 (Green)', color: '#228B22', visible: true, path: "M0,80 Q50,70 100,75 T200,65 T300,50 T400,80 T500,70 T600,85 T700,80" },
  { id: 3, name: 'Decoder 3 (Orange)', color: '#CC5500', visible: true, path: "M0,50 Q50,60 100,45 T200,55 T300,60 T400,40 T500,55 T600,45 T700,50" }
];

export default function Phase2Screen({ onBack }) {
  const [isLive, setIsLive] = useState(false);
  const [decoders, setDecoders] = useState(INITIAL_DECODERS);
  const [showExitModal, setShowExitModal] = useState(false);

  // Register tutorial action for "Start Inference"
  useWalkthroughAction('phase2-start-inference', useCallback(() => setIsLive(true), []));

  const toggleDecoder = (id) => {
    setDecoders(prev => prev.map(d => d.id === id ? { ...d, visible: !d.visible } : d));
  };

  const changeColor = (id, color) => {
    setDecoders(prev => prev.map(d => d.id === id ? { ...d, color } : d));
  };

  return (
    <div className="h-full flex flex-col bg-white">
      <style>{`
        .terminal-scroll::-webkit-scrollbar {
          width: 14px;
        }
        .terminal-scroll::-webkit-scrollbar-track {
          background: #0C0C0C; 
          border-left: 1px solid #222;
        }
        .terminal-scroll::-webkit-scrollbar-thumb {
          background: #333;
          border-left: 1px solid #222;
        }
        @keyframes reveal-history {
          0% { clip-path: inset(0 100% 0 0); }
          100% { clip-path: inset(0 0 0 0); }
        }
        .animate-reveal-history {
          animation: reveal-history 8s linear forwards;
        }
        @keyframes reveal-graph {
          0% { clip-path: inset(0 100% 0 0); }
          100% { clip-path: inset(0 30% 0 0); }
        }
        .animate-reveal-graph {
          animation: reveal-graph 8s linear forwards;
        }
        @keyframes slide-now {
          0% { left: 0%; }
          100% { left: 70%; }
        }
        .animate-now-line {
          animation: slide-now 8s linear forwards;
        }
      `}</style>
      {/* Stream Header */}
      <div data-tour="stream-header" className="h-14 border-b border-gray-200 bg-[#F3F3F3] flex items-center justify-between px-6 shrink-0 shadow-sm z-10">
        <div className="flex items-center gap-4">
          {onBack && (
            <button onClick={() => setShowExitModal(true)} className="p-1 -ml-2 text-gray-500 hover:text-gray-800 transition-colors" title="Back to Training Pipeline">
              <ArrowLeft className="w-5 h-5" />
            </button>
          )}
          <div className="flex items-center gap-2">
            <div className={`w-2.5 h-2.5 rounded-full ${isLive ? 'bg-green-500 animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]' : 'bg-gray-400'}`}></div>
            <span className="font-semibold text-sm tracking-wide">{isLive ? 'LIVE INFERENCE' : 'INFERENCE HALTED'}</span>
          </div>
          <div className="h-4 w-px bg-gray-300"></div>
          <span className="text-xs text-gray-500 font-mono">Target: Bittium NeurOne (LSL)</span>
        </div>
        <div className="flex gap-4 text-xs font-mono text-gray-500 bg-white px-3 py-1 border border-gray-200 rounded-sm">
          <span>Latency: <span className="text-green-600 font-bold">42ms</span></span>
          <span>|</span>
          <span>Buffer: OK</span>
        </div>
      </div>

      {/* Main View: 3 Panels */}
      <div className="flex-1 flex overflow-hidden">

        {/* LEFT PANEL: Decoder Controls */}
        <div data-tour="decoder-controls" className="w-64 bg-gray-50 border-r border-gray-200 flex flex-col shadow-[4px_0_15px_-3px_rgba(0,0,0,0.02)] z-10 relative">
          <div className="p-4 border-b border-gray-200 bg-white">
            <h3 className="font-semibold text-sm text-gray-700">Decoders</h3>
          </div>
          <div className="p-4 space-y-3 flex-1 overflow-y-auto">
            {decoders.map(dec => (
              <div key={dec.id} className="flex items-center justify-between bg-white border border-gray-200 p-2.5 rounded-sm shadow-sm transition-colors hover:border-[#0078D4]/50">
                <label className="flex items-center gap-2.5 cursor-pointer flex-1 min-w-0">
                  <input
                    type="checkbox"
                    checked={dec.visible}
                    onChange={() => toggleDecoder(dec.id)}
                    className="w-4 h-4 text-[#0078D4] rounded-sm focus:ring-[#0078D4] border-gray-300 cursor-pointer"
                  />
                  <span className="text-sm font-medium text-gray-700 truncate">{dec.name}</span>
                </label>
                <div className="shrink-0 ml-3 relative flex items-center justify-center w-6 h-6 rounded-full border border-gray-200 overflow-hidden cursor-pointer" title="Change line color">
                  <input
                    type="color"
                    value={dec.color}
                    onChange={(e) => changeColor(dec.id, e.target.value)}
                    className="absolute -top-2 -left-2 w-10 h-10 p-0 border-0 cursor-pointer"
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* CENTER PANEL: Canvas (Summary + Graph) */}
        <div className="flex-1 relative flex flex-col items-center bg-gray-100 overflow-y-auto">
          <div className="w-full max-w-5xl flex-1 flex flex-col py-8 px-8">

            <div className="w-full flex flex-col relative pr-4">
              {/* Title */}
              <div className="w-full text-left text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Decision History</div>

              {/* Time Scale Strip Above Everything */}
              <div className="w-full flex relative text-xs font-mono font-semibold text-gray-800 mb-2 h-4">
                <span className="absolute left-0">T - 10s</span>
                <span className="absolute left-[35%] -translate-x-1/2">T - 5s</span>
                <span className="absolute left-[70%] text-[#0078D4] font-bold -translate-x-1/2">NOW</span>
              </div>

              {/* DECISION HISTORY STRIP */}
              <div data-tour="decision-history" className="h-10 w-full border border-gray-200 bg-white relative flex mb-6">
                <div className="w-[70%] h-full flex relative">
                  {isLive ? (
                    <div className="h-full w-full flex relative animate-reveal-history">
                      {/* Empty block */}
                      <div className="h-full w-[20%] border-r border-dashed border-gray-200 bg-gray-50 flex flex-col items-center justify-center">
                        <span className="text-xs text-gray-300 font-mono">---</span>
                      </div>
                      {/* Decision Block 1 */}
                      <div className="h-full w-[15%] border-r border-dashed border-gray-200 bg-[#C41E3A]/5 flex flex-col items-center justify-center border-b-2 border-b-[#C41E3A]">
                        <span className="text-[10px] font-bold text-gray-700 truncate w-full text-center px-1">Decoder 1</span>
                      </div>
                      {/* Empty block */}
                      <div className="h-full w-[35%] border-r border-dashed border-gray-200 bg-gray-50 flex flex-col items-center justify-center">
                        <span className="text-xs text-gray-300 font-mono">---</span>
                      </div>
                      {/* Decision Block 2 */}
                      <div className="h-full w-[10%] border-r border-dashed border-gray-200 bg-[#228B22]/5 flex flex-col items-center justify-center border-b-2 border-b-[#228B22]">
                        <span className="text-[10px] font-bold text-gray-700 truncate w-full text-center px-1">Decoder 2</span>
                      </div>
                      {/* Empty block (to now) */}
                      <div className="h-full w-[20%] bg-gray-50 flex flex-col items-center justify-center">
                        <span className="text-xs text-gray-300 font-mono">---</span>
                      </div>
                    </div>
                  ) : (
                    <div className="h-full w-full bg-gray-50 flex flex-col items-center justify-center">
                      <span className="text-xs text-gray-400 font-mono">STANDING BY</span>
                    </div>
                  )}
                </div>
                {/* Now Marker connecting history to graph */}
                {isLive && <div className="absolute top-0 bottom-[-24px] w-px bg-[#0078D4] z-10 shadow-[0_0_8px_rgba(0,120,212,0.8)] animate-now-line"></div>}
              </div>

              {/* PROBABILITY GRAPH TITLE */}
              <div className="w-full text-left text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Probability Analysis</div>

              {/* PROBABILITY GRAPH (Taller, Full Width) */}
              <div data-tour="probability-graph" className="w-full h-80 bg-white border-l-2 border-b-2 border-gray-400 relative shrink-0">
                {/* Y-Axis Labels */}
                <div className="absolute -left-12 top-0 text-xs font-mono font-semibold text-gray-800">1.0</div>
                <div className="absolute -left-12 top-[15%] text-xs font-mono font-bold text-[#0078D4]">0.85</div>
                <div className="absolute -left-12 top-1/2 text-xs font-mono font-semibold text-gray-800">0.5</div>
                <div className="absolute -left-12 bottom-[-4px] text-xs font-mono font-semibold text-gray-800">0.0</div>

                {/* Threshold / Chance Lines (70% Past/Present) */}
                <div className="absolute top-[15%] left-0 w-[70%] border-t border-[#0078D4]/30 pointer-events-none"></div>
                <div className="absolute top-1/2 left-0 w-[70%] border-t border-dashed border-[#757575] pointer-events-none"></div>

                {/* Threshold / Chance Lines (30% Future Space fading out) */}
                <div className="absolute top-[15%] left-[70%] right-0 border-t border-[#0078D4]/10 pointer-events-none"></div>
                <div className="absolute top-1/2 left-[70%] right-0 border-t border-dashed border-[#757575]/50 pointer-events-none"></div>

                {/* NOW Line at 70% width */}
                {isLive && <div className="absolute top-0 bottom-0 w-px bg-[#0078D4] shadow-[0_0_8px_rgba(0,120,212,0.8)] z-10 animate-now-line"></div>}

                {/* Graph Canvas */}
                <div className="w-full h-full overflow-hidden relative">
                  <svg className={`absolute inset-0 w-full h-full ${isLive ? 'animate-reveal-graph' : ''}`} preserveAspectRatio="none" viewBox="0 0 1000 100">
                    {isLive && decoders.filter(d => d.visible).map(d => (
                      <path key={d.id} d={d.path} fill="none" stroke={d.color} strokeWidth="2.5" strokeLinejoin="round" opacity="0.85" />
                    ))}
                  </svg>
                </div>
              </div>

              {/* X-AXIS BELOW GRAPH */}
              <div className="w-full flex relative text-xs font-mono font-semibold text-gray-800 mt-2 h-4">
                <span className="absolute left-0">T - 10s</span>
                <span className="absolute left-[35%] -translate-x-1/2">T - 5s</span>
                <span className="absolute left-[70%] text-[#0078D4] font-bold -translate-x-1/2">NOW</span>
                <span className="absolute right-0">T + 4.3s</span>
              </div>

              {/* TERMINAL TRIGGER LOG TITLE */}
              <div className="w-full text-left text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 mt-8">Trigger Log</div>
              
              {/* TERMINAL TRIGGER LOG */}
              <div data-tour="terminal-log" className="w-full bg-[#0C0C0C] border border-gray-800 rounded-sm shadow-inner flex flex-col shrink-0 h-60">
                <div className="h-6 bg-[#2D2D2D] border-b border-black flex items-center px-3 justify-between shrink-0 z-10 shadow-sm">
                  <div className="flex gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-full bg-red-500"></div>
                    <div className="w-2.5 h-2.5 rounded-full bg-yellow-500"></div>
                    <div className="w-2.5 h-2.5 rounded-full bg-green-500"></div>
                  </div>
                  <span className="text-[10px] text-gray-400 font-mono">Trigger Log - /dev/ttyS0</span>
                  <div className="w-10"></div>
                </div>
                <div className="flex-1 min-h-0 p-4 font-mono text-[10px] text-green-400 flex flex-col overflow-y-auto w-full terminal-scroll">
                  <div className="opacity-50 mb-1 shrink-0">[11:58:14.020] INFERENCE_ENGINE: Standing by... buffer processing ok</div>
                  <div className="opacity-50 mb-1 shrink-0">[11:58:14.852] INFERENCE_ENGINE: Packet #294025 received.</div>
                  <div className="text-red-500 font-bold mt-2 mb-2 leading-tight shrink-0" style={{ whiteSpace: 'pre' }}>
{`=============================================================
  _______ _____  _____  _____  _____ ______ _____  
 |__   __|  __ \\|_   _|/ ____|/ ____|  ____|  __ \\ 
    | |  | |__) | | | | |  __| |  __| |__  | |__) |
    | |  |  _  /  | | | | |_ | | |_ |  __| |  _  / 
    | |  | | \\ \\ _| |_| |__| | |__| | |____| | \\ \\ 
    |_|  |_|  \\_\\_____|\\_____|\\_____|______|_|  \\_\\
=============================================================`}
                  </div>
                  <div className="text-yellow-300 font-bold shrink-0">
                    <div>[11:58:15.200] TRIGGER EVENT: DECODER 1 (RED) EXCEEDED 0.85 THRESHOLD</div>
                    <div>[11:58:15.202] LSL_OUT: Pushing trigger byte 0x5EFC</div>
                  </div>
                </div>
              </div>

            </div>

          </div>
        </div>

        {/* RIGHT PANEL: Trigger Controls */}
        <div data-tour="decision-settings" className="w-80 bg-[#F9F9F9] border-l border-gray-200 flex flex-col shadow-[-4px_0_15px_-3px_rgba(0,0,0,0.02)] z-10">
          <div className="p-4 border-b border-gray-200 flex items-center bg-white">
            <Settings2 className="w-4 h-4 mr-2 text-gray-600" />
            <h3 className="font-semibold text-sm">Decision Settings</h3>
          </div>

          <div className="p-6 space-y-6 flex-1 overflow-y-auto">
            {/* Threshold Rule */}
            <div>
              <div className="flex justify-between items-end mb-1">
                <label className="text-xs font-semibold text-gray-700 uppercase">Trigger Threshold</label>
                <span className="font-mono text-xs text-blue-600 font-bold">0.85</span>
              </div>
              <p className="text-[10px] text-gray-500 mb-2 leading-tight">Fire if probability exceeds this value.</p>
              <input type="range" min="50" max="100" defaultValue="85" className="w-full accent-[#0078D4]" />
            </div>

            {/* Sustained Activation */}
            <div>
              <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">Sustained Activation</label>
              <p className="text-[10px] text-gray-500 mb-2 leading-tight">Required signal length (ms) above threshold.</p>
              <div className="flex items-center gap-2">
                <input type="number" defaultValue="50" className="border border-gray-300 rounded-sm px-2 py-1 font-mono text-sm w-20 text-right" />
                <span className="text-xs text-gray-600 font-mono">ms</span>
              </div>
            </div>

            {/* Conflict Resolution */}
            <div>
              <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">Conflict Resolution</label>
              <p className="text-[10px] text-gray-500 mb-2 leading-tight">Handling simultaneous high probabilities.</p>
              <select className="w-full border border-gray-300 rounded-sm px-2 py-1.5 text-xs bg-white focus:border-[#0078D4] focus:outline-none">
                <option>Highest Probability Wins</option>
                <option>Require &gt; 15% Margin</option>
                <option>Suppress Trigger</option>
              </select>
            </div>
          </div>

          <div className="p-5 bg-white border-t border-gray-200">
            {isLive ? (
              <button
                onClick={() => setIsLive(false)}
                className="w-full bg-[#C41E3A] hover:bg-[#A31830] text-white py-3.5 rounded-sm text-sm font-bold uppercase flex justify-center items-center shadow-md transition-colors"
              >
                <Square className="w-4 h-4 mr-2" fill="currentColor" /> Halt Inference
              </button>
            ) : (
              <button
                data-tour="start-inference-btn"
                onClick={() => setIsLive(true)}
                className="w-full bg-green-600 hover:bg-green-700 text-white py-3.5 rounded-sm text-sm font-bold uppercase flex justify-center items-center shadow-md transition-colors"
              >
                <Play className="w-4 h-4 mr-2" fill="currentColor" /> Start Inference
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Exit Confirmation Modal */}
      {showExitModal && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
          <div className="bg-white rounded-md shadow-xl w-full max-w-md overflow-hidden">
            <div className="p-6">
              <h3 className="text-lg font-bold text-gray-800 mb-2">Leave Live Inference?</h3>
              <p className="text-sm text-gray-600 mb-6">
                Are you sure you want to return to the Training Pipeline? 
              </p>
              <div className="flex justify-end gap-3 font-medium text-sm">
                <button 
                  onClick={() => setShowExitModal(false)}
                  className="px-4 py-2 border border-gray-300 rounded-sm text-gray-700 hover:bg-gray-50 transition-colors"
                >
                  Cancel
                </button>
                <button 
                  onClick={() => { setShowExitModal(false); onBack && onBack(); }}
                  className="px-4 py-2 bg-[#0078D4] text-white rounded-sm hover:bg-blue-700 transition-colors"
                >
                  Confirm Exit
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
