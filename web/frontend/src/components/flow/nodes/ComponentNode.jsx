import React, { memo, useState } from 'react';
import { Handle, Position } from 'reactflow';
import { Settings, Trash2, ChevronDown, ChevronRight, Save } from 'lucide-react';
import { getElementIcon } from '../../../utils/iconMap';

// Config Components
import SceneConfig from '../../SceneConfig';
import TelescopeConfig from '../../TelescopeConfig';

// Helper for "nodrag" to allow input interaction
const NoDrag = ({ children }) => (
    <div className="nodrag cursor-auto" onMouseDown={(e) => e.stopPropagation()}>
        {children}
    </div>
);

export default memo(({ data, selected, id }) => {
    // Robust Data Unpacking
    let effectiveData = data;
    if (data.elements && data.elements.length > 0) {
        effectiveData = {
            ...data.elements[0],
        };
    }

    // State for expansion
    const [expanded, setExpanded] = useState(true);

    // Extraction with Fallbacks
    const type = effectiveData.type || 'unknown';
    const label = effectiveData.label || type;
    const iconPath = effectiveData.iconPath || getElementIcon(type);
    const { icon: Icon, onEdit, onDelete, config: initialConfig } = effectiveData;

    // Internal state for config editing (shallow copy)
    const [config, setConfig] = useState(initialConfig || {});

    // Default counts
    const inputCount = effectiveData.inputs !== undefined ? effectiveData.inputs : (type === 'scene' ? 0 : 1);
    const outputCount = effectiveData.outputs !== undefined ? effectiveData.outputs : (type === 'camera' ? 0 : 1);

    // Handlers
    const handleConfigChange = (newConfig) => {
        // Deep merge or replace? Usually replace for simple objects, but setState pattern implies replace.
        // We update local state. For upstream update, we might need a save or auto-sync.
        // The original concept passed 'onChange'. PipelineEditor needs to handle this update.
        setConfig(newConfig);

        // Propagate change immediately if a handler exists in data
        // Note: 'data' in ReactFlow is not automatically creating a setter for us unless we passed one.
        // We usually rely on 'onEdit' passing back the ID and new data, OR a direct 'onChange' prop.
        // data.onChange might have been passed? If not, we might need to assume 'onEdit' is for modal.
        // Let's assume we need to call a function passed in data called 'onChange'.
        if (data.onChange) {
            data.onChange(id, { ...effectiveData, config: newConfig });
        }
    };

    // Handler helpers for specific components
    const setStars = (v) => handleConfigChange({ ...config, stars: v });
    const setPlanets = (v) => handleConfigChange({ ...config, planets: v });
    const setZodiacal = (v) => handleConfigChange({ ...config, zodiacal: v });


    const renderConfig = () => {
        if (type === 'scene') {
            return (
                <SceneConfig
                    stars={config.stars || []} setStars={setStars}
                    planets={config.planets || []} setPlanets={setPlanets}
                    zodiacal={config.zodiacal || {}} setZodiacal={setZodiacal}
                />
            );
        } else if (type === 'telescope' || type === 'telescope_array') {
            const mode = type === 'telescope_array' ? 'array' : 'single';
            return (
                <TelescopeConfig config={config} setConfig={handleConfigChange} mode={mode} />
            );
        } else if (type === 'camera') {
            return (
                <div className="space-y-2 p-2">
                    <div>
                        <label className="block text-[10px] font-medium text-slate-500 mb-1">Exposure (s)</label>
                        <input
                            type="number"
                            value={config.exposure || 0.1}
                            onChange={(e) => handleConfigChange({ ...config, exposure: parseFloat(e.target.value) })}
                            className="w-full text-xs p-1 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded"
                        />
                    </div>
                    <div>
                        <label className="block text-[10px] font-medium text-slate-500 mb-1">Wavelength (um)</label>
                        <input
                            type="number"
                            step={0.1}
                            value={config.wavelength || 1.0}
                            onChange={(e) => handleConfigChange({ ...config, wavelength: parseFloat(e.target.value) })}
                            className="w-full text-xs p-1 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded"
                        />
                    </div>
                </div>
            );
        }
        return <div className="p-2 text-xs italic text-slate-400">No inline config</div>;
    };

    // Helper to generate handles
    const renderHandles = (handleType, count, position) => {
        return Array.from({ length: count }).map((_, i) => (
            <Handle
                key={`${handleType}-${i}`}
                type={handleType}
                position={position}
                id={`${handleType}-${i}`}
                style={{
                    top: `${((i + 1) / (count + 1)) * 100}%`,
                    width: '6px',
                    height: '6px',
                    background: '#94a3b8',
                    border: '1px solid white'
                }}
                className={`!w-1.5 !h-1.5 transition-colors hover:!bg-blue-500 hover:!border-blue-500`}
            />
        ));
    };

    return (
        <div className={`
            relative flex flex-col min-w-[200px] bg-white dark:bg-slate-900 
            border rounded-md shadow-sm transition-all
            ${selected ? 'border-blue-500 ring-2 ring-blue-500/30' : 'border-slate-300 dark:border-slate-700'}
        `}>
            {/* Header / Title */}
            <div
                className={`
                    flex items-center gap-2 px-2 py-2 border-b border-inherit rounded-t-md
                    ${selected ? 'bg-blue-50 dark:bg-blue-900/20' : 'bg-slate-50 dark:bg-slate-800'}
                `}
            >
                {/* Fixed Expanded State - No Toggle */}
                <div className="p-1 rounded bg-white dark:bg-slate-700 shadow-sm shrink-0">
                    {iconPath ? (
                        <img src={iconPath} alt={label} className="w-4 h-4 dark:invert" />
                    ) : Icon ? (
                        <Icon className="w-4 h-4 text-slate-600 dark:text-slate-300" />
                    ) : (
                        <Settings className="w-4 h-4 text-slate-400" />
                    )}
                </div>
                <span className="text-xs font-semibold text-slate-700 dark:text-slate-200 truncate flex-1" title={label}>
                    {label}
                </span>

                {/* Quick Actions (Delete only to reduce clutter, edit is now inline or via modal) */}
                <button
                    onClick={(e) => { e.stopPropagation(); onDelete && onDelete(id); }}
                    className="p-1 rounded hover:bg-red-100 hover:text-red-500 text-slate-400 transition-colors"
                >
                    <Trash2 className="w-3 h-3" />
                </button>
            </div>

            {/* Body */}
            <div className="relative">
                {/* Input Handles Container - Stick to Left */}
                <div className="absolute left-0 top-0 bottom-0 w-1 z-10">
                    {renderHandles('target', inputCount, Position.Left)}
                </div>

                {/* Content Area */}
                <div className="bg-white dark:bg-slate-900 rounded-b-md overflow-hidden">
                    {expanded && (
                        <NoDrag>
                            <div className="border-t border-slate-100 dark:border-slate-800 p-1 max-h-[300px] overflow-y-auto custom-scrollbar">
                                {renderConfig()}
                            </div>
                        </NoDrag>
                    )}

                    {!expanded && (
                        <div className="h-6 flex items-center justify-center">
                            <span className="text-[9px] text-slate-400 uppercase tracking-widest">{type}</span>
                        </div>
                    )}
                </div>

                {/* Output Handles Container - Stick to Right */}
                <div className="absolute right-0 top-0 bottom-0 w-1 z-10">
                    {renderHandles('source', outputCount, Position.Right)}
                </div>
            </div>
        </div>
    );
});
