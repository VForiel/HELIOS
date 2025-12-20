import React, { memo, useState } from 'react';
import { Handle, Position, useStore } from 'reactflow';
import { Settings, Trash2, ChevronDown, ChevronRight, Save, Eye, Layers } from 'lucide-react';
import { getElementIcon } from '../../../utils/iconMap';
import { usePipelineContext } from '../../../contexts/PipelineContext';

// Config Components
import SceneConfig from '../../SceneConfig';
import TelescopeConfig from '../../TelescopeConfig';

// Helper for "nodrag" to allow input interaction
const NoDrag = ({ children }) => (
    <div className="nodrag cursor-auto" onMouseDown={(e) => e.stopPropagation()}>
        {children}
    </div>
);

// Zoom Selector for performance
const zoomSelector = (s) => s.transform[2];

export default memo(({ data, selected, id }) => {
    // Robust Data Unpacking
    let effectiveData = data;
    if (data.elements && data.elements.length > 0) {
        effectiveData = {
            ...data.elements[0],
        };
    }

    // Zoom Level
    const zoom = useStore(zoomSelector);
    const showIconOnly = zoom < 0.6; // Semantic Zoom Threshold

    // State for expansion
    const [expanded, setExpanded] = useState(true);

    // Extraction with Fallbacks
    const type = effectiveData.type || 'unknown';
    const label = effectiveData.label || type;
    const iconPath = effectiveData.iconPath || getElementIcon(type);
    // Fix: onInspect from Context
    const { onInspect } = usePipelineContext();
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
        } else if (type === 'atmosphere') {
            return (
                <div className="space-y-2 p-2">
                    <div>
                        <label className="block text-[10px] font-medium text-slate-500 mb-1">Seeing (arcsec)</label>
                        <input
                            type="number"
                            step={0.1}
                            value={config.seeing || 0.8}
                            onChange={(e) => handleConfigChange({ ...config, seeing: parseFloat(e.target.value) })}
                            className="w-full text-xs p-1 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded"
                        />
                    </div>
                </div>
            )
        }
        else {
            // Generic or unknown
            return <div className="p-2 text-xs text-slate-400">No settings available</div>;
        }
    };

    // SEMANTIC ZOOM: Icon View
    if (showIconOnly) {
        return (
            <div
                className={`
                    w-64 h-64 rounded-3xl flex flex-col items-center justify-center gap-4
                    bg-white dark:bg-slate-900 border-4 transition-all shadow-xl cursor-pointer
                    ${selected ? 'border-blue-500 shadow-blue-500/30 ring-4 ring-blue-500/20' : 'border-slate-300 dark:border-slate-700'}
                    hover:scale-105 active:scale-95
                `}
                onClick={(e) => {
                    // Only trigger if semantic zoom is active (icon mode)
                    if (data.onOpenConfig) {
                        e.stopPropagation(); // Prevent selecting the node if we just want to open config? 
                        // Actually, maybe we accept selection too. But let's stop propagation to avoid side effects if desired.
                        // However, selecting it is good visual feedback.
                        // Let's call the handler.
                        data.onOpenConfig({ id, data, position: { x: 0, y: 0 } }); // Pass pseudo-node or minimal needed
                    }
                }}
            >
                {/* Inputs */}
                {Array.from({ length: inputCount }).map((_, i) => (
                    <Handle
                        key={`in-${i}`}
                        type="target"
                        position={Position.Left}
                        id={`input-${i}`}
                        style={{ top: '50%', background: '#3b82f6', width: '24px', height: '24px', border: '4px solid white', left: '-12px' }}
                    />
                ))}

                {/* Icon */}
                {iconPath ? (
                    <img src={iconPath} alt={label} className="w-32 h-32 dark:invert opacity-90" />
                ) : Icon ? (
                    <Icon className="w-32 h-32 text-slate-600 dark:text-slate-300" />
                ) : (
                    <Layers className="w-32 h-32 text-slate-400" />
                )}

                {/* Label (Bold & Large) */}
                <span className="text-2xl font-black text-slate-700 dark:text-slate-200 truncate max-w-[90%] pointer-events-none mt-2">
                    {label}
                </span>

                {/* Outputs */}
                {Array.from({ length: outputCount }).map((_, i) => (
                    <Handle
                        key={`out-${i}`}
                        type="source"
                        position={Position.Right}
                        id={`output-${i}`}
                        style={{ top: '50%', background: '#3b82f6', width: '24px', height: '24px', border: '4px solid white', right: '-12px' }}
                    />
                ))}
            </div>
        );
    }

    // STANDARD MODE (Expanded/Detailed)
    return (
        <div className={`
            min-w-[280px] rounded-lg border bg-white dark:bg-slate-900 shadow-lg transition-all duration-200
            ${selected ? 'border-blue-500 ring-1 ring-blue-500' : 'border-slate-200 dark:border-slate-800 hover:border-slate-300'}
        `}>
            {/* Header */}
            <div className={`
                flex items-center justify-between p-3 border-b border-slate-100 dark:border-slate-800
                ${selected ? 'bg-blue-50/50 dark:bg-blue-900/10' : ''}
            `}>
                <div className="flex items-center gap-3">
                    <div className="p-1.5 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400">
                        {iconPath ? (
                            <img src={iconPath} className="w-4 h-4 dark:invert" alt="" />
                        ) : Icon ? (
                            <Icon size={16} />
                        ) : (
                            <Layers size={16} />
                        )}
                    </div>
                    <span className="font-semibold text-slate-700 dark:text-slate-200 text-sm">
                        {label}
                    </span>
                </div>

                <div className="flex items-center gap-1">
                    <button
                        onClick={onInspect ? () => onInspect(id, config) : undefined}
                        className="p-1 text-slate-400 hover:text-blue-500 transition-colors rounded"
                        title="Inspect Output"
                    >
                        <Eye size={14} />
                    </button>
                    {onDelete && (
                        <button
                            onClick={(e) => { e.stopPropagation(); onDelete(id); }}
                            className="p-1 text-slate-400 hover:text-red-500 transition-colors rounded"
                        >
                            <Trash2 size={14} />
                        </button>
                    )}
                    <button
                        onClick={() => setExpanded(!expanded)}
                        className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
                    >
                        {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                    </button>
                </div>
            </div>

            {/* Content (Collapsible) */}
            {expanded && (
                <div className="p-3 bg-white dark:bg-slate-900 rounded-b-lg">
                    <NoDrag>
                        {renderConfig()}
                    </NoDrag>
                </div>
            )}

            {/* Inputs - Left Side */}
            {Array.from({ length: inputCount }).map((_, i) => (
                <Handle
                    key={`in-${i}`}
                    type="target"
                    position={Position.Left}
                    id={`input-${i}`}
                    style={{
                        top: inputCount === 1 ? '50%' : `${((i + 1) * 100) / (inputCount + 1)}%`,
                        background: '#3b82f6',
                        width: '10px', height: '10px',
                        border: '2px solid white'
                    }}
                />
            ))}

            {/* Outputs - Right Side */}
            {Array.from({ length: outputCount }).map((_, i) => (
                <Handle
                    key={`out-${i}`}
                    type="source"
                    position={Position.Right}
                    id={`output-${i}`}
                    style={{
                        top: outputCount === 1 ? '50%' : `${((i + 1) * 100) / (outputCount + 1)}%`,
                        background: '#3b82f6',
                        width: '10px', height: '10px',
                        border: '2px solid white'
                    }}
                />
            ))}
        </div>
    );
});
