import React, { useState, useEffect } from 'react';
import { X, Save } from 'lucide-react';
import SceneConfig from './SceneConfig';
import TelescopeConfig from './TelescopeConfig';

export default function ComponentConfigModal({ node, isOpen, onClose, onChange }) {
    if (!isOpen || !node) return null;

    // Robust Unpacking for Legacy "Layer" Nodes
    let effectiveData = node.data;
    if (node.data.elements && node.data.elements.length > 0) {
        effectiveData = { ...node.data.elements[0] };
    }

    const { type, label } = effectiveData;
    // We initialize local config from effectiveData found above
    const [localConfig, setLocalConfig] = useState(effectiveData.config || {});

    useEffect(() => {
        // re-evaluate effectiveData inside effect or trust node prop change triggers render
        let currentData = node.data;
        if (node.data.elements && node.data.elements.length > 0) {
            currentData = node.data.elements[0];
        }
        setLocalConfig(currentData.config || {});
    }, [node]);

    const handleSave = () => {
        onChange(node.id, localConfig);
        onClose();
    };

    // Specific Setters for complex components
    const setStars = (v) => setLocalConfig(c => ({ ...c, stars: v }));
    const setPlanets = (v) => setLocalConfig(c => ({ ...c, planets: v }));
    const setZodiacal = (v) => setLocalConfig(c => ({ ...c, zodiacal: v }));

    const handleGenericChange = (key, value) => {
        setLocalConfig(c => ({ ...c, [key]: value }));
    };

    // Config Body Renderer
    const renderConfig = () => {
        if (type === 'scene') {
            return (
                <SceneConfig
                    stars={localConfig.stars || []} setStars={setStars}
                    planets={localConfig.planets || []} setPlanets={setPlanets}
                    zodiacal={localConfig.zodiacal || {}} setZodiacal={setZodiacal}
                />
            );
        } else if (type === 'telescope' || type === 'telescope_array') {
            // Basic single vs array mode check
            const mode = type === 'telescope_array' ? 'array' : 'single';
            return (
                <TelescopeConfig config={localConfig} setConfig={setLocalConfig} mode={mode} />
            );
        } else if (type === 'camera') {
            return (
                <div className="space-y-4">
                    <div>
                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">Exposure (s)</label>
                        <input
                            type="number"
                            value={localConfig.exposure || 0.1}
                            onChange={(e) => handleGenericChange('exposure', parseFloat(e.target.value))}
                            className="w-full text-sm p-2 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-md"
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">Wavelength (um)</label>
                        <input
                            type="number"
                            step={0.1}
                            value={localConfig.wavelength || 1.0}
                            onChange={(e) => handleGenericChange('wavelength', parseFloat(e.target.value))}
                            className="w-full text-sm p-2 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-md"
                        />
                    </div>
                </div>
            );
        } else {
            // Generic Fields if defined in data? Or standard fallback
            return (
                <div className="text-gray-500 italic text-sm">
                    No specific configuration available for {label}.
                </div>
            );
        }
    };

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="bg-white dark:bg-slate-900 rounded-lg shadow-2xl border border-slate-200 dark:border-slate-700 w-[600px] max-w-[90vw] max-h-[80vh] flex flex-col">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-slate-200 dark:border-slate-800">
                    <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">{label} Configuration</h2>
                    <button onClick={onClose} className="p-1 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-full transition-colors">
                        <X className="w-5 h-5 text-slate-500" />
                    </button>
                </div>

                {/* Body */}
                <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                    {renderConfig()}
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-slate-200 dark:border-slate-800 flex justify-end gap-2 bg-slate-50 dark:bg-slate-950/50 rounded-b-lg">
                    <button onClick={onClose} className="px-4 py-2 text-sm text-slate-600 hover:bg-slate-200 rounded-md transition-colors">
                        Cancel
                    </button>
                    <button onClick={handleSave} className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-md flex items-center gap-2 shadow-sm transition-colors">
                        <Save className="w-4 h-4" /> Save Changes
                    </button>
                </div>
            </div>
        </div>
    );
}
