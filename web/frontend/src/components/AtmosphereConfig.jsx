import React from 'react';
import { CloudRain } from 'lucide-react';

export default function AtmosphereConfig({ config, setConfig }) {
    return (
        <div className="bg-slate-800 p-5 rounded-lg border border-slate-700">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-cyan-400 flex items-center">
                    <CloudRain className="w-5 h-5 mr-2" /> Atmosphere
                </h3>
                <div className="relative inline-block w-10 mr-2 align-middle select-none transition duration-200 ease-in">
                    <input
                        type="checkbox"
                        name="atm-toggle"
                        id="atm-toggle"
                        checked={config.enabled}
                        onChange={(e) => setConfig({ ...config, enabled: e.target.checked })}
                        className="toggle-checkbox absolute block w-5 h-5 rounded-full bg-white border-4 appearance-none cursor-pointer"
                        style={{ right: config.enabled ? 0 : 'auto', left: config.enabled ? 'auto' : 0 }}
                    />
                    <label htmlFor="atm-toggle" className={`toggle-label block overflow-hidden h-5 rounded-full cursor-pointer ${config.enabled ? 'bg-cyan-500' : 'bg-slate-600'}`}></label>
                </div>
            </div>

            {config.enabled && (
                <div className="space-y-4 text-sm">
                    <div>
                        <label className="block text-slate-400 mb-1">RMS OPD (nm)</label>
                        <input
                            type="number"
                            value={config.rms_nm}
                            onChange={(e) => setConfig({ ...config, rms_nm: parseFloat(e.target.value) })}
                            className="w-full bg-slate-900 rounded px-3 py-2 border border-slate-700 focus:outline-none focus:border-cyan-500"
                        />
                        <p className="text-xs text-slate-500 mt-1">Optical Path Difference Root Mean Square</p>
                    </div>
                    <div>
                        <label className="block text-slate-400 mb-1">Wind Speed (m/s)</label>
                        <input
                            type="number"
                            value={config.wind_speed}
                            onChange={(e) => setConfig({ ...config, wind_speed: parseFloat(e.target.value) })}
                            className="w-full bg-slate-900 rounded px-3 py-2 border border-slate-700 focus:outline-none focus:border-cyan-500"
                        />
                    </div>
                </div>
            )}
        </div>
    );
}
