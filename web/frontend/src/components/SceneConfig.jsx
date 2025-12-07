import React from 'react';
import { Plus, Trash2, Globe, Sun, Zap } from 'lucide-react';

export default function SceneConfig({ stars, setStars, planets, setPlanets, zodiacal, setZodiacal }) {

    // --- Star Helpers ---
    const addStar = () => {
        setStars([...stars, { temperature: 5778, magnitude: 4.83, x_arcsec: 0, y_arcsec: 0 }]);
    };
    const updateStar = (index, field, value) => {
        const newStars = [...stars];
        newStars[index][field] = parseFloat(value);
        setStars(newStars);
    };
    const removeStar = (index) => {
        if (stars.length > 1) { // Prevent removing last star? Or allow it? Let's allow but maybe warn or just re-add default if empty in backend config
            setStars(stars.filter((_, i) => i !== index));
        }
    };

    // --- Planet Helpers ---
    const addPlanet = () => {
        setPlanets([...planets, { mass: 1.0, separation: 1.0, angle: 0.0, radius: 1.0 }]);
    };
    const updatePlanet = (index, field, value) => {
        const newPlanets = [...planets];
        newPlanets[index][field] = parseFloat(value);
        setPlanets(newPlanets);
    };
    const removePlanet = (index) => {
        setPlanets(planets.filter((_, i) => i !== index));
    };

    return (
        <div className="space-y-6">

            {/* Stars Section */}
            <div className="bg-white dark:bg-slate-800 p-5 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm dark:shadow-none">
                <h3 className="text-lg font-medium text-blue-600 dark:text-blue-400 mb-3 flex items-center">
                    <Sun className="w-5 h-5 mr-2" /> Stars
                </h3>
                <div className="space-y-3">
                    {stars.map((star, index) => (
                        <div key={index} className="bg-slate-50 dark:bg-slate-900/50 p-3 rounded border border-slate-200 dark:border-slate-800 text-sm">
                            <div className="flex justify-between items-center mb-2">
                                <span className="font-semibold text-slate-700 dark:text-slate-300">Star #{index + 1}</span>
                                {stars.length > 0 && (
                                    <button onClick={() => removeStar(index)} className="text-red-400 hover:text-red-300">
                                        <Trash2 className="w-3.5 h-3.5" />
                                    </button>
                                )}
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                                <div>
                                    <label className="block text-xs text-slate-500 mb-1">Temp (K)</label>
                                    <input type="number" value={star.temperature} onChange={(e) => updateStar(index, 'temperature', e.target.value)}
                                        className="w-full bg-white dark:bg-slate-800 rounded px-2 py-1 border border-slate-300 dark:border-slate-700 text-slate-900 dark:text-slate-100 focus:outline-none focus:border-blue-500" />
                                </div>
                                <div>
                                    <label className="block text-xs text-slate-500 mb-1">Mag</label>
                                    <input type="number" value={star.magnitude} onChange={(e) => updateStar(index, 'magnitude', e.target.value)}
                                        className="w-full bg-white dark:bg-slate-800 rounded px-2 py-1 border border-slate-300 dark:border-slate-700 text-slate-900 dark:text-slate-100 focus:outline-none focus:border-blue-500" />
                                </div>
                                <div>
                                    <label className="block text-xs text-slate-500 mb-1">X (arcsec)</label>
                                    <input type="number" value={star.x_arcsec} onChange={(e) => updateStar(index, 'x_arcsec', e.target.value)}
                                        className="w-full bg-white dark:bg-slate-800 rounded px-2 py-1 border border-slate-300 dark:border-slate-700 text-slate-900 dark:text-slate-100 focus:outline-none focus:border-blue-500" />
                                </div>
                                <div>
                                    <label className="block text-xs text-slate-500 mb-1">Y (arcsec)</label>
                                    <input type="number" value={star.y_arcsec} onChange={(e) => updateStar(index, 'y_arcsec', e.target.value)}
                                        className="w-full bg-white dark:bg-slate-800 rounded px-2 py-1 border border-slate-300 dark:border-slate-700 text-slate-900 dark:text-slate-100 focus:outline-none focus:border-blue-500" />
                                </div>
                            </div>
                        </div>
                    ))}
                    <button onClick={addStar} className="w-full py-1.5 rounded border border-dashed border-slate-600 text-slate-400 hover:text-white text-xs flex items-center justify-center">
                        <Plus className="w-3.5 h-3.5 mr-1" /> Add Star
                    </button>
                </div>
            </div>

            {/* Planets Section */}
            <div className="bg-white dark:bg-slate-800 p-5 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm dark:shadow-none">
                <h3 className="text-lg font-medium text-green-600 dark:text-green-400 mb-3 flex items-center">
                    <Globe className="w-5 h-5 mr-2" /> Planets
                </h3>
                <div className="space-y-3">
                    {planets.map((planet, index) => (
                        <div key={index} className="bg-slate-50 dark:bg-slate-900/50 p-3 rounded border border-slate-200 dark:border-slate-800 text-sm">
                            <div className="flex justify-between items-center mb-2">
                                <span className="font-semibold text-slate-700 dark:text-slate-300">Planet #{index + 1}</span>
                                <button onClick={() => removePlanet(index)} className="text-red-400 hover:text-red-300">
                                    <Trash2 className="w-3.5 h-3.5" />
                                </button>
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                                <div>
                                    <label className="block text-xs text-slate-500 mb-1">Separation (AU)</label>
                                    <input type="number" value={planet.separation} onChange={(e) => updatePlanet(index, 'separation', e.target.value)}
                                        className="w-full bg-white dark:bg-slate-800 rounded px-2 py-1 border border-slate-300 dark:border-slate-700 text-slate-900 dark:text-slate-100 focus:outline-none focus:border-green-500" />
                                </div>
                                <div>
                                    <label className="block text-xs text-slate-500 mb-1">Angle (deg)</label>
                                    <input type="number" value={planet.angle} onChange={(e) => updatePlanet(index, 'angle', e.target.value)}
                                        className="w-full bg-white dark:bg-slate-800 rounded px-2 py-1 border border-slate-300 dark:border-slate-700 text-slate-900 dark:text-slate-100 focus:outline-none focus:border-green-500" />
                                </div>
                                <div>
                                    <label className="block text-xs text-slate-500 mb-1">Mass (Mjup)</label>
                                    <input type="number" value={planet.mass} onChange={(e) => updatePlanet(index, 'mass', e.target.value)}
                                        className="w-full bg-white dark:bg-slate-800 rounded px-2 py-1 border border-slate-300 dark:border-slate-700 text-slate-900 dark:text-slate-100 focus:outline-none focus:border-green-500" />
                                </div>
                                <div>
                                    <label className="block text-xs text-slate-500 mb-1">Radius (Rjup)</label>
                                    <input type="number" value={planet.radius} onChange={(e) => updatePlanet(index, 'radius', e.target.value)}
                                        className="w-full bg-white dark:bg-slate-800 rounded px-2 py-1 border border-slate-300 dark:border-slate-700 text-slate-900 dark:text-slate-100 focus:outline-none focus:border-green-500" />
                                </div>
                            </div>
                        </div>
                    ))}
                    <button onClick={addPlanet} className="w-full py-1.5 rounded border border-dashed border-slate-600 text-slate-400 hover:text-white text-xs flex items-center justify-center">
                        <Plus className="w-3.5 h-3.5 mr-1" /> Add Planet
                    </button>
                </div>
            </div>

            {/* Zodiacal Light Section */}
            <div className="bg-white dark:bg-slate-800 p-5 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm dark:shadow-none">
                <div className="flex items-center justify-between mb-3">
                    <h3 className="text-lg font-medium text-yellow-500 dark:text-yellow-400 flex items-center">
                        <Zap className="w-5 h-5 mr-2" /> Zodiacal Light
                    </h3>
                    <div className="relative inline-block w-10 mr-2 align-middle select-none transition duration-200 ease-in">
                        <input type="checkbox" name="toggle" id="zodi-toggle" checked={zodiacal.enabled} onChange={(e) => setZodiacal({ ...zodiacal, enabled: e.target.checked })} className="toggle-checkbox absolute block w-5 h-5 rounded-full bg-white border-4 appearance-none cursor-pointer checked:right-0 right-5" style={{ right: zodiacal.enabled ? 0 : 'auto', left: zodiacal.enabled ? 'auto' : 0 }} />
                        <label htmlFor="zodi-toggle" className={`toggle-label block overflow-hidden h-5 rounded-full cursor-pointer ${zodiacal.enabled ? 'bg-yellow-400' : 'bg-slate-600'}`}></label>
                    </div>
                </div>
                {zodiacal.enabled && (
                    <div className="grid grid-cols-2 gap-3 text-sm">
                        <div>
                            <label className="block text-xs text-slate-500 mb-1">Brightness</label>
                            <input type="number" value={zodiacal.brightness} step="0.1" onChange={(e) => setZodiacal({ ...zodiacal, brightness: parseFloat(e.target.value) })}
                                className="w-full bg-white dark:bg-slate-900 rounded px-2 py-1 border border-slate-300 dark:border-slate-700 text-slate-900 dark:text-slate-100 focus:outline-none focus:border-yellow-500" />
                        </div>
                        <div>
                            <label className="block text-xs text-slate-500 mb-1">Radius (arcsec)</label>
                            <input type="number" value={zodiacal.radius || ''} placeholder="Auto" onChange={(e) => setZodiacal({ ...zodiacal, radius: e.target.value ? parseFloat(e.target.value) : null })}
                                className="w-full bg-white dark:bg-slate-900 rounded px-2 py-1 border border-slate-300 dark:border-slate-700 text-slate-900 dark:text-slate-100 focus:outline-none focus:border-yellow-500" />
                        </div>
                    </div>
                )}
            </div>

        </div>
    );
}
