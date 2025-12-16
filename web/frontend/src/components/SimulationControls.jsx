import React, { useState, useEffect } from 'react';
import { Calendar, Clock, Lock, Unlock, ChevronUp, ChevronDown, Play, Pause } from 'lucide-react';
import NumberInput from './NumberInput';

export default function SimulationControls({ onExpandChange }) {
    // Temporal controls state
    const [observationDate, setObservationDate] = useState('2024-01-01');
    const [observationTime, setObservationTime] = useState('00:00:00');
    const [durationHours, setDurationHours] = useState(1);
    const [durationMinutes, setDurationMinutes] = useState(0);
    const [currentSeconds, setCurrentSeconds] = useState(0); // Current time in seconds (float)

    const [isPlaying, setIsPlaying] = useState(false);

    // Random seed control state
    const [seedLocked, setSeedLocked] = useState(false);
    const [seedValue, setSeedValue] = useState(42);

    // Collapse state - hidden by default
    const [isExpanded, setIsExpanded] = useState(false);

    // Tooltip state for slider hover
    const [showTooltip, setShowTooltip] = useState(false);
    const [tooltipPosition, setTooltipPosition] = useState({ x: 0, seconds: 0 });

    // Notify parent when expand state changes
    useEffect(() => {
        if (onExpandChange) {
            onExpandChange(isExpanded);
        }
    }, [isExpanded, onExpandChange]);

    // Calculate total duration in minutes for slider
    const totalDurationMinutes = durationHours * 60 + durationMinutes;
    const totalDurationSeconds = totalDurationMinutes * 60;

    // Auto-advance timeline when playing (real-time)
    useEffect(() => {
        if (!isPlaying || totalDurationSeconds === 0) return;

        const interval = setInterval(() => {
            setCurrentSeconds(prev => {
                const next = prev + 0.01;
                if (next >= totalDurationSeconds) {
                    setIsPlaying(false);
                    return totalDurationSeconds;
                }
                return next;
            });
        }, 10); // Update every 10ms

        return () => clearInterval(interval);
    }, [isPlaying, totalDurationSeconds]);


    // Format time as (hh:)mm:ss:cc
    const formatTime = (totalSeconds) => {
        const seconds = Math.floor(totalSeconds);
        const centiseconds = Math.floor((totalSeconds % 1) * 100);
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        const cs = centiseconds;

        if (hours > 0) {
            return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}:${String(cs).padStart(2, '0')}`;
        }
        return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}:${String(cs).padStart(2, '0')}`;
    };

    // Format duration as (hh)h (mm)m
    const formatDuration = () => {
        if (durationHours > 0) {
            return `${durationHours}h ${durationMinutes}m`;
        }
        return `${durationMinutes}m`;
    };

    // Calculate percentage for slider background
    const timelinePercentage = totalDurationSeconds > 0 ? (currentSeconds / totalDurationSeconds) * 100 : 0;

    // Handle slider hover for tooltip
    const handleSliderMouseMove = (e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percentage = (x / rect.width);
        const seconds = percentage * totalDurationSeconds;
        setTooltipPosition({ x: e.clientX - rect.left, seconds: Math.max(0, Math.min(seconds, totalDurationSeconds)) });
    };

    return (
        <>
            {/* Bottom Bar - Collapsible */}
            <div className={`fixed bottom-0 left-0 right-0 bg-slate-900/95 backdrop-blur-md border-t border-slate-700 shadow-2xl z-40 transition-transform duration-300 ${isExpanded ? 'translate-y-0' : 'translate-y-full'
                }`}>
                {/* Toggle Button - Inside the bar at top-right */}
                <button
                    onClick={() => setIsExpanded(!isExpanded)}
                    className="absolute -top-10 right-4 p-2 bg-slate-900 border border-slate-700 border-b-0 rounded-t-lg hover:bg-slate-800 transition-colors shadow-lg group"
                    title={isExpanded ? 'Hide simulation controls' : 'Show simulation controls'}
                >
                    {isExpanded ? <ChevronDown className="w-5 h-5 text-slate-400 group-hover:text-white" /> : <ChevronUp className="w-5 h-5 text-slate-400 group-hover:text-white" />}
                </button>

                <div className="px-6 py-4">
                    <div className="flex items-center justify-between gap-8 max-w-screen-2xl mx-auto">
                        {/* Left section - Temporal Controls */}
                        <div className="flex items-center gap-6 flex-1">

                            {/* Time Config Group */}
                            <div className="flex items-center gap-4 bg-slate-800/50 p-2 rounded-lg border border-slate-700/50">
                                {/* Date Picker */}
                                <div className="flex items-center gap-2">
                                    <Calendar className="w-4 h-4 text-purple-400" />
                                    <input
                                        type="date"
                                        value={observationDate}
                                        onChange={(e) => setObservationDate(e.target.value)}
                                        className="px-2 py-1 text-sm bg-slate-800 border border-slate-600 rounded text-slate-200 focus:outline-none focus:border-blue-500 hover:border-slate-500 transition-colors"
                                    />
                                </div>

                                {/* Time Picker */}
                                <div className="flex items-center gap-2">
                                    <Clock className="w-4 h-4 text-purple-400" />
                                    <input
                                        type="time"
                                        step="1"
                                        value={observationTime}
                                        onChange={(e) => setObservationTime(e.target.value)}
                                        className="px-2 py-1 text-sm bg-slate-800 border border-slate-600 rounded text-slate-200 focus:outline-none focus:border-blue-500 hover:border-slate-500 transition-colors"
                                    />
                                </div>
                            </div>

                            {/* Duration */}
                            <div className="flex items-center gap-2 bg-slate-800/50 p-2 rounded-lg border border-slate-700/50">
                                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider mr-1">Duration</span>

                                <div className="flex items-center gap-1">
                                    <NumberInput
                                        value={durationHours}
                                        onChange={(val) => setDurationHours(Math.max(0, parseInt(val) || 0))}
                                        min={0}
                                        max={24}
                                        size="sm"
                                        className="!w-20 !bg-slate-900 !border-slate-700"
                                    />
                                    <span className="text-xs text-slate-500 font-mono">h</span>
                                </div>

                                <div className="flex items-center gap-1">
                                    <NumberInput
                                        value={durationMinutes}
                                        onChange={(val) => setDurationMinutes(Math.max(0, Math.min(59, parseInt(val) || 0)))}
                                        min={0}
                                        max={59}
                                        size="sm"
                                        className="!w-20 !bg-slate-900 !border-slate-700"
                                    />
                                    <span className="text-xs text-slate-500 font-mono">m</span>
                                </div>
                            </div>

                            {/* Timeline Slider with Play/Pause */}
                            <div className="flex-1 flex items-center gap-4 min-w-[300px] bg-slate-800/30 p-2 rounded-xl border border-slate-700/30">
                                {/* Play/Pause Button */}
                                <button
                                    onClick={() => setIsPlaying(!isPlaying)}
                                    disabled={totalDurationMinutes === 0}
                                    className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center transition-all shadow-lg ${totalDurationMinutes === 0
                                        ? 'bg-slate-800 text-slate-600 cursor-not-allowed opacity-50'
                                        : isPlaying
                                            ? 'bg-gradient-to-tr from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white shadow-blue-500/30'
                                            : 'bg-slate-700 hover:bg-slate-600 text-slate-200 hover:text-white'
                                        }`}
                                    title={isPlaying ? 'Pause' : 'Play'}
                                >
                                    {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current ml-0.5" />}
                                </button>

                                <div className="flex-1 relative flex flex-col justify-center h-10">
                                    <div className="flex justify-between items-end mb-1 px-1">
                                        <span className="text-xs text-blue-400 font-mono font-bold tracking-tight">
                                            {formatTime(currentSeconds)}
                                        </span>
                                        <span className="text-[10px] text-slate-500 font-mono">
                                            {formatDuration()}
                                        </span>
                                    </div>

                                    {/* Slider with hover tooltip */}
                                    <div
                                        className="relative h-4 group"
                                        onMouseEnter={() => setShowTooltip(true)}
                                        onMouseLeave={() => setShowTooltip(false)}
                                        onMouseMove={handleSliderMouseMove}
                                    >
                                        <input
                                            type="range"
                                            min="0"
                                            max={totalDurationSeconds}
                                            step="0.01"
                                            value={currentSeconds}
                                            onChange={(e) => {
                                                const newSeconds = parseFloat(e.target.value);
                                                setCurrentSeconds(newSeconds);
                                            }}
                                            className="w-full h-1.5 rounded-full appearance-none cursor-pointer focus:outline-none slider-track"
                                            style={{
                                                background: `linear-gradient(to right, #3b82f6 ${timelinePercentage}%, #1e293b ${timelinePercentage}%)`
                                            }}
                                            disabled={totalDurationMinutes === 0}
                                        />

                                        {/* Tooltip */}
                                        {showTooltip && totalDurationMinutes > 0 && (
                                            <div
                                                className="absolute -top-10 bg-slate-800 text-white text-[10px] px-2 py-1 rounded shadow-xl border border-slate-600 pointer-events-none font-mono whitespace-nowrap z-50"
                                                style={{
                                                    left: `${tooltipPosition.x}px`,
                                                    transform: 'translateX(-50%)',
                                                    opacity: showTooltip ? 1 : 0,
                                                    transition: 'opacity 0.15s'
                                                }}
                                            >
                                                {formatTime(tooltipPosition.seconds)}
                                                <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-slate-800 border-b border-r border-slate-600 rotate-45"></div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Right section - Random Seed Control */}
                        <div className="flex items-center gap-3 border-l border-slate-700 pl-6">
                            <button
                                onClick={() => setSeedLocked(!seedLocked)}
                                className={`p-2 rounded-lg transition-colors border ${seedLocked
                                    ? 'bg-amber-500/10 border-amber-500/50 text-amber-500 hover:bg-amber-500/20'
                                    : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-200 hover:border-slate-600'
                                    }`}
                                title={seedLocked ? 'Seed locked' : 'Seed unlocked'}
                            >
                                {seedLocked ? <Lock className="w-4 h-4" /> : <Unlock className="w-4 h-4" />}
                            </button>
                            <div className="flex flex-col">
                                <span className="text-[10px] uppercase text-slate-500 font-semibold mb-0.5">RNG Seed</span>
                                {seedLocked ? (
                                    <NumberInput
                                        value={seedValue}
                                        onChange={(val) => setSeedValue(parseInt(val) || 0)}
                                        size="xs"
                                        className="!w-24 !bg-slate-900 !border-slate-700"
                                    />
                                ) : (
                                    <div className="h-6 w-24 flex items-center justify-center text-xs bg-slate-900 border border-slate-800 rounded text-slate-600 italic select-none">
                                        Dynamic
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Custom slider styling */}
                <style jsx>{`
                    /* Webkit */
                    input[type=range]::-webkit-slider-thumb {
                        -webkit-appearance: none;
                        height: 14px;
                        width: 14px;
                        border-radius: 50%;
                        background: #ffffff;
                        cursor: pointer;
                        box-shadow: 0 0 0 1px rgba(0,0,0,0.1), 0 2px 4px rgba(0,0,0,0.2);
                        margin-top: -5px; /* Adjust for track height difference */
                        transition: transform 0.1s, box-shadow 0.1s;
                    }
                    input[type=range]::-webkit-slider-thumb:hover {
                        transform: scale(1.2);
                        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.3);
                    }
                    input[type=range]::-webkit-slider-runnable-track {
                        width: 100%;
                        height: 4px;
                        cursor: pointer;
                        border-radius: 2px;
                        /* Background handled inline */
                    }

                    /* Firefox */
                    input[type=range]::-moz-range-thumb {
                        height: 14px;
                        width: 14px;
                        border: none;
                        border-radius: 50%;
                        background: #ffffff;
                        cursor: pointer;
                        box-shadow: 0 0 0 1px rgba(0,0,0,0.1), 0 2px 4px rgba(0,0,0,0.2);
                        transition: transform 0.1s, box-shadow 0.1s;
                    }
                    input[type=range]::-moz-range-thumb:hover {
                        transform: scale(1.2);
                        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.3);
                    }
                    /* Track background in Firefox is trickier with gradient, relying on inline style works for many cases but -moz-range-progress is better native solution */
                    /* But keeping inline style for consistency across browsers mostly works */
                `}</style>
            </div>
        </>
    );
}
