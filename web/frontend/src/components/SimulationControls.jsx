import React, { useState, useEffect } from 'react';
import { Calendar, Clock, Lock, Unlock, ChevronUp, ChevronDown, Play, Pause } from 'lucide-react';

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
            return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}:${cs.toString().padStart(2, '0')}`;
        }
        return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}:${cs.toString().padStart(2, '0')}`;
    };

    // Format duration as (hh)h (mm)m
    const formatDuration = () => {
        if (durationHours > 0) {
            return `${durationHours}h ${durationMinutes}m`;
        }
        return `${durationMinutes}m`;
    };

    // Calculate percentage for slider background if needed (optional)
    // const timelinePercentage = totalDurationSeconds > 0 ? (currentSeconds / totalDurationSeconds) * 100 : 0;

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
            <div className={`fixed bottom-0 left-0 right-0 bg-slate-900 border-t border-slate-700 shadow-lg z-40 transition-transform duration-300 ${isExpanded ? 'translate-y-0' : 'translate-y-full'
                }`}>
                {/* Toggle Button - Inside the bar at top-right */}
                <button
                    onClick={() => setIsExpanded(!isExpanded)}
                    className="absolute -top-10 right-4 p-2 bg-slate-900 border border-slate-700 border-b-0 rounded-t-lg hover:bg-slate-800 transition-colors shadow-lg"
                    title={isExpanded ? 'Hide simulation controls' : 'Show simulation controls'}
                >
                    {isExpanded ? <ChevronDown className="w-5 h-5 text-slate-400" /> : <ChevronUp className="w-5 h-5 text-slate-400" />}
                </button>

                <div className="px-4 py-3">
                    <div className="flex items-center justify-between gap-6 max-w-screen-2xl mx-auto">
                        {/* Left section - Temporal Controls */}
                        <div className="flex items-center gap-4 flex-1">
                            {/* Date Picker */}
                            <div className="flex items-center gap-2">
                                <Calendar className="w-4 h-4 text-slate-400" />
                                <input
                                    type="date"
                                    value={observationDate}
                                    onChange={(e) => setObservationDate(e.target.value)}
                                    className="px-2 py-1 text-sm bg-slate-800 border border-slate-700 rounded text-slate-200 focus:outline-none focus:border-blue-500"
                                />
                            </div>

                            {/* Time Picker */}
                            <div className="flex items-center gap-2">
                                <Clock className="w-4 h-4 text-slate-400" />
                                <input
                                    type="time"
                                    step="1"
                                    value={observationTime}
                                    onChange={(e) => setObservationTime(e.target.value)}
                                    className="px-2 py-1 text-sm bg-slate-800 border border-slate-700 rounded text-slate-200 focus:outline-none focus:border-blue-500"
                                />
                            </div>

                            {/* Duration */}
                            <div className="flex items-center gap-2">
                                <span className="text-xs text-slate-400">Duration:</span>
                                <input
                                    type="number"
                                    min="0"
                                    max="24"
                                    value={durationHours}
                                    onChange={(e) => setDurationHours(Math.max(0, parseInt(e.target.value) || 0))}
                                    className="w-12 px-2 py-1 text-sm bg-slate-800 border border-slate-700 rounded text-slate-200 focus:outline-none focus:border-blue-500"
                                />
                                <span className="text-xs text-slate-400">h</span>
                                <input
                                    type="number"
                                    min="0"
                                    max="59"
                                    value={durationMinutes}
                                    onChange={(e) => setDurationMinutes(Math.max(0, Math.min(59, parseInt(e.target.value) || 0)))}
                                    className="w-12 px-2 py-1 text-sm bg-slate-800 border border-slate-700 rounded text-slate-200 focus:outline-none focus:border-blue-500"
                                />
                                <span className="text-xs text-slate-400">m</span>
                            </div>

                            {/* Timeline Slider with Play/Pause */}
                            <div className="flex-1 flex items-center gap-2 min-w-[200px]">
                                {/* Play/Pause Button */}
                                <button
                                    onClick={() => setIsPlaying(!isPlaying)}
                                    disabled={totalDurationMinutes === 0}
                                    className={`p-1.5 rounded transition-colors ${totalDurationMinutes === 0
                                        ? 'bg-slate-800 text-slate-600 cursor-not-allowed'
                                        : isPlaying
                                            ? 'bg-blue-600 hover:bg-blue-500 text-white'
                                            : 'bg-slate-800 hover:bg-slate-700 text-slate-400'
                                        }`}
                                    title={isPlaying ? 'Pause' : 'Play'}
                                >
                                    {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                                </button>

                                <span className="text-xs text-slate-400 whitespace-nowrap font-mono">
                                    {formatTime(currentSeconds)}
                                </span>

                                {/* Slider with hover tooltip */}
                                <div
                                    className="flex-1 relative"
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
                                        className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider-thumb"
                                        disabled={totalDurationMinutes === 0}
                                    />

                                    {/* Tooltip */}
                                    {showTooltip && totalDurationMinutes > 0 && (
                                        <div
                                            className="absolute -top-8 bg-slate-800 text-slate-200 text-xs px-2 py-1 rounded shadow-lg pointer-events-none font-mono whitespace-nowrap"
                                            style={{
                                                left: `${tooltipPosition.x}px`,
                                                transform: 'translateX(-50%)',
                                                opacity: showTooltip ? 1 : 0,
                                                transition: 'opacity 0.15s'
                                            }}
                                        >
                                            {formatTime(tooltipPosition.seconds)}
                                        </div>
                                    )}
                                </div>

                                <span className="text-xs text-slate-400 whitespace-nowrap">
                                    {formatDuration()}
                                </span>
                            </div>
                        </div>

                        {/* Right section - Random Seed Control */}
                        <div className="flex items-center gap-3 border-l border-slate-700 pl-4">
                            <button
                                onClick={() => setSeedLocked(!seedLocked)}
                                className={`p-2 rounded transition-colors ${seedLocked
                                    ? 'bg-amber-600 hover:bg-amber-500 text-white'
                                    : 'bg-slate-800 hover:bg-slate-700 text-slate-400'
                                    }`}
                                title={seedLocked ? 'Seed locked' : 'Seed unlocked'}
                            >
                                {seedLocked ? <Lock className="w-4 h-4" /> : <Unlock className="w-4 h-4" />}
                            </button>
                            <div className="flex items-center gap-2">
                                <span className="text-xs text-slate-400">Seed:</span>
                                {seedLocked ? (
                                    <input
                                        type="number"
                                        value={seedValue}
                                        onChange={(e) => setSeedValue(parseInt(e.target.value) || 0)}
                                        className="w-24 px-2 py-1 text-sm bg-slate-800 border border-slate-700 rounded text-slate-200 focus:outline-none focus:border-blue-500"
                                    />
                                ) : (
                                    <span className="w-24 px-2 py-1 text-sm bg-slate-900 border border-slate-800 rounded text-slate-500 italic">
                                        Dynamique
                                    </span>
                                )}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Custom slider styling */}
                <style jsx>{`
                    .slider-thumb::-webkit-slider-thumb {
                        appearance: none;
                        width: 16px;
                        height: 16px;
                        border-radius: 50%;
                        background: #3b82f6;
                        cursor: pointer;
                        transition: background 0.2s;
                    }
                    .slider-thumb::-webkit-slider-thumb:hover {
                        background: #2563eb;
                    }
                    .slider-thumb::-moz-range-thumb {
                        width: 16px;
                        height: 16px;
                        border-radius: 50%;
                        background: #3b82f6;
                        cursor: pointer;
                        border: none;
                        transition: background 0.2s;
                    }
                    .slider-thumb::-moz-range-thumb:hover {
                        background: #2563eb;
                    }
                    .slider-thumb:disabled::-webkit-slider-thumb {
                        background: #475569;
                        cursor: not-allowed;
                    }
                    .slider-thumb:disabled::-moz-range-thumb {
                        background: #475569;
                        cursor: not-allowed;
                    }
                `}</style>
            </div>
        </>
    );
}
