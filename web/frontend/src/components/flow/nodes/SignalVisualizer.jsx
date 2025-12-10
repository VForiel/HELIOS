import React, { useMemo } from 'react';
import { useNodeId, useEdges, Position } from 'reactflow';
import { Circle, Disc } from 'lucide-react';

export default function SignalVisualizer({ capacity = 1 }) {
    const nodeId = useNodeId();
    const edges = useEdges();

    // Calculate incoming signal count
    const incomingCount = useMemo(() => {
        const edge = edges.find(e => e.target === nodeId);
        if (!edge) return 0;
        return edge.data?.pathCount || 1;
    }, [edges, nodeId]);

    if (incomingCount === 0) return null;

    // Generate indicators
    const signals = [];
    const maxDotLimit = 16; // Avoid rendering too many dots

    for (let i = 0; i < Math.min(incomingCount, maxDotLimit); i++) {
        const isActive = i < capacity;
        signals.push(
            <div key={i} title={isActive ? `Active Input ${i + 1}` : `Input ${i + 1} (Unused)`}>
                {isActive ? (
                    <div className="w-2.5 h-2.5 bg-blue-500 rounded-full shadow-[0_0_5px_rgba(59,130,246,0.6)] animate-pulse" />
                ) : (
                    <div className="w-2.5 h-2.5 border-2 border-slate-300 dark:border-slate-600 rounded-full border-dashed" />
                )}
            </div>
        );
    }

    if (incomingCount > maxDotLimit) {
        signals.push(<span key="more" className="text-[10px] text-slate-400">+{incomingCount - maxDotLimit}</span>)
    }

    return (
        <div className="flex flex-col gap-1 mb-3 px-1">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest flex justify-between">
                <span>Signal Match</span>
                <span>{Math.min(incomingCount, capacity)} / {incomingCount}</span>
            </span>
            <div className="flex flex-wrap gap-1.5 p-2 bg-slate-50 dark:bg-slate-900/50 rounded border border-slate-100 dark:border-slate-800">
                {signals}
            </div>
        </div>
    );
}
