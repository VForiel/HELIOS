import React, { useMemo } from 'react';
import { useNodeId, useEdges, Position } from 'reactflow';
import { Circle, Disc, XCircle } from 'lucide-react';

export default function SignalVisualizer({ config, layout = 'horizontal', spacing = 15, isInput = true }) {
    // Config: { total, capacity, incoming }
    // spacing: center-to-center distance in pixels

    // Spacing Visualization:
    // With 12px items:
    // [12px]
    // Gap = Spacing - 12 (must be >= 0)
    // [12px]
    // Center to center distance = 6 + Gap + 6 = 12 + Gap = 12 + (S - 12) = S. Correct.

    const total = config?.total || 0;
    const capacity = config?.capacity || 0;
    const incoming = config?.incoming || 0;

    if (total === 0) return null;

    // Vertical Layout aligned with edges (Flexbox Implementation)
    if (layout === 'vertical') {
        const signals = [];

        // Strict Top-to-Bottom Ordering Logic
        // 1. Active Signals match first (min(total signals, capacity))
        // 2. Excess Signals match next (if signals > capacity) -> Error
        // 3. Excess Capacity matches last (if capacity > signals) -> Passive

        // We iterate 0..Total to render the slots.
        // Total = max(incoming, capacity)

        for (let i = 0; i < total; i++) {
            let state = 'passive';

            if (isInput) {
                // Input Logic:
                if (i < capacity && i < incoming) {
                    state = 'active'; // Matched (Exists in both)
                } else if (i >= capacity && i < incoming) {
                    state = 'error'; // Overload (Exists in Incoming, not Capacity)
                } else {
                    state = 'passive'; // Unused (Exists in Capacity, not Incoming)
                }
            } else {
                // Output Logic:
                // Generally, if signal exists (incoming), it is propagated.
                // But visualized as "matched" if within capacity?
                if (i < capacity && i < incoming) {
                    state = 'active';
                } else if (i >= capacity && i < incoming) {
                    state = 'active'; // Actually, purely signal driven for generic output?
                    // If 4 signals -> 2 lenses -> 2 outputs.
                    // outputCapacity = 2. outgoing = 2.
                    // So total = 2. i=0,1 (active).
                    // Overload implies input > capacity. The outputs are capped.
                } else {
                    state = 'passive';
                }
            }

            signals.push(
                <div
                    key={i}
                    title={`${isInput ? 'Input' : 'Output'} ${i + 1} (${state})`}
                    className="flex items-center justify-center transition-all duration-500"
                    style={{
                        width: '12px',
                        height: '12px'
                    }}
                >
                    {state === 'active' && (
                        <div className={`w-2.5 h-2.5 rounded-full shadow-[0_0_5px_rgba(59,130,246,0.6)] ${isInput ? 'bg-blue-500' : 'bg-purple-500'}`} />
                    )}
                    {state === 'passive' && (
                        <div className="w-2.5 h-2.5 border-2 border-slate-300 dark:border-slate-600 rounded-full bg-slate-100 dark:bg-slate-900 border-dashed" />
                    )}
                    {state === 'error' && (
                        <XCircle className="w-3 h-3 text-red-500" />
                    )}
                </div>
            );
        }

        // Flexbox container
        const gap = Math.max(0, spacing - 12);

        return (
            <div
                className="flex flex-col items-center justify-center pointer-events-none"
                style={{ gap: `${gap}px` }}
            >
                {signals}
            </div>
        );
    }

    return null;
}
