import React from 'react';
import { getBezierPath } from 'reactflow';

export default function ParallelEdge({
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    style = {},
    markerEnd,
    data
}) {
    const pathCount = data?.pathCount || 1;
    const isBroadcast = data?.broadcast || false;

    // Geometry Data from Logic
    // If not present (e.g. initial load), default to centered bundle
    const sourceTotal = data?.sourceCapacity || pathCount;
    // Broadcast target capacity might be distinct from pathCount
    const targetTotal = data?.targetCapacity || pathCount;

    // Use targetCapacity for render count if broadcasting, otherwise use pathCount
    const renderCount = isBroadcast ? (data?.targetCapacity || 1) : pathCount;

    const paths = [];

    // Bundle settings
    const spacing = 15;

    for (let i = 0; i < renderCount; i++) {
        // Calculate Active Indices (Centered Logic)
        // We select the "Middle" ports of the Source and Target to connect

        let sourceIndex;
        let targetIndex;

        if (isBroadcast) {
            // Broadcast: Fan out from center of source
            sourceIndex = (sourceTotal - 1) / 2;
            // Target: Spread across all available target ports (assuming filling them)
            targetIndex = i + Math.floor((targetTotal - renderCount) / 2);
        } else {
            // Parallel: 1-to-1 mapping with offset
            // Source Index: Starting from (Total - PathCount) / 2
            // Target Index: Starting from (Total - PathCount) / 2

            // We use Math.floor to bias towards Top/Left in odd mismatches, consistent with 0.5 logic usually
            sourceIndex = i + Math.floor((sourceTotal - renderCount) / 2);
            targetIndex = i + Math.floor((targetTotal - renderCount) / 2);
        }

        // Calculate Offsets
        const sourceOffset = (sourceIndex - (sourceTotal - 1) / 2) * spacing;
        const targetOffset = (targetIndex - (targetTotal - 1) / 2) * spacing;

        const [edgePath] = getBezierPath({
            sourceX,
            sourceY: sourceY + sourceOffset,
            sourcePosition,
            targetX,
            targetY: targetY + targetOffset, // Use calculated target Y
            targetPosition,
        });

        paths.push(
            <path
                key={i}
                id={`${id}_${i}`}
                style={{
                    ...style,
                    strokeWidth: 1.5,
                    // Force consistent animation timing for all paths
                    animationDelay: '0s',
                    animationDuration: '1s'
                }}
                className="react-flow__edge-path stroke-slate-400 dark:stroke-slate-500 hover:stroke-blue-500 transition-colors"
                d={edgePath}
                markerEnd={markerEnd}
            />
        );
    }

    return (
        <>
            {paths}
            {/* Invisible wider path for easier interaction/selection */}
            <path
                d={getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition })[0]}
                style={{ strokeWidth: Math.max(20, renderCount * spacing), stroke: 'transparent', fill: 'none', cursor: 'pointer' }}
                className="react-flow__edge-interaction"
            />
        </>
    );
}
