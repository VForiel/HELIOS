/**
 * Calculates the Input/Output capacity for a generic node based on its internal elements.
 * 
 * @param {Object} node - The ReactFlow node object.
 * @returns {Object} { inputs: number, outputs: number, type: string }
 */
export const calculateNodeIO = (node) => {
    if (!node || !node.data || !node.data.elements) {
        return { inputs: 0, outputs: 0, type: 'unknown' };
    }

    const elements = node.data.elements;
    if (elements.length === 0) {
        return { inputs: 0, outputs: 0, type: 'empty' };
    }

    // Heuristic: Sum capacities for valid element types
    let totalInputs = 0;
    let totalOutputs = 0;
    let type = 'generic'; // Default

    // Check for Scene (Source)
    const scenes = elements.filter(el => el.type === 'scene');
    if (scenes.length > 0) {
        type = 'source';
        totalInputs = 0;
        totalOutputs = 1; // Scene provides 1 wavefront
        return { inputs: totalInputs, outputs: totalOutputs, type };
    }

    // Check for Sources/Splitters (Telescopes)
    const telescopes = elements.filter(el => el.type === 'telescope');
    if (telescopes.length > 0) {
        type = 'splitter'; // Changed from 'source' to 'splitter' (1 In -> N Out)
        totalInputs = 1; // Telescope Array takes 1 wavefront
        totalOutputs = telescopes.reduce((acc, t) => acc + (t.config?.collectors?.length || 1), 0);
    }

    // Check for Sinks (Cameras)
    const cameras = elements.filter(el => el.type === 'camera');
    if (cameras.length > 0) {
        if (type === 'splitter') type = 'transform'; // Mixed?
        else type = 'sink';
        // Assume each camera handles 1 beam unless configured otherwise
        totalInputs = cameras.reduce((acc, c) => acc + (c.config?.inputs || 1), 0);
    }

    // Check for Fibers (Transform)
    const fibers = elements.filter(el => el.type === 'fiber_in');
    if (fibers.length > 0) {
        if (type !== 'splitter' && type !== 'sink') type = 'transform';
        const fiberCap = fibers.reduce((acc, f) => acc + (f.config?.modes || 1), 0);
        totalInputs += fiberCap;
        totalOutputs += fiberCap;
    }

    // Generic Elements (Mirrors, Splitters -> 1:1)
    const generics = elements.filter(el => !['scene', 'telescope', 'camera', 'fiber_in'].includes(el.type));
    if (generics.length > 0) {
        if (type === 'generic') {
            totalInputs += generics.length;
            totalOutputs += generics.length;
        }
    }

    // Fallback if mixed types kept 0
    if (type === 'splitter' && totalOutputs === 0) totalOutputs = 1;
    if (type === 'sink' && totalInputs === 0) totalInputs = 1;
    if (type === 'generic' && totalInputs === 0) {
        totalInputs = elements.length || 0;
        totalOutputs = elements.length || 0;
    }

    return { inputs: totalInputs, outputs: totalOutputs, type };
};

/**
 * Propagates signals through a linear pipeline of nodes.
 * 
 * @param {Array} nodes - List of nodes.
 * @param {Array} edges - List of edges.
 * @returns {Object} { updatedNodes, updatedEdges }
 */
export const propagateSignals = (nodes, edges) => {
    const nodeMap = new Map(nodes.map(n => [n.id, { ...n, data: { ...n.data, io: { incoming: 0, capacity: 0, outgoing: 0, status: 'idle' } } }]));
    const edgeMap = new Map(edges.map(e => [e.id, { ...e, data: { ...e.data, pathCount: 0 } }]));

    const sources = nodes.filter(n => calculateNodeIO(n).type === 'source');

    const queue = [...sources];
    const visited = new Set();

    while (queue.length > 0) {
        const current = queue.shift();
        if (visited.has(current.id)) continue;
        visited.add(current.id);

        const currentNode = nodeMap.get(current.id);
        const io = calculateNodeIO(currentNode);

        // Input Logic
        let incomingCount = 0;
        if (io.type === 'source') {
            incomingCount = io.outputs;
        } else {
            const incomingEdges = edges.filter(e => e.target === current.id);
            for (const e of incomingEdges) {
                const sourceEdge = edgeMap.get(e.id);
                incomingCount += (sourceEdge.data.pathCount || 0);
            }
        }

        // Calculation Logic
        const capacity = io.inputs || io.outputs;
        let outgoingCount = 0;
        let status = 'active';

        // IO Object Construction
        if (io.type === 'source') {
            outgoingCount = io.outputs;
            currentNode.data.io = {
                incoming: 0,
                capacity: 0,
                inputTotal: 0,
                outgoing: outgoingCount,
                outputCapacity: outgoingCount,
                status: 'source'
            };
        } else if (io.type === 'sink') {
            // Camera
            if (incomingCount > capacity) status = 'error';
            else if (incomingCount < capacity) status = 'passive';

            currentNode.data.io = {
                incoming: incomingCount,
                capacity: capacity,
                inputTotal: Math.max(incomingCount, capacity),
                outgoing: 0,
                outputCapacity: 0,
                status
            };
        } else if (io.type === 'splitter') {
            // Telescope Logic: 1 In -> N Out (if input present)
            if (incomingCount > 0) {
                // Determine if we have enough coverage? 
                // Simplification based on User Request: "Takes 1 signal in... N signal out"
                // If we have at least 1 input, we activate all outputs.
                outgoingCount = io.outputs;
                status = 'active';
            } else {
                outgoingCount = 0;
                status = 'passive';
            }

            currentNode.data.io = {
                incoming: incomingCount,
                capacity: io.inputs, // 1
                inputTotal: Math.max(incomingCount, io.inputs),
                outgoing: outgoingCount,
                outputCapacity: io.outputs,
                status
            };

        } else {
            // Generic (1:1 per channel)
            outgoingCount = Math.min(incomingCount, capacity);

            if (incomingCount > capacity) status = 'error';
            else if (incomingCount < capacity) status = 'passive';

            currentNode.data.io = {
                incoming: incomingCount,
                capacity: capacity,
                inputTotal: Math.max(incomingCount, capacity),
                outgoing: outgoingCount,
                outputCapacity: capacity,
                status
            };
        }

        // Propagate to Outgoing Edges
        const outgoingEdges = edges.filter(e => e.source === current.id);
        for (const e of outgoingEdges) {
            const edge = edgeMap.get(e.id);
            edge.data.pathCount = outgoingCount;
            // edge.type = 'parallel'; // Don't force this here let PipelineEditor handle type, just update data

            // Pass Geometry Data to Edge for alignment
            edge.data.sourceCapacity = currentNode.data.io.outputCapacity;

            const targetNode = nodes.find(n => n.id === e.target);
            if (targetNode) {
                const targetIO = calculateNodeIO(targetNode);
                // Input Total = max(incoming, capacity). We know incoming = outgoingCount.
                edge.data.targetCapacity = Math.max(outgoingCount, targetIO.inputs);
                queue.push(targetNode);
            }
        }
    }

    return {
        nodes: Array.from(nodeMap.values()),
        edges: Array.from(edgeMap.values())
    };
};
