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

    // Heuristic: The FIRST element often defines the layer's primary role
    const telescope = elements.find(el => el.type === 'telescope');
    if (telescope) {
        const count = telescope.config?.collectors?.length || 1;
        return { inputs: 0, outputs: count, type: 'source' };
    }

    const camera = elements.find(el => el.type === 'camera');
    if (camera) {
        return { inputs: 1, outputs: 0, type: 'sink' };
    }

    const fiber = elements.find(el => el.type === 'fiber_in');
    if (fiber) {
        const modes = fiber.config?.modes || 1;
        return { inputs: modes, outputs: modes, type: 'transform' };
    }

    // Generic Optical Elements (Lens, Mirror, etc.)
    const count = elements.length;
    return { inputs: count, outputs: count, type: 'generic' };
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
        } else {
            // Generic
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
            edge.type = 'parallel'; // FORCE VISUAL COMPONENT

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
