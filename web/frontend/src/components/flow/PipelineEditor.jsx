import React, { useState, useCallback, useRef, useMemo, useEffect } from 'react';
import ReactFlow, {
    ReactFlowProvider,
    addEdge,
    updateEdge,
    useNodesState,
    useEdgesState,
    Controls,
    ControlButton,
    Background,
    MiniMap,
    useReactFlow,
    Panel
} from 'reactflow';
import 'reactflow/dist/style.css';
import TelescopeNode from './nodes/TelescopeNode';
import CameraNode from './nodes/CameraNode';
import GenericNode from './nodes/GenericNode';
import { Menu, Sun, Moon, Heart, Github, Book, Download, Upload, Cpu, Disc, Divide, GitFork, Zap, Activity, Hand, MousePointer2, Stars, Search, Camera, CloudFog, X, Code, Languages } from 'lucide-react';
import { getElementIcon, getPhotonicIcon } from '../../utils/iconMap';

import LayerNode from './nodes/LayerNode';
import ParallelEdge from './edges/ParallelEdge';
import CodeViewer from '../CodeViewer';

const nodeTypes = {
    layer: LayerNode
};

const edgeTypes = {
    parallel: ParallelEdge
};

let id = 1;
const getId = () => `node_${id++}`;

// History Helper
const useUndoRedo = (initialNodes, initialEdges) => {
    const [past, setPast] = useState([]);
    const [future, setFuture] = useState([]);

    const takeSnapshot = useCallback((nodes, edges) => {
        setPast(old => {
            const newPast = [...old, { nodes, edges }];
            if (newPast.length > 50) newPast.shift(); // Limit to 50
            return newPast;
        });
        setFuture([]);
    }, []);

    const canUndo = past.length > 0;
    const canRedo = future.length > 0;

    return { past, setPast, future, setFuture, takeSnapshot, canUndo, canRedo };
};

export default function PipelineEditor({
    stars, setStars,
    planets, setPlanets,
    zodiacal, setZodiacal,
    atmosphere, setAtmosphere,
    telescope, setTelescope,
    camera, setCamera,
    runSimulation,
    onToggleSidebar,
    onToggleTheme,
    isDark,
    language,
    setLanguage,
    languages,
    t
}) {
    const reactFlowWrapper = useRef(null);
    const edgeUpdateSuccessful = useRef(true);
    const [reactFlowInstance, setReactFlowInstance] = useState(null);
    const [clipboard, setClipboard] = useState(null);
    const [interactionMode, setInteractionMode] = useState('nav'); // 'nav' | 'select'

    // Inspect Modal State
    const [inspectData, setInspectData] = useState(null); // { image: blobUrl, title: string }
    const [inspectLoading, setInspectLoading] = useState(false);

    // Language Dropdown State
    const [isLanguageDropdownOpen, setIsLanguageDropdownOpen] = useState(false);

    // Toast Notification State
    const [toasts, setToasts] = useState([]); // Array of { id, message, type }

    const showToast = useCallback((message, type = 'error') => {
        const id = Date.now();
        setToasts(prev => [...prev, { id, message, type }]);
        // Auto-remove after 5 seconds
        setTimeout(() => {
            setToasts(prev => prev.filter(t => t.id !== id));
        }, 5000);
    }, []);


    // Initial Nodes (Layers)
    const initialNodes = [
        {
            id: 'layer-1',
            type: 'layer',
            position: { x: 50, y: 100 },
            data: {
                elements: [
                    { type: 'scene', label: 'Scene', config: { stars, planets, zodiacal }, iconPath: getElementIcon('scene') }
                ]
            }
        },
        {
            id: 'layer-2',
            type: 'layer',
            position: { x: 500, y: 100 },
            data: {
                elements: [
                    { type: 'telescope', label: 'Telescope', config: telescope, iconPath: getElementIcon('telescope') }
                ]
            }
        },
        {
            id: 'layer-3',
            type: 'layer',
            position: { x: 950, y: 100 },
            data: {
                elements: [
                    { type: 'camera', label: 'Camera', config: camera, iconPath: getElementIcon('camera') }
                ]
            }
        }
    ];

    const initialEdges = [
        { id: 'e1-2', source: 'layer-1', target: 'layer-2', animated: true },
        { id: 'e2-3', source: 'layer-2', target: 'layer-3', animated: true },
    ];

    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

    // Sync Edges with Telescope Config (Parallel Paths)
    // Sync Edges and Calculate IO (Global Propagation)
    // This effect runs whenever nodes or edges structure changes to ensure consistency
    useEffect(() => {
        import('../../utils/optical_logic').then(({ propagateSignals }) => {
            const { nodes: newNodes, edges: newEdges } = propagateSignals(nodes, edges);

            // Check for differences to avoid infinite loop
            let nodesChanged = false;
            let edgesChanged = false;

            // Simple diffing logic 
            const updatedEdges = newEdges.map((ne, i) => {
                const old = edges.find(e => e.id === ne.id);
                if (!old || (old.data?.pathCount !== ne.data.pathCount) || (old.type !== ne.type)) {
                    edgesChanged = true;
                    return ne; // Use new edge
                }
                return old; // Keep old reference
            });
            // If length changed (e.g. edge added/removed outside this logic), setEdges triggers anyway. 
            // We focus on data updates here.

            const updatedNodes = newNodes.map((nn, i) => {
                const old = nodes.find(n => n.id === nn.id);
                // Compare IO state
                const newIO = nn.data.io;
                const oldIO = old?.data?.io;

                // Deep comparison for IO object
                if (!oldIO || JSON.stringify(oldIO) !== JSON.stringify(newIO)) {
                    nodesChanged = true;
                    return nn;
                }
                return old;
            });

            if (edgesChanged) {
                console.log('[PipelineEditor] Propagating signals: Edges updated.');
                setEdges(updatedEdges);
            }

            if (nodesChanged) {
                console.log('[PipelineEditor] Propagating signals: Nodes IO updated.');
                setNodes(updatedNodes);
            }
        });
    }, [
        // Dependencies to trigger recalculation
        nodes.map(n => JSON.stringify(n.data.elements)).join(','),
        edges.map(e => e.source + e.target).join(','),
        setNodes,
        setEdges
    ]);



    // OLD LOGIC COMMENTED OUT
    /* 
    useEffect(() => {
        const collectorCount = telescope.collectors ? telescope.collectors.length : 1;
        console.log('[PipelineEditor] Telescope update detected. Collectors:', collectorCount);
    
        setEdges(eds => eds.map(e => {
            const sourceNode = nodes.find(n => n.id === e.source);
            // Check if source node is a layer containing a telescope element
            const hasTelescope = sourceNode?.data?.elements?.some(el => el.type === 'telescope');
    
            if (hasTelescope) {
                console.log(`[PipelineEditor] Checking edge ${e.id} from telescope source. Current pathCount: ${e.data?.pathCount}, Target: ${collectorCount}`);
                if (e.data?.pathCount !== collectorCount || e.type !== 'parallel') {
                    console.log(`[PipelineEditor] Updating edge ${e.id} to parallel with pathCount ${collectorCount}`);
                    return {
                        ...e,
                        type: 'parallel',
                        data: { ...e.data, pathCount: collectorCount }
                    };
                }
            }
            return e;
        }));
    }, [telescope.collectors, nodes, setEdges]); */

    // Undo/Redo Hook
    const { past, setPast, future, setFuture, takeSnapshot, canUndo, canRedo } = useUndoRedo(initialNodes, initialEdges);

    const undo = useCallback(() => {
        if (!canUndo) return;
        const current = { nodes, edges };
        const previous = past[past.length - 1];
        const newPast = past.slice(0, past.length - 1);

        setPast(newPast);
        setFuture([current, ...future]);
        setNodes(previous.nodes);
        setEdges(previous.edges);
    }, [nodes, edges, past, future, canUndo, setNodes, setEdges, setPast, setFuture]);

    const redo = useCallback(() => {
        if (!canRedo) return;
        const current = { nodes, edges };
        const next = future[0];
        const newFuture = future.slice(1);

        setPast([...past, current]);
        setFuture(newFuture);
        setNodes(next.nodes);
        setEdges(next.edges);
    }, [nodes, edges, past, future, canRedo, setNodes, setEdges, setPast, setFuture]);

    // Snapshot Trigger
    const registerChange = useCallback(() => {
        takeSnapshot(nodes, edges);
    }, [nodes, edges, takeSnapshot]);

    // Copy/Paste Logic
    const handleCopy = useCallback(() => {
        const selected = nodes.filter(n => n.selected);
        if (selected.length > 0) {
            setClipboard(selected);
            console.log("Copied", selected.length, "nodes");
        }
    }, [nodes]);

    const handlePaste = useCallback(() => {
        if (!clipboard || clipboard.length === 0) return;
        registerChange(); // Snapshot before paste

        const newNodes = clipboard.map(node => {
            const newId = getId();
            return {
                ...node,
                id: newId,
                position: { x: node.position.x + 50, y: node.position.y + 50 },
                selected: true,
                data: { ...node.data } // Deep copy needed if nested, spread is shallow but okay for simple config? 
                // GenericNode checks 'fields' in data
            };
        });

        // Deselect current
        setNodes(nds => nds.map(n => ({ ...n, selected: false })).concat(newNodes));
    }, [clipboard, registerChange, setNodes]);



    // Inspect Logic
    const handleInspect = async (nodeId) => {
        try {
            setInspectLoading(true);
            setInspectData(null);

            // 1. Get Linear Pipeline
            const pipeline = getPipeline();

            // 2. Find Index of Node in Linear Pipeline
            // We need to trace which "linear index" corresponds to this nodeId.
            // Simplified Assumption: The order in getPipeline() matches the topological order.
            // But complex branching makes this tricky.
            // BETTER APPROACH:
            // Send the node ID to backend? But backend doesn't know our graph IDs.
            // Refined Logic:
            // The getPipeline() returns a nested list structure for parallel branches.
            // The backend flattens this. 
            // We need to match the flattening logic to find the index.

            // For now, let's just find the index of the node in the `nodes` array? No, that's arbitrary.
            // Let's rely on the fact that `getPipeline` traverses from Root.
            // We need to find "Where is this node in the flattened execution list?"

            // Let's implement a quick helper to flatten and find index.
            let targetIndex = -1;
            let currentIndex = 0;

            const traverseAndFind = (list) => {
                // Using the same logic as backend's "flat_layers.append(layer_obj)"
                // Backend iterates: for layer_conf in request.layers:
                // if nested? Backend currently iterates top level.

                // Wait, backend `request.layers` is `List[LayerConfig]`.
                // `getPipeline` returns a list where items can be arrays (parallel).
                // Does backend support nested lists in `PipelineRequest`?
                // Checking app.py... `class PipelineRequest(BaseModel): layers: List[LayerConfig]`
                // And `LayerConfig` has `type`, `config`. 
                // If `getPipeline` returns nested arrays, Pydantic might fail or flatten it if configured?
                // Let's check `getPipeline` output structure again.
                // It returns `[layerElements, branches]`.
                // This seems to NOT match `List[LayerConfig]` directly if strict.

                // Assuming the current backend `run_pipeline` iterates flatly, 
                // `getPipeline` likely generates a flat list for linear cases,
                // but for parallel?
                // The backend flat_layers used in `inspect_node` iterates `request.layers` directly.

                // Let's assume we flatten the pipeline before sending.
                // We will send a Flattened version where parallel branches are serialized sequentially?
                // Or we just send the linear path UP TO the target node?
                // "Inspet Node" implies inspecting the state AT that node.

                // Simple Robust Strategy:
                // 1. Find the path from Root to Target Node.
                // 2. Construct a pipeline of just that path.
                // 3. Send that to `/api/simulate` (or inspect endpoint) and get the LAST layer's output.
                // This avoids index confusion and handles branching implicitly by just picking one path.
                // BUT, if it's a combiner node, it needs all inputs.

                // Revised Strategy:
                // 1. Serialize the FULL graph to a flat list (Topological Sort).
                // 2. Find the index of the target node in that sort.
                // 3. Send flat list + index to backend.

                // Implementation of Topological Sort / Flattening:
                // We reuse `getPipeline` but we need to know which item in the result corresponds to our `nodeId`.

                // HACK for now: Flatten the graph based on visual position or edge traversal?
                // `getPipeline` seems to try to build a structure.
                // Let's just traverse and count.

            };

            // Hacky Index Finding based on `getPipeline` recursion order:
            // We will modify `getPipeline` or traverse its result?
            // Since `getPipeline` returns config objects, we lose the ID.

            // Let's just try to map IDs to the output of getPipeline.
            // We will re-run the `buildChain` logic but keep IDs.

            const startNodes = nodes.filter(n =>
                n.data.elements && n.data.elements.some(el => el.type === 'scene')
            );
            const root = startNodes.length > 0 ? startNodes[0] : nodes[0];

            if (!root) throw new Error("No Scene Found");

            let flatList = [];
            let visited = new Set();
            let foundIndex = -1;

            const traverse = (node) => {
                if (!node) return;
                visited.add(node.id);

                // Add this layer to list
                if (node.id === nodeId) {
                    foundIndex = flatList.length;
                }
                flatList.push(node); // Placeholder

                const outEdges = edges.filter(e => e.source === node.id);
                if (outEdges.length > 0) {
                    // If multiple, which one first?
                    // Depth first
                    const nextNodes = outEdges.map(e => nodes.find(n => n.id === e.target)).filter(n => n && !visited.has(n.id));
                    nextNodes.forEach(traverse);
                }
            };

            traverse(root);

            if (foundIndex === -1) {
                // Maybe node is disconnected? 
                // Or maybe it IS the root? (Included in logic)
                console.warn("Target node not found in traversal", nodeId);
                // Fallback if disconnected: just inspect it alone? (Won't verify inputs)
                return;
            }

            targetIndex = foundIndex;
            const finalPipeline = getPipeline(); // This needs to match the traversal order!
            // `getPipeline` currently handles branching by returning nested arrays?
            // If the backend expects a flat list, we should FLATTEN `finalPipeline` similarly.

            const flatPipelineConfig = finalPipeline.flat(Infinity);
            // Note: `getPipeline` builds `[layerElements]`. layerElements is a list of dicts.
            // So `flat(Infinity)` will give a list of dicts (elements).
            // This assumes 1 Node = 1 Layer = 1 Element in backend?
            // Backend `inspect_node` iterates layers.
            // `LayerNode` has `elements` (array). 
            // When we flattened, did we preserve the "Node = Layer" grouping?
            // `getPipeline` logic: `return [layerElements, ...]`
            // layerElements is `[{type, config}, {type, config}]`.
            // So one Node can trigger MULTIPLE backend layers.

            // We need to count ELEMENTS, not Nodes.
            // Recalculate index based on elements count.

            let elementIndex = 0;
            let targetElementIndex = -1;
            visited = new Set();

            const traverseElements = (node) => {
                if (!node) return;
                visited.add(node.id);

                const nodeElements = node.data.elements || [];
                if (node.id === nodeId) {
                    // We want the output of the WHOLE node (last element in it)
                    // So index is current + length - 1
                    targetElementIndex = elementIndex + nodeElements.length - 1;
                }

                elementIndex += nodeElements.length;

                const outEdges = edges.filter(e => e.source === node.id);
                if (outEdges.length > 0) {
                    // Sorting to ensure deterministic order matching `getPipeline`
                    // `getPipeline` filters: `targets = outEdges.map...`
                    // It uses default edge sort? `edges` array order.
                    const nextNodes = outEdges
                        .map(e => nodes.find(n => n.id === e.target))
                        .filter(n => n && !visited.has(n.id));
                    nextNodes.forEach(traverseElements);
                }
            };

            traverseElements(root);

            console.log(`Inspect: Node ${nodeId} -> Element Index ${targetElementIndex}`);

            // Send request
            const payload = { mode: 'pipeline', layers: flatPipelineConfig };
            const response = await fetch(`/api/inspect_node?target_index=${targetElementIndex}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error("Inspection Failed");

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);

            setInspectData({
                image: url,
                title: `Node Inspection`
            });

        } catch (e) {
            console.error(e);
            alert("Inspection Error: " + e.message);
        } finally {
            setInspectLoading(false);
        }
    };

    // Inject handleInspect into nodes
    useEffect(() => {
        setNodes((nds) => nds.map(node => {
            // Avoid infinite loop by checking if func already set? 
            // React state updates might be safe, but best to be careful.
            // We just overwrite it.
            return { ...node, data: { ...node.data, onInspect: handleInspect } };
        }));
    }, [nodes.length]); // Updating on structure change mostly. Or just once? 
    // Ideally we pass it in `initialNodes` or via `onNodesChange` interception.
    // But `setNodes` inside `useEffect` with dependency on `nodes` is DANGEROUS (loop).
    // Let's use a ref or memoized nodes?
    // BETTER: The `nodes` state in `PipelineEditor` is source of truth.
    // We should update the data there ONCE or pass it via context?
    // For now, let's update it in the `useMemo` block where we update other data.

    // --> Moving this logic to line 260-275 block.

    // Update node data when props change + Inspect Handler
    useMemo(() => {
        setNodes((nds) =>
            nds.map((node) => {
                let newData = { ...node.data, onInspect: handleInspect, t };
                if (node.type === 'scene') {
                    newData = { ...newData, stars, setStars, planets, setPlanets, zodiacal, setZodiacal };
                } else if (node.type === 'atmosphere') {
                    newData = { ...newData, config: atmosphere, setConfig: setAtmosphere };
                } else if (node.type === 'telescope') {
                    newData = { ...newData, config: telescope, setConfig: setTelescope };
                } else if (node.type === 'camera') {
                    newData = { ...newData, config: camera, setConfig: setCamera };
                }
                return { ...node, data: newData };
            })
        );
    }, [stars, planets, zodiacal, atmosphere, telescope, camera, setNodes, t]); // Added t dependency




    // Layer type mapping (must match backend)
    const getLayerType = useCallback((elementType) => {
        const mapping = {
            'scene': 'GenerationLayer',
            'atmosphere': 'GenerationLayer',
            'telescope': 'SamplingLayer',
            'lens': 'OpticalLayer',
            'beam_splitter': 'OpticalLayer',
            'coronagraph': 'OpticalLayer',
            'fiber_in': 'OpticalLayer',
            'fiber_out': 'OpticalLayer',
            'mmi': 'OpticalLayer',
            'photonic': 'OpticalLayer',
            'camera': 'DetectionLayer',
        };
        return mapping[elementType] || 'Layer';
    }, []);

    // Track invalid connection attempts
    const lastInvalidConnection = useRef(null);

    // Comprehensive connection validation with layer types (NO TOAST - only returns boolean)
    const isValidConnection = useCallback((connection) => {
        // Prevent self-loops
        if (connection.source === connection.target) {
            lastInvalidConnection.current = {
                sourceType: 'self',
                targetType: 'self',
                message: t('validation.selfLoop')
            };
            return false;
        }

        // Get source and target nodes
        const sourceNode = nodes.find(n => n.id === connection.source);
        const targetNode = nodes.find(n => n.id === connection.target);

        if (!sourceNode || !targetNode) return false;

        // Determine layer types from elements
        const getNodeLayerType = (node) => {
            if (!node.data.elements || node.data.elements.length === 0) return null;
            // Use the first element's layer type (nodes should be homogeneous)
            return getLayerType(node.data.elements[0].type);
        };

        const sourceType = getNodeLayerType(sourceNode);
        const targetType = getNodeLayerType(targetNode);

        if (!sourceType || !targetType) return true; // Allow if types unknown (fallback)

        // Strict validation rules matching backend
        const validConnections = {
            'GenerationLayer': ['GenerationLayer', 'SamplingLayer'],
            'SamplingLayer': ['OpticalLayer', 'DetectionLayer'],  // Can connect to BOTH
            'OpticalLayer': ['OpticalLayer', 'DetectionLayer'],
            'DetectionLayer': ['DataLayer'],
            'DataLayer': ['DataLayer'],
            'Layer': ['Layer', 'GenerationLayer', 'SamplingLayer', 'OpticalLayer', 'DetectionLayer', 'DataLayer'] // Fallback
        };

        const allowed = validConnections[sourceType] || [];
        const isValid = allowed.includes(targetType);

        if (!isValid) {
            // Store the invalid connection details for onConnectEnd to show toast
            lastInvalidConnection.current = {
                sourceType,
                targetType
            };
        }

        return isValid;
    }, [nodes, getLayerType, t]);



    const onConnect = useCallback((params) => {
        if (!reactFlowInstance) return;

        // Validate connection and show toast if invalid
        const sourceNode = reactFlowInstance.getNode(params.source);
        const targetNode = reactFlowInstance.getNode(params.target);

        if (!sourceNode || !targetNode) return;

        // Check for self-loop
        if (params.source === params.target) {
            showToast(`❌ ${t('validation.invalidConnection')} : ${t('validation.selfLoop')}`);
            return;
        }

        // Get layer types
        const getNodeLayerType = (node) => {
            if (!node.data.elements || node.data.elements.length === 0) return null;
            return getLayerType(node.data.elements[0].type);
        };

        const sourceType = getNodeLayerType(sourceNode);
        const targetType = getNodeLayerType(targetNode);

        // Validate if types are known
        if (sourceType && targetType) {
            const validConnections = {
                'GenerationLayer': ['GenerationLayer', 'SamplingLayer'],
                'SamplingLayer': ['OpticalLayer', 'DetectionLayer'],
                'OpticalLayer': ['OpticalLayer', 'DetectionLayer'],
                'DetectionLayer': ['DataLayer'],
                'DataLayer': ['DataLayer'],
                'Layer': ['Layer', 'GenerationLayer', 'SamplingLayer', 'OpticalLayer', 'DetectionLayer', 'DataLayer']
            };

            const allowed = validConnections[sourceType] || [];
            const isValid = allowed.includes(targetType);

            if (!isValid) {
                // Show toast with explanation
                const ruleExplanations = {
                    'GenerationLayer': t('validation.generationLayer'),
                    'SamplingLayer': t('validation.samplingLayer'),
                    'OpticalLayer': t('validation.opticalLayer'),
                    'DetectionLayer': t('validation.detectionLayer'),
                    'DataLayer': t('validation.dataLayer')
                };

                const explanation = ruleExplanations[sourceType] || `${sourceType} cannot connect to ${targetType}`;
                showToast(`❌ ${t('validation.invalidConnection')} : ${explanation}`);
                console.warn(`Invalid connection: ${sourceType} cannot connect to ${targetType}`);
                return; // Don't create the connection
            }
        }

        registerChange();

        // Edge Type Logic based on Layer Content
        let edgeType = 'default';
        let edgeData = { pathCount: 1 };

        if (sourceNode && sourceNode.data.elements) {
            const telescopeElem = sourceNode.data.elements.find(el => el.type === 'telescope');
            if (telescopeElem) {
                edgeType = 'parallel';
                edgeData = { pathCount: telescopeElem.config.collectors ? telescopeElem.config.collectors.length : 1 };
            }
        }

        setEdges((eds) => {
            // Remove existing connections TO the target (prevents multiple inputs)
            // AND remove existing connections FROM the source (prevents multiple outputs)
            const filtered = eds.filter(e =>
                e.target !== params.target && e.source !== params.source
            );
            return addEdge({ ...params, animated: true, type: edgeType, data: edgeData }, filtered);
        });
    }, [setEdges, reactFlowInstance, registerChange, showToast, getLayerType, t]);

    const onEdgeUpdateStart = useCallback(() => {
        edgeUpdateSuccessful.current = false;
    }, []);

    const onEdgeUpdate = useCallback((oldEdge, newConnection) => {
        edgeUpdateSuccessful.current = true;
        setEdges((els) => updateEdge(oldEdge, newConnection, els));
    }, [setEdges]);

    const onEdgeUpdateEnd = useCallback((_, edge) => {
        if (!edgeUpdateSuccessful.current) {
            setEdges((eds) => eds.filter((e) => e.id !== edge.id));
        }
        edgeUpdateSuccessful.current = true;
    }, [setEdges]);

    // Track connection attempt to show toast on failed connections
    const connectionAttempt = useRef(null);

    const onConnectStart = useCallback((event, { nodeId, handleType }) => {
        // Clear previous invalid connection tracking
        lastInvalidConnection.current = null;
    }, []);

    const onConnectEnd = useCallback((event) => {
        // Check if the last connection attempt was invalid
        if (lastInvalidConnection.current) {
            const { sourceType, targetType, message } = lastInvalidConnection.current;

            if (message) {
                // Self-loop or custom message
                showToast(`❌ ${t('validation.invalidConnection')} : ${message}`);
            } else {
                // Type-based validation error
                const ruleExplanations = {
                    'GenerationLayer': t('validation.generationLayer'),
                    'SamplingLayer': t('validation.samplingLayer'),
                    'OpticalLayer': t('validation.opticalLayer'),
                    'DetectionLayer': t('validation.detectionLayer'),
                    'DataLayer': t('validation.dataLayer')
                };

                const explanation = ruleExplanations[sourceType] || `${sourceType} cannot connect to ${targetType}`;
                showToast(`❌ ${t('validation.invalidConnection')} : ${explanation}`);
                console.warn(`Invalid connection attempt: ${sourceType} cannot connect to ${targetType}`);
            }

            lastInvalidConnection.current = null;
        }
    }, [showToast, t]);




    const onDragOver = useCallback((event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }, []);

    const onDrop = useCallback(
        (event) => {
            event.preventDefault();
            registerChange();

            const type = event.dataTransfer.getData('application/reactflow');
            if (typeof type === 'undefined' || !type) return;

            const clientX = event.clientX;
            const clientY = event.clientY;

            const position = reactFlowInstance.project({
                x: clientX - reactFlowWrapper.current.getBoundingClientRect().left,
                y: clientY - reactFlowWrapper.current.getBoundingClientRect().top,
            });

            // Check intersection with existing nodes
            const hitBox = { x: position.x - 20, y: position.y - 20, width: 40, height: 40 };
            const intersections = reactFlowInstance.getIntersectingNodes(hitBox);

            // Define new element
            // Icons already imported

            let newElement = { type, config: {}, label: type, iconPath: getElementIcon(type) };

            if (type === 'scene') newElement = { type, label: 'Scene', config: { stars, planets, zodiacal }, iconPath: getElementIcon('scene') };
            else if (type === 'atmosphere') newElement = { type, label: 'Atmosphere', config: atmosphere, iconPath: getElementIcon('atmosphere') };
            else if (type === 'telescope') newElement = { type, label: 'Telescope', config: telescope, iconPath: getElementIcon('telescope') };
            else if (type === 'camera') newElement = { type, label: 'Camera', config: camera, iconPath: getElementIcon('camera') };
            else if (type === 'lens') newElement = { type, label: 'Lens', config: { focal_length: 1.0 }, iconPath: getElementIcon('lens'), fields: [{ name: 'focal_length', type: 'number', label: 'Focal Length', step: 0.1 }] };
            else if (type === 'beam_splitter') newElement = { type, label: 'Beam Splitter', config: { split_ratio: 0.5 }, iconPath: getElementIcon('beam_splitter'), fields: [{ name: 'split_ratio', type: 'number', label: 'Split Ratio', step: 0.1 }] };
            else if (type === 'coronagraph') newElement = { type, label: 'Coronagraph', config: { type: '4quadrants' }, iconPath: getElementIcon('coronagraph'), fields: [{ name: 'type', type: 'select', label: 'Mask Type', options: [{ value: '4quadrants', label: '4-Quadrants' }, { value: 'vortex', label: 'Vortex' }] }] };
            else if (type === 'fiber_in') newElement = { type, label: 'Fiber Injection', config: { modes: 1 }, iconPath: getElementIcon('fiber_in'), fields: [{ name: 'modes', type: 'number', label: 'Modes', step: 1 }] };
            else if (type === 'fiber_out') newElement = { type, label: 'Fiber Output', config: {}, iconPath: getElementIcon('fiber_out'), fields: [] };
            else if (type === 'mmi') newElement = { type, label: 'MMI Coupler', config: { inputs: 2, outputs: 2 }, iconPath: getElementIcon('mmi'), fields: [{ name: 'inputs', type: 'number', label: 'Inputs', step: 1 }, { name: 'outputs', type: 'number', label: 'Outputs', step: 1 }] };


            if (intersections.length > 0) {
                // Add to existing Layer
                const targetNode = intersections[0]; // Take top one
                const newElements = [...(targetNode.data.elements || []), newElement];

                setNodes((nds) => nds.map((n) => {
                    if (n.id === targetNode.id) {
                        return { ...n, data: { ...n.data, elements: newElements } };
                    }
                    return n;
                }));
            } else {
                // Create New Layer Node
                const nodeId = getId();
                const newNode = {
                    id: nodeId,
                    type: 'layer',
                    position,
                    data: {
                        elements: [newElement]

                    }
                };
                setNodes((nds) => nds.concat(newNode));
            }
        },
        [reactFlowInstance, registerChange, setNodes, stars, telescope, camera, atmosphere]
    );

    const getPipeline = () => {
        // Traversal Logic - Refactored for Layers
        // We traverse Layers. For each Layer, we just append its Elements to the chain.
        // Parallel branching logic now applies to LAYERS.

        const startNodes = nodes.filter(n =>
            n.data.elements && n.data.elements.some(el => el.type === 'scene')
        );
        const root = startNodes.length > 0 ? startNodes[0] : nodes[0]; // Fallback

        if (!root) return [];

        let visited = new Set();

        const buildChain = (node) => {
            // 1. Current Layer Elements
            // We need to return a List of Configs.
            const layerElements = node.data.elements.map(el => ({
                type: el.type,
                config: el.config,
                metadata: { position: node.position }
            }));

            // 2. Children
            const outEdges = edges.filter(e => e.source === node.id);
            const targets = outEdges
                .map(e => nodes.find(n => n.id === e.target))
                .filter(n => n && !visited.has(n.id));

            targets.forEach(t => visited.add(t.id));

            if (targets.length === 0) return [layerElements];
            else if (targets.length === 1) {
                // Linear connection to next Layer
                return [layerElements, ...buildChain(targets[0])];
            } else {
                // Parallel Layers from this Layer
                const branches = targets.map(t => {
                    const res = buildChain(t);
                    return res;
                });
                return [layerElements, branches];
            }
        };

        visited.add(root.id);
        const configList = buildChain(root);
        return configList;
    };

    const handleRun = () => {
        const pipeline = getPipeline();
        runSimulation(pipeline);
    };

    const handleExport = async () => {
        try {
            const pipeline = getPipeline();
            // Wrap in payload
            const payload = { mode: 'pipeline', layers: pipeline };
            console.log("Export Payload:", JSON.stringify(payload, null, 2));

            const response = await fetch('/api/pipeline/export_file', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error("Export failed");

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = "helios_pipeline.json";
            document.body.appendChild(a);
            a.click();

            // Delay revocation to ensure download starts
            setTimeout(() => {
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            }, 1000);
        } catch (e) {
            console.error(e);
            alert("Export Failed: " + e.message);
        }
    };

    const handleGetCode = async () => {
        try {
            const pipeline = getPipeline();
            const payload = { mode: 'pipeline', layers: pipeline };

            const response = await fetch('/api/generate_code', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error("Code generation failed");

            const data = await response.json();
            setInspectData({ code: data.code, title: "Python Code" });
        } catch (err) {
            console.error(err);
            alert("Failed to generate code");
        }
    };

    // Keyboard Listeners
    useEffect(() => {
        const handleKeyDown = (e) => {
            // Check modifier (Ctrl or Cmd)
            const meta = e.ctrlKey || e.metaKey;

            if (meta && e.key === 'z') {
                e.preventDefault();
                undo();
            } else if (meta && e.key === 'y') {
                e.preventDefault();
                redo();
            } else if (meta && e.key === 'c') {
                // Let native copy work for text? Only if no input focused.
                if (document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
                    e.preventDefault();
                    handleCopy();
                }
            } else if (meta && e.key === 'v') {
                if (document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
                    e.preventDefault();
                    handlePaste();
                }
            } else if (meta && e.key === 's') {
                e.preventDefault();
                handleExport(); // Use existing export logic
            } else if (meta && e.key === 'a') {
                e.preventDefault();
                setNodes(nds => nds.map(n => ({ ...n, selected: true })));
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [undo, redo, handleCopy, handlePaste]); // handleExport needs to be stable or ref

    const fileInputRef = useRef(null);

    const handleImportClick = () => {
        fileInputRef.current.click();
    };

    const handleFileChange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        try {
            const text = await file.text();
            const jsonData = JSON.parse(text);

            const response = await fetch('/api/pipeline/import_file', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(jsonData)
            });

            if (!response.ok) throw new Error("Import failed");

            const result = await response.json();
            const layers = result.layers || [];

            const newNodes = [];
            const newEdges = [];

            // Helper to recursively process layers
            const processLayerList = (list, parentId, autoX) => {
                const items = Array.isArray(list) ? list : [list];
                let currentX = autoX;
                let lastId = parentId;

                items.forEach((item, idx) => {
                    if (Array.isArray(item)) {
                        // Parallel Block
                        const branches = item;
                        branches.forEach(branch => {
                            processLayerList(branch, lastId, currentX);
                        });
                        // X position doesn't increment for parallel branches typically, 
                        // but logic needs review if they merge back. For now, simple recursion.
                    } else {
                        // Single Layer
                        const layer = item;
                        const id = "node_imp_" + newNodes.length;

                        let position = { x: currentX, y: 100 + (Math.random() * 50) };
                        if (layer.metadata && layer.metadata.position) {
                            position = layer.metadata.position;
                            if (position.x >= currentX) currentX = position.x + 400;
                        } else {
                            const pNode = newNodes.find(n => n.id === parentId);
                            if (parentId && pNode) {
                                position.y = pNode.position.y;
                            }
                            currentX += 450;
                        }

                        let data = { elements: [] };
                        const type = layer.type;
                        const conf = layer.config || {};

                        // Map to Elements
                        if (type === 'scene') {
                            if (conf.stars) setStars(conf.stars);
                            if (conf.planets) setPlanets(conf.planets);
                            if (conf.zodiacal) setZodiacal(conf.zodiacal);
                            data.elements.push({ type, label: 'Scene', config: { stars: conf.stars, planets: conf.planets, zodiacal: conf.zodiacal }, iconPath: getElementIcon('scene') });
                        } else if (type === 'atmosphere') {
                            setAtmosphere(conf);
                            data.elements.push({ type, label: 'Atmosphere', config: conf, iconPath: getElementIcon('atmosphere') });
                        } else if (type === 'telescope') {
                            setTelescope(conf);
                            data.elements.push({ type, label: 'Telescope', config: conf, iconPath: getElementIcon('telescope') });
                        } else if (type === 'camera') {
                            setCamera(conf);
                            data.elements.push({ type, label: 'Camera', config: conf, iconPath: getElementIcon('camera') });
                        } else {
                            // Generics
                            let element = { type, config: conf, label: type, iconPath: getElementIcon(type) };
                            if (type === 'lens') element = { ...element, label: 'Lens', iconPath: getElementIcon('lens'), fields: [{ name: 'focal_length', type: 'number', label: 'Focal Length', step: 0.1 }] };
                            // Add more specific types as needed
                            data.elements.push(element);
                        }

                        newNodes.push({ id, type: 'layer', position, data });

                        if (parentId) {
                            const edge = {
                                id: "e_" + parentId + "_" + id,
                                source: parentId,
                                target: id,
                                animated: true,
                                type: 'default',
                                data: { pathCount: 1 }
                            };

                            // Check for parallel edge based on parent being Telescope
                            const pNode = newNodes.find(n => n.id === parentId);
                            if (pNode && pNode.data.elements) {
                                const tel = pNode.data.elements.find(el => el.type === 'telescope');
                                if (tel) {
                                    edge.type = 'parallel';
                                    edge.data.pathCount = tel.config.collectors ? tel.config.collectors.length : 1;
                                }
                            }
                            newEdges.push(edge);
                        }
                        lastId = id;
                    }
                });
            };

            processLayerList(layers, null, 50);

            setNodes(newNodes);
            setEdges(newEdges);
            e.target.value = null; // Reset input

        } catch (err) {
            console.error(err);
            alert("Import Failed: " + err.message);
        }
    };

    return (
        <div className="flex h-full flex-col">
            {/* TOP BAR */}
            <div className={"h-14 border-b flex items-center px-4 justify-between transition-colors " + (isDark ? 'bg-slate-900 border-slate-700' : 'bg-white border-slate-200')}>
                <div className="flex items-center gap-4">
                    <button
                        onClick={onToggleSidebar}
                        className={"p-2 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors " + (isDark ? 'text-slate-300' : 'text-slate-600')}
                        title={t('toolbar.toggleSidebar')}
                    >
                        <Menu className="w-5 h-5" />
                    </button>
                    <h1 className="text-xl font-bold bg-gradient-to-r from-blue-500 to-indigo-500 bg-clip-text text-transparent">
                        HELIOS <span className="text-xs font-normal text-slate-500 inline-block ml-2 align-middle">{t('app.subtitle')}</span>
                    </h1>
                </div>

                <div className="flex items-center gap-2">
                    {/* Support Button */}
                    <a
                        href="https://paypal.me/vincentforiel"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-pink-500 hover:text-pink-600"
                        title={t('toolbar.support')}
                    >
                        <Heart className="w-5 h-5 fill-current" />
                    </a>

                    {/* GitHub Button */}
                    <a
                        href="https://github.com/vforiel/helios"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-slate-600 dark:text-slate-300 hover:text-black dark:hover:text-white"
                        title={t('toolbar.github')}
                    >
                        <Github className="w-5 h-5" />
                    </a>

                    {/* Documentation Button */}
                    <a
                        href="http://helios-project.rtfd.io/"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400"
                        title={t('toolbar.documentation')}
                    >
                        <Book className="w-5 h-5" />
                    </a>

                    <div className="w-px h-6 bg-slate-200 dark:bg-slate-700 mx-1"></div>

                    {/* Import/Export */}
                    <input
                        type="file"
                        ref={fileInputRef}
                        style={{ display: 'none' }}
                        accept=".json"
                        onChange={handleFileChange}
                    />

                    <button
                        onClick={handleImportClick}
                        className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400"
                        title={t('toolbar.import')}
                    >
                        <Upload className="w-5 h-5" />
                    </button>

                    <button
                        onClick={handleExport}
                        className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400"
                        title={t('toolbar.export')}
                    >
                        <Download className="w-5 h-5" />
                    </button>

                    <button
                        onClick={handleGetCode}
                        className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400"
                        title={t('toolbar.getCode')}
                    >
                        <Code className="w-5 h-5" />
                    </button>

                    <div className="w-px h-6 bg-slate-200 dark:bg-slate-700 mx-1"></div>
                    <button
                        onClick={onToggleTheme}
                        className={"p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors " + (isDark ? 'text-yellow-400' : 'text-slate-600')}
                        title={t('toolbar.toggleTheme')}
                    >
                        {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                    </button>

                    {/* Language Selector */}
                    <div className="relative">
                        <button
                            onClick={() => setIsLanguageDropdownOpen(!isLanguageDropdownOpen)}
                            className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 flex items-center gap-1"
                            title={t('toolbar.language')}
                        >
                            <Languages className="w-5 h-5" />
                            <span className="text-xs font-semibold">
                                {languages.find(l => l.code === language)?.flag || '🌐'}
                            </span>
                        </button>

                        {/* Language Dropdown */}
                        {isLanguageDropdownOpen && (
                            <>
                                {/* Backdrop to close dropdown */}
                                <div
                                    className="fixed inset-0 z-40"
                                    onClick={() => setIsLanguageDropdownOpen(false)}
                                ></div>

                                {/* Dropdown Menu */}
                                <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-slate-800 rounded-lg shadow-xl border border-slate-200 dark:border-slate-700 z-50 overflow-hidden">
                                    {languages.map((lang) => (
                                        <button
                                            key={lang.code}
                                            onClick={() => {
                                                setLanguage(lang.code);
                                                setIsLanguageDropdownOpen(false);
                                            }}
                                            className={`w-full px-4 py-2.5 text-left flex items-center gap-3 transition-colors ${language === lang.code
                                                ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400'
                                                : 'hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300'
                                                }`}
                                        >
                                            <span className="text-xl">{lang.flag}</span>
                                            <div className="flex-1">
                                                <div className="font-medium text-sm">{lang.nativeName}</div>
                                                <div className="text-xs opacity-60">{lang.name}</div>
                                            </div>
                                            {language === lang.code && (
                                                <div className="w-2 h-2 rounded-full bg-blue-600 dark:bg-blue-400"></div>
                                            )}
                                        </button>
                                    ))}
                                </div>
                            </>
                        )}
                    </div>

                    <button
                        onClick={handleRun}
                        className="bg-blue-600 hover:bg-blue-500 text-white px-5 py-2 rounded-lg text-sm font-semibold shadow-md transition-transform active:scale-95 flex items-center"
                    >
                        {t('toolbar.runPipeline')}
                    </button>
                </div>
            </div>

            <div className="flex-1" ref={reactFlowWrapper}>
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    onConnectStart={onConnectStart}
                    onConnectEnd={onConnectEnd}
                    onEdgeUpdate={onEdgeUpdate}
                    onEdgeUpdateStart={onEdgeUpdateStart}
                    onEdgeUpdateEnd={onEdgeUpdateEnd}
                    deleteKeyCode={['Backspace', 'Delete']}
                    isValidConnection={isValidConnection}
                    onInit={setReactFlowInstance}
                    onDrop={onDrop}
                    onDragOver={onDragOver}
                    nodeTypes={nodeTypes}
                    edgeTypes={edgeTypes}
                    connectionLineType="smoothstep"
                    selectionOnDrag={interactionMode === 'select'}
                    panOnDrag={interactionMode === 'nav' || [1, 2]}
                    selectionMode="partial"
                    fitView
                    minZoom={0.1}
                >
                    <Controls>
                        <ControlButton
                            onClick={() => setInteractionMode('nav')}
                            title="Navigation Mode (Pan)"
                            className={interactionMode === 'nav' ? 'text-blue-600 bg-blue-100 font-bold' : ''}
                        >
                            <Hand className="w-4 h-4 p-0.5" />
                        </ControlButton>
                        <ControlButton
                            onClick={() => setInteractionMode('select')}
                            title="Selection Mode (Box Select)"
                            className={interactionMode === 'select' ? 'text-blue-600 bg-blue-100 font-bold' : ''}
                        >
                            <MousePointer2 className="w-4 h-4 p-0.5" />
                        </ControlButton>
                    </Controls>
                    <Background color={isDark ? "#1e293b" : "#e2e8f0"} gap={16} />
                </ReactFlow>

                {/* INSPECT MODAL */}
                {(inspectLoading || inspectData) && (
                    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm">
                        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-2xl border border-slate-200 dark:border-slate-700 p-4 max-w-2xl w-full mx-4 flex flex-col gap-4 relative">
                            <button
                                onClick={() => { setInspectData(null); setInspectLoading(false); }}
                                className="absolute top-2 right-2 p-1 rounded hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
                            >
                                <X className="w-5 h-5 text-slate-500" />
                            </button>

                            <h2 className="text-lg font-bold text-slate-800 dark:text-slate-100">
                                {inspectLoading ? "Inspecting Layer..." : inspectData?.title || "Inspection Result"}
                            </h2>

                            <div className={`flex items-center justify-center bg-slate-50 dark:bg-slate-900 rounded border border-slate-200 dark:border-slate-700 overflow-hidden ${inspectData?.code ? 'h-[500px]' : 'min-h-[300px]'}`}>
                                {inspectLoading ? (
                                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                                ) : inspectData?.code ? (
                                    <CodeViewer code={inspectData.code} />
                                ) : (
                                    <img src={inspectData.image} alt="Inspection" className="max-h-[500px] object-contain" />
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {/* Toast Notifications - Bottom Right */}
                <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-md">
                    {toasts.map(toast => (
                        <div
                            key={toast.id}
                            className={`
                                px-4 py-3 rounded-lg shadow-lg border
                                ${toast.type === 'error'
                                    ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-800 dark:text-red-200'
                                    : 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800 text-blue-800 dark:text-blue-200'
                                }
                                animate-slide-in-right
                                backdrop-blur-sm
                            `}
                            style={{
                                animation: 'slideInRight 0.3s ease-out'
                            }}
                        >
                            <div className="flex items-start gap-2">
                                <div className="flex-1 text-sm font-medium">
                                    {toast.message}
                                </div>
                                <button
                                    onClick={() => setToasts(prev => prev.filter(t => t.id !== toast.id))}
                                    className="text-current opacity-50 hover:opacity-100 transition-opacity"
                                >
                                    <X className="w-4 h-4" />
                                </button>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Add keyframes for animation */}
                <style>{`
                    @keyframes slideInRight {
                        from {
                            transform: translateX(100%);
                            opacity: 0;
                        }
                        to {
                            transform: translateX(0);
                            opacity: 1;
                        }
                    }
                `}</style>
            </div>
        </div >
    );
}
