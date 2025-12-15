import React, { useState, useCallback, useRef, useMemo, useEffect } from 'react';
import ReactFlow, {
    useReactFlow,
    Controls,
    ControlButton,
    Background,
    addEdge,
    updateEdge
} from 'reactflow';
import 'reactflow/dist/style.css';
import TelescopeNode from './nodes/TelescopeNode';
import TelescopeArrayNode from './nodes/TelescopeArrayNode'; // Imported
import CameraNode from './nodes/CameraNode';
import GenericNode from './nodes/GenericNode';
import { Menu, Sun, Moon, Heart, Github, Book, Download, Upload, Cpu, Disc, Divide, GitFork, Zap, Activity, Hand, MousePointer2, Stars, Search, Camera, CloudFog, X, Code, Languages, LayoutList, GitGraph, Grid } from 'lucide-react'; // Added Grid
import { getElementIcon, getPhotonicIcon } from '../../utils/iconMap';

import LayerNode from './nodes/LayerNode';
import ParallelEdge from './edges/ParallelEdge';
import CodeViewer from '../CodeViewer';
import SimulationControls from '../SimulationControls';
import LayeredView from '../LayeredView';

const nodeTypes = {
    layer: LayerNode,
    telescope_array: TelescopeArrayNode // Registered (though we use LayerNode wrapping usually? Wait, checking code...)
};
// Wait, the previous code showed TelescopeNode is NOT in nodeTypes?
// Line 23: const nodeTypes = { layer: LayerNode };
// It seems nodes are rendered INSIDE LayerNode? Or are TelescopeNode/CameraNode used directly?
// Looking at onDrop (line 573):
// It creates "newElement" which is added to a "layer" node's data.elements.
// Or it creates a new "layer" node containing the element.
// So LayerNode renders the specific component based on type?
// Let's check LayerNode.jsx.

const edgeTypes = {
    parallel: ParallelEdge
};

let id = 1;
const getId = () => `node_${id++}`;

// History Helper
const useUndoRedo = (nodes, edges, setNodes, setEdges) => {
    const [past, setPast] = useState([]);
    const [future, setFuture] = useState([]);

    const takeSnapshot = useCallback(() => {
        setPast(old => {
            const newPast = [...old, { nodes, edges }];
            if (newPast.length > 50) newPast.shift(); // Limit to 50
            return newPast;
        });
        setFuture([]);
    }, [nodes, edges]);

    const canUndo = past.length > 0;
    const canRedo = future.length > 0;

    const undo = useCallback(() => {
        if (!canUndo) return;
        const current = { nodes, edges };
        const previous = past[past.length - 1];
        const newPast = past.slice(0, past.length - 1);

        setPast(newPast);
        setFuture([current, ...future]);
        setNodes(previous.nodes);
        setEdges(previous.edges);
    }, [nodes, edges, past, future, canUndo, setNodes, setEdges]); // Added dependencies

    const redo = useCallback(() => {
        if (!canRedo) return;
        const current = { nodes, edges };
        const next = future[0];
        const newFuture = future.slice(1);

        setPast([...past, current]);
        setFuture(newFuture);
        setNodes(next.nodes);
        setEdges(next.edges);
    }, [nodes, edges, past, future, canRedo, setNodes, setEdges]);

    return { past, setPast, future, setFuture, takeSnapshot, canUndo, canRedo, undo, redo };
};

export default function PipelineEditor({
    nodes, setNodes, onNodesChange,
    edges, setEdges, onEdgesChange,
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
    t,
    viewMode,
    setViewMode
}) {
    const reactFlowWrapper = useRef(null);
    const edgeUpdateSuccessful = useRef(true);
    const [reactFlowInstance, setReactFlowInstance] = useState(null);
    const [clipboard, setClipboard] = useState(null);
    const [interactionMode, setInteractionMode] = useState('nav'); // 'nav' | 'select'

    // Inspect Modal State
    const [inspectData, setInspectData] = useState(null);
    const [inspectLoading, setInspectLoading] = useState(false);

    // Language Dropdown State
    const [isLanguageDropdownOpen, setIsLanguageDropdownOpen] = useState(false);

    // Toast Notification State
    const [toasts, setToasts] = useState([]);

    // Bottom Bar State
    const [isBottomBarExpanded, setIsBottomBarExpanded] = useState(false);

    const showToast = useCallback((message, type = 'error') => {
        const id = Date.now();
        setToasts(prev => [...prev, { id, message, type }]);
        setTimeout(() => {
            setToasts(prev => prev.filter(t => t.id !== id));
        }, 5000);
    }, []);

    // Sync Logic... (Kept as is but removing local state)
    useEffect(() => {
        import('../../utils/optical_logic').then(({ propagateSignals }) => {
            const { nodes: newNodes, edges: newEdges } = propagateSignals(nodes, edges);

            let nodesChanged = false;
            let edgesChanged = false;

            const updatedEdges = newEdges.map((ne, i) => {
                const old = edges.find(e => e.id === ne.id);
                if (!old || (old.data?.pathCount !== ne.data.pathCount) || (old.type !== ne.type)) {
                    edgesChanged = true;
                    return ne;
                }
                return old;
            });

            const updatedNodes = newNodes.map((nn, i) => {
                const old = nodes.find(n => n.id === nn.id);
                const newIO = nn.data.io;
                const oldIO = old?.data?.io;

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
        nodes.map(n => JSON.stringify(n.data.elements)).join(','),
        edges.map(e => e.source + e.target).join(','),
        setNodes,
        setEdges
    ]);

    // Undo/Redo Hook (Passed props)
    const { takeSnapshot, undo, redo } = useUndoRedo(nodes, edges, setNodes, setEdges);

    // Snapshot Trigger
    const registerChange = useCallback(() => {
        takeSnapshot();
    }, [takeSnapshot]);

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



    // Refs for stable access in callbacks to prevent dependency loops
    const nodesRef = useRef(nodes);
    const edgesRef = useRef(edges);
    useEffect(() => {
        nodesRef.current = nodes;
        edgesRef.current = edges;
    }, [nodes, edges]);

    // Inspect Logic
    const handleInspect = useCallback(async (nodeId) => {
        const currentNodes = nodesRef.current;
        const currentEdges = edgesRef.current;

        try {
            setInspectLoading(true);
            setInspectData(null);

            // Re-implement getPipeline logic locally using refs to avoid dependency on the function
            // Traversal Logic
            const getPipelineSnapshot = () => {
                const startNodes = currentNodes.filter(n =>
                    n.data.elements && n.data.elements.some(el => el.type === 'scene')
                );
                const root = startNodes.length > 0 ? startNodes[0] : currentNodes[0];

                if (!root) return [];

                let visited = new Set();
                const buildChain = (node) => {
                    const layerElements = node.data.elements.map(el => ({
                        type: el.type,
                        config: el.config,
                        metadata: { position: node.position }
                    }));

                    const outEdges = currentEdges.filter(e => e.source === node.id);
                    const targets = outEdges
                        .map(e => currentNodes.find(n => n.id === e.target))
                        .filter(n => n && !visited.has(n.id));

                    targets.forEach(t => visited.add(t.id));

                    if (targets.length === 0) return [layerElements];
                    else if (targets.length === 1) return [layerElements, ...buildChain(targets[0])];
                    else {
                        const branches = targets.map(t => buildChain(t));
                        return [layerElements, branches];
                    }
                };

                visited.add(root.id);
                return buildChain(root);
            };

            // Traversal for Index Finding
            const startNodes = currentNodes.filter(n =>
                n.data.elements && n.data.elements.some(el => el.type === 'scene')
            );
            const root = startNodes.length > 0 ? startNodes[0] : currentNodes[0];

            if (!root) throw new Error("No Scene Found");

            let elementIndex = 0;
            let targetElementIndex = -1;
            let visited = new Set();

            const traverseElements = (node) => {
                if (!node) return;
                visited.add(node.id);

                const nodeElements = node.data.elements || [];
                if (node.id === nodeId) {
                    targetElementIndex = elementIndex + nodeElements.length - 1;
                }

                elementIndex += nodeElements.length;

                const outEdges = currentEdges.filter(e => e.source === node.id);
                if (outEdges.length > 0) {
                    const nextNodes = outEdges
                        .map(e => currentNodes.find(n => n.id === e.target))
                        .filter(n => n && !visited.has(n.id));
                    nextNodes.forEach(traverseElements);
                }
            };

            traverseElements(root);

            console.log(`Inspect: Node ${nodeId} -> Element Index ${targetElementIndex}`);

            // Send request
            const finalPipeline = getPipelineSnapshot();
            // We need to flatten the pipeline structure for the 'layers' argument if backend expects usage of it?
            // Existing logic passed 'flatPipelineConfig' = finalPipeline.flat(Infinity).
            const flatPipelineConfig = finalPipeline.flat(Infinity);

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
    }, []); // No dependencies, stable!

    // Update node data when props change + Inspect Handler
    // Changed from useMemo to useEffect to avoid side-effects during render
    useEffect(() => {
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

                // Only return new object if something actually changed? 
                // Hard to check deep equality cheaply. 
                // Relying on stable references of `stars`, `atmosphere`, `handleInspect`.
                return { ...node, data: newData };
            })
        );
    }, [stars, planets, zodiacal, atmosphere, telescope, camera, setNodes, t, handleInspect]);




    // Layer type mapping (must match backend)
    const getLayerType = useCallback((elementType) => {
        const mapping = {
            'scene': 'GenerationLayer',
            'atmosphere': 'GenerationLayer',
            'telescope': 'SamplingLayer',
            'telescope_array': 'SamplingLayer',
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
            else if (type === 'telescope_array') newElement = { type, label: 'Telescope Array', config: { ...telescope, preset: 'VLTI-UT' }, iconPath: getElementIcon('telescope') }; // Use Telescope icon for now or Grid?
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
                    {/* View Switcher */}
                    <div className="flex bg-slate-100 dark:bg-slate-800 rounded-lg p-1 border border-slate-200 dark:border-slate-700 mr-2">
                        <button
                            onClick={() => setViewMode('graph')}
                            className={`p-1.5 rounded transition-colors ${viewMode === 'graph' ? 'bg-white dark:bg-slate-700 shadow text-blue-600 dark:text-blue-400' : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'}`}
                            title={t ? t('toolbar.graphView') : "Graph View"} // Fallback if translation missing
                        >
                            <GitGraph className="w-5 h-5" />
                        </button>
                        <button
                            onClick={() => setViewMode('layered')}
                            className={`p-1.5 rounded transition-colors ${viewMode === 'layered' ? 'bg-white dark:bg-slate-700 shadow text-blue-600 dark:text-blue-400' : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'}`}
                            title={t ? t('toolbar.layeredView') : "Layered View"}
                        >
                            <LayoutList className="w-5 h-5" />
                        </button>
                    </div>

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

            <div className="flex-1 relative" ref={reactFlowWrapper}>
                {viewMode === 'graph' ? (
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
                        <Controls style={{ bottom: isBottomBarExpanded ? '70px' : '10px', transition: 'bottom 0.3s' }}>
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
                ) : (
                    <LayeredView
                        nodes={nodes}
                        setNodes={setNodes}
                        edges={edges}
                        setEdges={setEdges}
                        isDark={isDark}
                    />
                )}

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

            {/* Simulation Controls Bottom Bar */}
            <SimulationControls onExpandChange={setIsBottomBarExpanded} />
        </div >
    );
}
