import React, { useState, useRef, useEffect } from 'react';
import { Eye, X, Download, Maximize2, RefreshCw, Move, Settings2 } from 'lucide-react';

export default function LayerVisualizer({ type, config }) {
    const [showPreview, setShowPreview] = useState(false);
    const [image, setImage] = useState(null);
    const [filename, setFilename] = useState(`${type}_preview.png`);
    const [loading, setLoading] = useState(false);
    const [viewMode, setViewMode] = useState(type === 'camera' ? 'processed' : 'geometry'); // 'geometry' or 'sed' for scene, 'processed'/'raw'/'dark' for camera

    // Figsize state: [width, height] in inches
    const [figSize, setFigSize] = useState([6, 6]);
    const [showSizePopover, setShowSizePopover] = useState(false);
    const popoverRef = useRef(null);

    // Close popover on outside click
    useEffect(() => {
        function handleClickOutside(event) {
            if (popoverRef.current && !popoverRef.current.contains(event.target)) {
                setShowSizePopover(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, []);

    const handlePreview = async () => {
        setShowPreview(true);
        // Always fetch fresh data on user request

        if (image) {
            URL.revokeObjectURL(image);
            setImage(null);
        }

        setLoading(true);
        try {
            // Inject view_mode and figsize into config
            const effectiveConfig = {
                ...config,
                view_mode: viewMode,
                figsize: figSize
            };

            const payload = { type, config: effectiveConfig };
            const response = await fetch('/api/preview_layer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (response.ok) {
                const blob = await response.blob();

                // Try to get filename from header
                try {
                    const disposition = response.headers.get('Content-Disposition');
                    let name = `${type}_preview.png`;
                    if (disposition && disposition.indexOf('filename=') !== -1) {
                        const matches = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(disposition);
                        if (matches != null && matches[1]) {
                            name = matches[1].replace(/['"]/g, '');
                        }
                    }
                    setFilename(name);
                } catch (err) {
                    console.warn("Filename parsing error:", err);
                }

                // Explicitly force image/png type to ensure download works correctly
                const typedBlob = new Blob([blob], { type: 'image/png' });
                setImage(URL.createObjectURL(typedBlob));
            } else {
                console.error("Preview failed");
            }
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    // Re-fetch when toggling mode
    useEffect(() => {
        if (showPreview && (type === 'scene' || type === 'camera') && !loading) {
            handlePreview();
        }
    }, [viewMode]);

    return (
        <>
            <button
                onClick={handlePreview}
                className="p-1 rounded hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
                title="Visualize Layer"
            >
                <Eye className="w-4 h-4" />
            </button>

            {showPreview && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4" onClick={() => setShowPreview(false)}>
                    <div className="bg-white dark:bg-slate-900 rounded-lg shadow-2xl border border-slate-200 dark:border-slate-700 max-w-3xl w-full flex flex-col overflow-hidden" onClick={e => e.stopPropagation()}>
                        {/* Header */}
                        <div className="flex items-center justify-between p-3 border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950">
                            <div className="flex items-center gap-2 flex-wrap">
                                <h3 className="font-semibold text-slate-800 dark:text-slate-200 capitalize whitespace-nowrap">
                                    {type === 'camera'
                                        ? `Camera - ${viewMode === 'processed' ? 'Processed' : viewMode === 'raw' ? 'Raw' : 'Dark'}`
                                        : `${type} Visualization`}
                                </h3>
                                {type === 'scene' && (
                                    <div className="flex bg-slate-200 dark:bg-slate-800 rounded p-1">
                                        <button
                                            onClick={() => setViewMode('geometry')}
                                            className={`px-3 py-1 text-xs font-medium rounded transition-all ${viewMode === 'geometry'
                                                ? 'bg-white dark:bg-slate-700 text-blue-600 dark:text-blue-400 shadow-sm'
                                                : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                                                }`}
                                        >
                                            Geometry
                                        </button>
                                        <button
                                            onClick={() => setViewMode('sed')}
                                            className={`px-3 py-1 text-xs font-medium rounded transition-all ${viewMode === 'sed'
                                                ? 'bg-white dark:bg-slate-700 text-blue-600 dark:text-blue-400 shadow-sm'
                                                : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                                                }`}
                                        >
                                            SED
                                        </button>
                                    </div>
                                )}
                                {type === 'camera' && (
                                    <div className="flex bg-slate-200 dark:bg-slate-800 rounded p-1">
                                        <button
                                            onClick={() => setViewMode('processed')}
                                            className={`px-2 py-1 text-xs font-medium rounded transition-all ${viewMode === 'processed'
                                                ? 'bg-white dark:bg-slate-700 text-pink-600 dark:text-pink-400 shadow-sm'
                                                : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                                                }`}
                                        >
                                            Processed
                                        </button>
                                        <button
                                            onClick={() => setViewMode('raw')}
                                            className={`px-2 py-1 text-xs font-medium rounded transition-all ${viewMode === 'raw'
                                                ? 'bg-white dark:bg-slate-700 text-pink-600 dark:text-pink-400 shadow-sm'
                                                : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                                                }`}
                                        >
                                            Raw
                                        </button>
                                        <button
                                            onClick={() => setViewMode('dark')}
                                            className={`px-2 py-1 text-xs font-medium rounded transition-all ${viewMode === 'dark'
                                                ? 'bg-white dark:bg-slate-700 text-pink-600 dark:text-pink-400 shadow-sm'
                                                : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                                                }`}
                                        >
                                            Dark
                                        </button>
                                    </div>
                                )}
                            </div>
                            <button onClick={() => setShowPreview(false)} className="text-slate-500 hover:text-slate-700 dark:hover:text-white transition-colors">
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        {/* Image Area */}
                        <div className="p-4 flex items-center justify-center min-h-[300px] bg-slate-100 dark:bg-black/20 overflow-auto">
                            {loading ? (
                                <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
                            ) : image ? (
                                <img src={image} alt="Preview" className="max-h-[600px] object-contain rounded shadow-sm" />
                            ) : (
                                <div className="text-slate-500">Failed to load preview</div>
                            )}
                        </div>

                        {/* Footer Controls */}
                        <div className="p-3 border-t border-slate-200 dark:border-slate-800 flex justify-between items-center bg-slate-50 dark:bg-slate-950">
                            <div className="flex items-center gap-2">
                                {/* Refresh */}
                                <button
                                    onClick={handlePreview}
                                    className="p-2 rounded hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-400 transition-colors"
                                    title="Refresh"
                                >
                                    <RefreshCw className="w-4 h-4" />
                                </button>

                                {/* Figsize Popover Trigger */}
                                <div className="relative" ref={popoverRef}>
                                    <button
                                        onClick={() => setShowSizePopover(!showSizePopover)}
                                        className={`p-2 rounded transition-colors flex items-center gap-1 ${showSizePopover
                                            ? 'bg-slate-200 dark:bg-slate-800 text-blue-600 dark:text-blue-400'
                                            : 'hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-400'}`}
                                        title="Image Dimensions"
                                    >
                                        <Move className="w-4 h-4" />
                                    </button>

                                    {showSizePopover && (
                                        <div className="absolute bottom-full left-0 mb-2 p-3 bg-white dark:bg-slate-900 rounded-lg shadow-xl border border-slate-200 dark:border-slate-700 w-48 z-50">
                                            <h4 className="text-xs font-semibold text-slate-500 uppercase mb-2">Figure Size (inches)</h4>
                                            <div className="grid grid-cols-2 gap-2">
                                                <div>
                                                    <label className="text-[10px] text-slate-400 block mb-1">Width</label>
                                                    <input
                                                        type="number"
                                                        min="2"
                                                        max="20"
                                                        value={figSize[0]}
                                                        onChange={(e) => setFigSize([Number(e.target.value), figSize[1]])}
                                                        className="w-full px-2 py-1 text-sm bg-slate-100 dark:bg-slate-800 rounded border border-slate-200 dark:border-slate-700 outline-none focus:border-blue-500"
                                                    />
                                                </div>
                                                <div>
                                                    <label className="text-[10px] text-slate-400 block mb-1">Height</label>
                                                    <input
                                                        type="number"
                                                        min="2"
                                                        max="20"
                                                        value={figSize[1]}
                                                        onChange={(e) => setFigSize([figSize[0], Number(e.target.value)])}
                                                        className="w-full px-2 py-1 text-sm bg-slate-100 dark:bg-slate-800 rounded border border-slate-200 dark:border-slate-700 outline-none focus:border-blue-500"
                                                    />
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>

                            <div className="flex gap-2">
                                {image && (
                                    <>
                                        <a
                                            href={image}
                                            download={filename}
                                            className="p-2 bg-blue-600 hover:bg-blue-500 text-white rounded transition-colors shadow-sm"
                                            title={`Download ${filename}`}
                                        >
                                            <Download className="w-4 h-4" />
                                        </a>
                                        <a
                                            href={image}
                                            target="_blank"
                                            rel="noreferrer"
                                            className="p-2 bg-slate-200 dark:bg-slate-800 hover:bg-slate-300 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 rounded transition-colors"
                                            title="Open Full Size"
                                        >
                                            <Maximize2 className="w-4 h-4" />
                                        </a>
                                    </>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}
